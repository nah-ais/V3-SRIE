"""
data_processor.py
------------------
Helper untuk operasi data level-aplikasi:
  - Menyiapkan dataset gabungan (Login + Register) untuk matching.
  - Menerapkan keputusan reviewer (Simpan A/B, Keep keduanya).
  - Append data Register -> Login.
  - Export hasil akhir ke CSV/Excel.

Modul ini murni memanipulasi pandas DataFrame; tidak ada elemen UI di sini
supaya mudah di-unit-test terpisah dari Streamlit.
"""

from __future__ import annotations

import io
import pandas as pd

import config


def apply_review_decision(
    df: pd.DataFrame,
    pair_row: pd.Series,
    decision: str,
) -> pd.DataFrame:
    """
    Menerapkan keputusan panitia terhadap satu pasangan duplikat.

    Bersifat generik: bisa dipakai untuk df_login MAUPUN df_register,
    karena pengecekan duplikasi sekarang dilakukan terpisah pada masing-masing
    dataset (bukan lintas dataset).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset asal pasangan (df_login atau df_register).
    decision : str
        Salah satu dari: "keep_a", "keep_b", "keep_both"

    Returns
    -------
    df yang sudah diperbarui.
    """
    id_a = pair_row["id_a"]
    id_b = pair_row["id_b"]

    if decision == "keep_a":
        # Hapus B, simpan A
        df = df[df["id_kobo"] != id_b]

    elif decision == "keep_b":
        # Hapus A, simpan B
        df = df[df["id_kobo"] != id_a]

    elif decision == "keep_both":
        pass  # tidak ada perubahan, keduanya dipertahankan (misal beda kegiatan/valid)

    else:
        raise ValueError(f"Decision tidak dikenal: {decision}")

    return df.reset_index(drop=True)


def append_register_to_login(
    df_login: pd.DataFrame,
    df_register: pd.DataFrame,
    ids_to_append: list,
) -> pd.DataFrame:
    """
    Memindahkan (append) baris-baris tertentu dari df_register ke df_login,
    berdasarkan daftar id_kobo. Kolom yang tidak ada di Login akan diabaikan
    (hanya kolom yang cocok dengan skema Login yang dipertahankan).

    Returns
    -------
    pd.DataFrame
        df_login baru hasil append.
    """
    if not ids_to_append:
        return df_login

    rows_to_append = df_register[df_register["id_kobo"].isin(ids_to_append)].copy()
    if rows_to_append.empty:
        return df_login

    # Selaraskan skema kolom dengan df_login; kolom hilang diisi NA
    login_cols = df_login.columns.tolist()
    for col in login_cols:
        if col not in rows_to_append.columns:
            rows_to_append[col] = pd.NA
    rows_to_append = rows_to_append[login_cols]

    df_login_new = pd.concat([df_login, rows_to_append], ignore_index=True)
    return df_login_new


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Konversi DataFrame ke bytes CSV (siap dipakai st.download_button)."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")  # utf-8-sig agar aman dibuka Excel


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """
    Konversi beberapa DataFrame ke satu file Excel multi-sheet.

    Parameters
    ----------
    sheets : dict
        {"NamaSheet": dataframe, ...}
    """
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            for sheet_name, df in sheets.items():
                safe_name = sheet_name[:31]  # batas nama sheet Excel
                df.to_excel(writer, sheet_name=safe_name, index=False)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Library 'xlsxwriter' belum terpasang. Jalankan: pip install xlsxwriter"
        ) from e
    return buffer.getvalue()
