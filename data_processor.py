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
from matching import check_person_logged_in_for_event


def _parse_event_date(value):
    """
    Coba parse tanggal_kegiatan ke datetime untuk perbandingan 'acara terbaru'.

    Mencoba format umum secara berurutan agar robust terhadap variasi input
    dari Kobo (ISO 'YYYY-MM-DD', atau format lokal 'DD-MM-YYYY'/'DD/MM/YYYY'):
      1. Parse tanpa asumsi dayfirst (menangani ISO 'YYYY-MM-DD' dengan benar).
      2. Jika gagal (NaT), coba ulang dengan dayfirst=True (menangani 'DD-MM-YYYY').
    """
    if pd.isna(value):
        return pd.NaT
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=False)
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return parsed


def resolve_register_duplicate(
    df_login: pd.DataFrame,
    df_register: pd.DataFrame,
    pair_row: pd.Series,
    decision: str,
    threshold: float = config.DUPLICATE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Implementasi lengkap flowchart penanganan duplikat Register:

        Duplikat?
          -> Ya: cek acaranya sama? (informasional saja, keputusan sama)
               -> cek apakah di absensi (Login) sudah ada untuk ACARA TERBARU
                  dari pasangan duplikat ini?
                     -> Ya: tidak perlu menambahkan apa-apa.
                     -> Tidak: append data acara terbaru itu ke Login,
                               lalu hapus salah satu data duplikat di Register.
          -> Tidak (decision == "keep_both"): aman, tidak ada tindakan otomatis
             (dianggap dua orang berbeda).

    Parameters
    ----------
    df_login, df_register : pd.DataFrame
        Dataset saat ini (SEBELUM perubahan apa pun untuk pair ini).
    pair_row : pd.Series
        Satu baris dari hasil find_duplicate_pairs_register (harus punya
        id_a, id_b, judul_kegiatan_a/b, tanggal_kegiatan_a/b, timestamp_submit_a/b).
    decision : str
        "keep_a", "keep_b", atau "keep_both".
    threshold : float
        Ambang batas similarity untuk cek keberadaan di Login.

    Returns
    -------
    (df_login_baru, df_register_baru, info_message)
        info_message : penjelasan tindakan otomatis yang terjadi, untuk
        ditampilkan ke panitia di UI (transparansi keputusan sistem).
    """
    id_a, id_b = pair_row["id_a"], pair_row["id_b"]

    if decision == "keep_both":
        # "aman" — dianggap dua orang berbeda, tidak ada tindakan otomatis.
        return df_login, df_register, (
            "Kedua data dianggap peserta yang BERBEDA — tidak ada perubahan otomatis. "
            "Masing-masing tetap akan dicek kelengkapan Login-nya secara normal via Auto-Append."
        )

    if decision not in ("keep_a", "keep_b"):
        raise ValueError(f"Decision tidak dikenal: {decision}")

    # ---- STEP 1: Hapus salah satu duplikat di Register sesuai keputusan panitia ----
    if decision == "keep_a":
        df_register_new = df_register[df_register["id_kobo"] != id_b].reset_index(drop=True)
    else:  # keep_b
        df_register_new = df_register[df_register["id_kobo"] != id_a].reset_index(drop=True)

    # ---- STEP 2: Tentukan "acara terbaru" di antara A dan B ----
    # Prioritas pembanding: tanggal_kegiatan (tanggal acara sebenarnya).
    # Fallback: timestamp_submit (waktu submit form) jika tanggal_kegiatan kosong/tidak valid.
    date_a = _parse_event_date(pair_row.get("tanggal_kegiatan_a"))
    date_b = _parse_event_date(pair_row.get("tanggal_kegiatan_b"))

    if pd.notna(date_a) and pd.notna(date_b):
        latest_id = id_b if date_b >= date_a else id_a
    elif pd.notna(date_a):
        latest_id = id_a
    elif pd.notna(date_b):
        latest_id = id_b
    else:
        ts_a, ts_b = pair_row.get("timestamp_submit_a"), pair_row.get("timestamp_submit_b")
        latest_id = id_b if (pd.notna(ts_b) and (pd.isna(ts_a) or ts_b >= ts_a)) else id_a

    # Ambil data lengkap baris "acara terbaru" dari df_register ASLI (sebelum dihapus),
    # supaya datanya tetap tersedia untuk di-append meskipun baris itu yang akan dihapus.
    latest_row_df = df_register[df_register["id_kobo"] == latest_id]
    if latest_row_df.empty:
        return df_login, df_register_new, "⚠️ Data acara terbaru tidak ditemukan, tidak ada tindakan otomatis."
    latest_row = latest_row_df.iloc[0].to_dict()
    event_label = latest_row.get("judul_kegiatan", "-")

    # ---- STEP 3: Cek apakah sudah ada di Login untuk acara terbaru tsb ----
    already_logged = check_person_logged_in_for_event(latest_row, df_login, threshold)

    if already_logged:
        info = (
            f"✅ Data untuk kegiatan **'{event_label}'** SUDAH ada di Login — "
            "tidak perlu menambahkan apa-apa (sesuai flowchart)."
        )
        return df_login, df_register_new, info

    # ---- STEP 4: Belum ada di Login -> append otomatis ----
    df_login_new = append_register_to_login(df_login, df_register, [latest_id])
    info = (
        f"➕ Data untuk kegiatan **'{event_label}'** BELUM ada di Login — "
        "otomatis di-append ke dataset Login."
    )
    return df_login_new, df_register_new, info


def apply_review_decision(
    df: pd.DataFrame,
    pair_row: pd.Series,
    decision: str,
) -> pd.DataFrame:
    """
    Menerapkan keputusan panitia terhadap satu pasangan duplikat.

    Bersifat generik: dipakai untuk dataset LOGIN (dedup sederhana, tanpa
    logika auto-append lanjutan — karena Login memang sudah menandakan
    kehadiran, tidak perlu di-append ke mana pun).

    Untuk dataset REGISTER, gunakan `resolve_register_duplicate()` sebagai
    gantinya, karena Register butuh logika tambahan (cek & append ke Login
    untuk acara terbaru) sesuai flowchart bisnis.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset asal pasangan (biasanya df_login).
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
