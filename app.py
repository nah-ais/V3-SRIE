"""
app.py
------
Aplikasi Streamlit: Sistem Pembersihan Data Kehadiran Acara.

Fitur utama:
  1. Dashboard metrik (total login, total register, jumlah potensi duplikat).
  2. Interactive Reviewer untuk pasangan data dengan skor kemiripan >= 95%.
  3. Auto-Append Register -> Login untuk peserta yang belum login.
  4. Export dataset hasil pembersihan (CSV / Excel).

Jalankan dengan: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

import config
from kobo_api import fetch_kobo_data, load_csv_fallback, KoboAPIError
from matching import find_duplicate_pairs, find_duplicate_pairs_register, find_registered_not_logged_in
from data_processor import (
    apply_review_decision,
    append_register_to_login,
    to_csv_bytes,
    to_excel_bytes,
)

st.set_page_config(
    page_title="Sistem Pembersihan Data Kehadiran Acara",
    page_icon="🧹",
    layout="wide",
)


# =========================================================
# INIT SESSION STATE
# =========================================================
def init_session_state():
    defaults = {
        config.SS_LOGIN_DF: pd.DataFrame(),
        config.SS_REGISTER_DF: pd.DataFrame(),
        config.SS_DUPLICATE_PAIRS_LOGIN: pd.DataFrame(),
        config.SS_DUPLICATE_PAIRS_REGISTER: pd.DataFrame(),
        config.SS_NOT_LOGIN_YET: pd.DataFrame(),
        config.SS_REVIEW_DECISIONS_LOGIN: {},
        config.SS_REVIEW_DECISIONS_REGISTER: {},
        config.SS_APPENDED_IDS: set(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# =========================================================
# SIDEBAR: SUMBER DATA
# =========================================================
def sidebar_data_source():
    st.sidebar.header("⚙️ Sumber Data")
    mode = st.sidebar.radio(
        "Pilih metode input data",
        ["KoboToolbox API", "Upload CSV (fallback/testing)"],
        help="Gunakan API untuk data live, atau CSV untuk pengujian/offline.",
    )

    if mode == "KoboToolbox API":
        with st.sidebar.form("kobo_api_form"):
            st.caption(
                "Kredensial & Form UID sudah terisi otomatis dari config.py. "
                "Ubah di sini jika perlu override sementara (tidak mengubah file config.py)."
            )
            api_token = st.text_input("API Token", value=config.KOBO_TOKEN, type="password")
            base_url = st.text_input("Base URL", value=config.KOBO_ENDPOINT)
            asset_uid_login = st.text_input(
                "Asset UID (Form UID) - Form Login",
                value=config.FORM_UID_LOGIN,
                help="Dari config.py: FORM_UID_LOGIN",
            )
            asset_uid_register = st.text_input(
                "Asset UID (Form UID) - Form Register",
                value=config.FORM_UID_REGISTRASI,
                help="Dari config.py: FORM_UID_REGISTRASI",
            )
            submitted = st.form_submit_button("🔄 Tarik Data dari Kobo")

        if submitted:
            if not api_token or api_token == "MY_TOKEN":
                st.sidebar.error("API Token belum diisi dengan benar. Set KOBO_TOKEN di config.py atau isi manual di sini.")
            elif not asset_uid_login or not asset_uid_register:
                st.sidebar.error("Mohon lengkapi kedua Form UID (Login & Register). Isi FORM_UID_LOGIN di config.py.")
            else:
                # noqa: proses fetch di bawah
                try:
                    with st.spinner("Mengambil data dari KoboToolbox..."):
                        df_login = fetch_kobo_data(
                            asset_uid_login, api_token, config.LOGIN_COLUMN_MAP, base_url
                        )
                        df_register = fetch_kobo_data(
                            asset_uid_register, api_token, config.REGISTER_COLUMN_MAP, base_url
                        )
                    st.session_state[config.SS_LOGIN_DF] = df_login
                    st.session_state[config.SS_REGISTER_DF] = df_register
                    # Reset hasil matching lama karena data berubah
                    st.session_state[config.SS_DUPLICATE_PAIRS_LOGIN] = pd.DataFrame()
                    st.session_state[config.SS_DUPLICATE_PAIRS_REGISTER] = pd.DataFrame()
                    st.session_state[config.SS_NOT_LOGIN_YET] = pd.DataFrame()
                    st.sidebar.success(
                        f"Berhasil! Login: {len(df_login)} baris, Register: {len(df_register)} baris."
                    )
                except KoboAPIError as e:
                    st.sidebar.error(f"Gagal mengambil data: {e}")
                except Exception as e:
                    st.sidebar.error(f"Terjadi kesalahan tak terduga: {e}")

    else:  # Upload CSV fallback
        st.sidebar.caption("Upload hasil export CSV dari KoboToolbox untuk masing-masing form.")
        file_login = st.sidebar.file_uploader("CSV Form Login", type=["csv"], key="upload_login")
        file_register = st.sidebar.file_uploader("CSV Form Register", type=["csv"], key="upload_register")

        if st.sidebar.button("📥 Muat Data CSV"):
            try:
                if file_login is not None:
                    st.session_state[config.SS_LOGIN_DF] = load_csv_fallback(
                        file_login, config.LOGIN_COLUMN_MAP
                    )
                if file_register is not None:
                    st.session_state[config.SS_REGISTER_DF] = load_csv_fallback(
                        file_register, config.REGISTER_COLUMN_MAP
                    )
                st.session_state[config.SS_DUPLICATE_PAIRS_LOGIN] = pd.DataFrame()
                st.session_state[config.SS_DUPLICATE_PAIRS_REGISTER] = pd.DataFrame()
                st.session_state[config.SS_NOT_LOGIN_YET] = pd.DataFrame()
                st.sidebar.success("Data CSV berhasil dimuat.")
            except KoboAPIError as e:
                st.sidebar.error(f"Gagal memuat CSV: {e}")
            except Exception as e:
                st.sidebar.error(f"Terjadi kesalahan tak terduga: {e}")

    st.sidebar.divider()
    st.sidebar.header("🎯 Threshold Deduplikasi")
    threshold = st.sidebar.slider(
        "Ambang Batas Skor Akhir (%)",
        min_value=50.0,
        max_value=100.0,
        value=config.DUPLICATE_THRESHOLD,
        step=0.5,
        help="Pasangan data dengan skor >= nilai ini akan ditandai sebagai Potensi Double Count.",
    )
    return threshold


# =========================================================
# SECTION 1: DASHBOARD METRIC
# =========================================================
def section_dashboard():
    st.subheader("📊 Dashboard Metrik")
    df_login = st.session_state[config.SS_LOGIN_DF]
    df_register = st.session_state[config.SS_REGISTER_DF]
    df_dupes_login = st.session_state[config.SS_DUPLICATE_PAIRS_LOGIN]
    df_dupes_register = st.session_state[config.SS_DUPLICATE_PAIRS_REGISTER]
    belum_login = st.session_state[config.SS_NOT_LOGIN_YET]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Login", len(df_login))
    col2.metric("Total Register", len(df_register))
    col3.metric("Duplikat di Login", len(df_dupes_login) if not df_dupes_login.empty else 0)
    col4.metric("Duplikat di Register", len(df_dupes_register) if not df_dupes_register.empty else 0)
    col5.metric("Register Belum Login", len(belum_login) if not belum_login.empty else 0)


# =========================================================
# SECTION 2: JALANKAN PROSES MATCHING
# =========================================================
def _estimate_pairs(df: pd.DataFrame) -> int:
    """Estimasi jumlah pasangan yang akan dibandingkan (setelah gate judul_kegiatan)."""
    if df.empty or config.DUPLICATE_GATE_COLUMN not in df.columns:
        return 0
    group_sizes = df.groupby(df[config.DUPLICATE_GATE_COLUMN].astype(str).str.strip().str.lower()).size()
    return int(sum(n * (n - 1) // 2 for n in group_sizes if n > 1))


def section_run_matching(threshold: float):
    st.subheader("🔍 Deteksi Duplikat")
    st.caption(
        "**Login**: dicek per `judul_kegiatan` (satu orang wajar hadir di banyak kegiatan berbeda). "
        "**Register**: dicek secara **global lintas seluruh baris** (tanpa gate kegiatan), fokus pada "
        "**Nama, Tanggal Lahir, dan Nama Kepala Keluarga** — karena identitas peserta di Register "
        "harus benar-benar unik, terlepas dari kegiatan apa pun."
    )

    df_login = st.session_state[config.SS_LOGIN_DF]
    df_register = st.session_state[config.SS_REGISTER_DF]

    if df_login.empty and df_register.empty:
        st.info("Silakan muat data Login dan/atau Register terlebih dahulu dari sidebar.")
        return

    est_login = _estimate_pairs(df_login)
    n_reg = len(df_register)
    est_register = n_reg * (n_reg - 1) // 2 if n_reg > 1 else 0

    st.caption(
        f"Login: **{len(df_login)}** baris → estimasi **{est_login:,}** pasangan (per judul_kegiatan). "
        f"Register: **{len(df_register)}** baris → estimasi **{est_register:,}** pasangan (global, O(N²))."
    )
    if (est_login + est_register) > 200_000:
        st.warning("⚠️ Jumlah pasangan cukup besar, proses mungkin memakan waktu.")

    if st.button("🚀 Jalankan Deteksi Duplikat (Login & Register)"):
        try:
            # ---- Deteksi duplikat di dataset LOGIN (per judul_kegiatan) ----
            if not df_login.empty:
                progress_login = st.progress(0.0, text="Memproses duplikasi di dataset Login...")

                def _cb_login(current, total):
                    progress_login.progress(min(current / total, 1.0), text=f"Login: {current:,}/{total:,} pasangan...")

                df_dupes_login = find_duplicate_pairs(
                    df_login, threshold=threshold, progress_callback=_cb_login, dataset_label="Login"
                )
                progress_login.empty()
                st.session_state[config.SS_DUPLICATE_PAIRS_LOGIN] = df_dupes_login
            else:
                df_dupes_login = pd.DataFrame()
                st.session_state[config.SS_DUPLICATE_PAIRS_LOGIN] = df_dupes_login

            # ---- Deteksi duplikat di dataset REGISTER (global, fokus nama/DOB/kepala keluarga) ----
            if not df_register.empty:
                progress_register = st.progress(0.0, text="Memproses duplikasi di dataset Register...")

                def _cb_register(current, total):
                    progress_register.progress(min(current / total, 1.0), text=f"Register: {current:,}/{total:,} pasangan...")

                df_dupes_register = find_duplicate_pairs_register(
                    df_register, threshold=threshold, progress_callback=_cb_register
                )
                progress_register.empty()
                st.session_state[config.SS_DUPLICATE_PAIRS_REGISTER] = df_dupes_register
            else:
                df_dupes_register = pd.DataFrame()
                st.session_state[config.SS_DUPLICATE_PAIRS_REGISTER] = df_dupes_register

            # ---- Deteksi Register yang belum Login (fuzzy, lintas dataset) ----
            df_not_login = find_registered_not_logged_in(df_login, df_register, threshold=threshold)
            st.session_state[config.SS_NOT_LOGIN_YET] = df_not_login

            total_dupes = len(df_dupes_login) + len(df_dupes_register)
            if total_dupes == 0:
                st.success("Tidak ditemukan pasangan duplikat di kedua dataset. Data relatif bersih! 🎉")
            else:
                st.success(
                    f"Ditemukan {len(df_dupes_login)} pasangan duplikat di Login "
                    f"dan {len(df_dupes_register)} pasangan duplikat di Register."
                )
        except ValueError as e:
            st.error(f"Data tidak lengkap untuk matching: {e}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses: {e}")


# =========================================================
# SECTION 3: INTERACTIVE REVIEWER
# =========================================================
def section_reviewer(dataset_key: str, dupes_state_key: str, decisions_state_key: str, dataset_df_state_key: str):
    """
    Reviewer generik untuk satu dataset (Login atau Register).

    Parameters
    ----------
    dataset_key : str
        Label untuk ditampilkan ("Login" / "Register").
    dupes_state_key, decisions_state_key, dataset_df_state_key : str
        Kunci session_state terkait dataset yang direview.
    """
    st.subheader(f"🕵️ Interactive Reviewer — Potensi Double Count ({dataset_key})")

    df_dupes = st.session_state[dupes_state_key]
    decisions = st.session_state[decisions_state_key]

    if df_dupes.empty:
        st.info(f"Belum ada pasangan duplikat di dataset {dataset_key} untuk direview. Jalankan deteksi terlebih dahulu.")
        return

    # Hanya tampilkan pasangan yang belum diputuskan
    pending = df_dupes[~df_dupes["pair_id"].isin(decisions.keys())]
    st.caption(f"Menampilkan {len(pending)} dari {len(df_dupes)} pasangan (yang belum direview).")

    for _, row in pending.iterrows():
        with st.container(border=True):
            is_register = dataset_key == "Register"

            if is_register:
                st.markdown(
                    f"**Skor Akhir: `{row['final_score']}%`** "
                    f"— Nama: `{row['score_nama']}%` | DOB: `{row['score_dob']}%` "
                    f"| Nama Kepala Keluarga: `{row['score_kepala_keluarga']}%`"
                )
            else:
                st.markdown(
                    f"**Skor Akhir: `{row['final_score']}%`** "
                    f"— Nama: `{row['score_nama']}%` | DOB: `{row['score_dob']}%` "
                    f"| Kelurahan: `{row['score_kelurahan']}%` | Area: `{row['score_area']}%` "
                    f"&nbsp;|&nbsp; 🎪 Kegiatan: **{row['judul_kegiatan']}**"
                )

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**🅰️ Data A (Input Pertama)**")
                st.write(f"ID: `{row['id_a']}`")
                st.write(f"Nama: {row['nama_a']}")
                st.write(f"Tgl Lahir: {row['tanggal_lahir_a']}")
                if is_register:
                    st.write(f"Nama Kepala Keluarga: {row['nama_kepala_keluarga_a']}")
                else:
                    st.write(f"Kelurahan: {row['kelurahan_a']}")
                    st.write(f"Area Program: {row['area_program_a']}")
                st.write(f"Waktu Submit: {row['timestamp_submit_a']}")

            with col_b:
                st.markdown("**🅱️ Data B (Input Susulan)**")
                st.write(f"ID: `{row['id_b']}`")
                st.write(f"Nama: {row['nama_b']}")
                st.write(f"Tgl Lahir: {row['tanggal_lahir_b']}")
                if is_register:
                    st.write(f"Nama Kepala Keluarga: {row['nama_kepala_keluarga_b']}")
                else:
                    st.write(f"Kelurahan: {row['kelurahan_b']}")
                    st.write(f"Area Program: {row['area_program_b']}")
                st.write(f"Waktu Submit: {row['timestamp_submit_b']}")

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            pair_id = row["pair_id"]

            if btn_col1.button("✅ Simpan A & Hapus B", key=f"keep_a_{dataset_key}_{pair_id}"):
                _apply_decision(row, "keep_a", decisions_state_key, dataset_df_state_key)
                st.rerun()

            if btn_col2.button("✅ Simpan B & Hapus A", key=f"keep_b_{dataset_key}_{pair_id}"):
                _apply_decision(row, "keep_b", decisions_state_key, dataset_df_state_key)
                st.rerun()

            if btn_col3.button("↔️ Keep Keduanya", key=f"keep_both_{dataset_key}_{pair_id}"):
                _apply_decision(row, "keep_both", decisions_state_key, dataset_df_state_key)
                st.rerun()


def _apply_decision(row: pd.Series, decision: str, decisions_state_key: str, dataset_df_state_key: str):
    """Handler internal: terapkan keputusan reviewer ke dataset terkait & simpan histori."""
    try:
        df_target = st.session_state[dataset_df_state_key]

        new_df = apply_review_decision(df_target, row, decision)

        st.session_state[dataset_df_state_key] = new_df
        st.session_state[decisions_state_key][row["pair_id"]] = decision

        label_map = {"keep_a": "Simpan A & Hapus B", "keep_b": "Simpan B & Hapus A", "keep_both": "Keep Keduanya"}
        st.toast(f"Keputusan '{label_map[decision]}' diterapkan untuk pasangan {row['pair_id']}.", icon="✅")
    except Exception as e:
        st.error(f"Gagal menerapkan keputusan: {e}")


# =========================================================
# SECTION 4: AUTO-APPEND REGISTER -> LOGIN
# =========================================================
def section_auto_append():
    st.subheader("➕ Auto-Append: Register Belum Login")

    df_not_login = st.session_state[config.SS_NOT_LOGIN_YET]

    if df_not_login.empty:
        st.info("Tidak ada data (atau belum dijalankan deteksi). Jalankan deteksi duplikat terlebih dahulu.")
        return

    st.caption(
        f"Ditemukan **{len(df_not_login)}** peserta yang sudah Register namun belum terdeteksi Login "
        "(dicek dengan fuzzy match, bukan exact match, agar toleran typo)."
    )

    display_cols = [c for c in config.DISPLAY_COLUMNS if c in df_not_login.columns]
    st.dataframe(df_not_login[display_cols], use_container_width=True, hide_index=True)

    selected_ids = st.multiselect(
        "Pilih peserta yang ingin di-append ke dataset Login (kosongkan untuk pilih semua otomatis saat klik tombol di bawah)",
        options=df_not_login["id_kobo"].tolist(),
        format_func=lambda x: f"{x} - {df_not_login.loc[df_not_login['id_kobo']==x, 'nama'].values[0]}"
        if x in df_not_login["id_kobo"].values else str(x),
    )

    col1, col2 = st.columns(2)
    if col1.button("➕ Append yang Dipilih ke Login"):
        try:
            ids_to_use = selected_ids if selected_ids else df_not_login["id_kobo"].tolist()
            new_login = append_register_to_login(
                st.session_state[config.SS_LOGIN_DF], st.session_state[config.SS_REGISTER_DF], ids_to_use
            )
            st.session_state[config.SS_LOGIN_DF] = new_login
            st.session_state[config.SS_APPENDED_IDS].update(ids_to_use)
            # Hilangkan dari daftar "belum login" karena sudah di-append
            st.session_state[config.SS_NOT_LOGIN_YET] = df_not_login[~df_not_login["id_kobo"].isin(ids_to_use)]
            st.success(f"{len(ids_to_use)} peserta berhasil di-append ke dataset Login.")
            st.rerun()
        except Exception as e:
            st.error(f"Gagal melakukan append: {e}")

    if col2.button("➕ Append Semua ke Login"):
        try:
            ids_to_use = df_not_login["id_kobo"].tolist()
            new_login = append_register_to_login(
                st.session_state[config.SS_LOGIN_DF], st.session_state[config.SS_REGISTER_DF], ids_to_use
            )
            st.session_state[config.SS_LOGIN_DF] = new_login
            st.session_state[config.SS_APPENDED_IDS].update(ids_to_use)
            st.session_state[config.SS_NOT_LOGIN_YET] = pd.DataFrame(columns=df_not_login.columns)
            st.success(f"Seluruh {len(ids_to_use)} peserta berhasil di-append ke dataset Login.")
            st.rerun()
        except Exception as e:
            st.error(f"Gagal melakukan append: {e}")


# =========================================================
# SECTION 5: EXPORT / DATABASE CONSTRAINT
# =========================================================
def section_export():
    st.subheader("💾 Export Dataset Hasil Pembersihan")

    df_login = st.session_state[config.SS_LOGIN_DF]
    df_register = st.session_state[config.SS_REGISTER_DF]
    decisions_made = len(st.session_state[config.SS_REVIEW_DECISIONS_LOGIN]) + len(
        st.session_state[config.SS_REVIEW_DECISIONS_REGISTER]
    )
    appended = len(st.session_state[config.SS_APPENDED_IDS])

    if decisions_made == 0 and appended == 0:
        st.warning(
            "⚠️ Anda belum melakukan review duplikat atau append data. "
            "Disarankan untuk menyelesaikan proses cleaning terlebih dahulu sebelum export, "
            "agar dataset final konsisten (database constraint)."
        )

    st.caption(
        f"Ringkasan sesi: **{decisions_made}** keputusan review diterapkan, "
        f"**{appended}** peserta register telah di-append ke login."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Export Login (CSV)",
            data=to_csv_bytes(df_login),
            file_name="login_clean.csv",
            mime="text/csv",
            disabled=df_login.empty,
        )
    with col2:
        st.download_button(
            "⬇️ Export Register (CSV)",
            data=to_csv_bytes(df_register),
            file_name="register_clean.csv",
            mime="text/csv",
            disabled=df_register.empty,
        )

    try:
        excel_bytes = to_excel_bytes({"Login": df_login, "Register": df_register})
        st.download_button(
            "⬇️ Export Gabungan (Excel, multi-sheet)",
            data=excel_bytes,
            file_name="dataset_kehadiran_clean.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=(df_login.empty and df_register.empty),
        )
    except RuntimeError as e:
        st.caption(f"ℹ️ Export Excel tidak tersedia: {e}")


# =========================================================
# MAIN LAYOUT
# =========================================================
def main():
    st.title("🧹 Sistem Pembersihan Data Kehadiran Acara")
    st.caption(
        "Deduplikasi berbasis fuzzy matching (RapidFuzz) untuk mengatasi double counting "
        "dan data tidak valid dari KoboToolbox."
    )

    threshold = sidebar_data_source()

    section_dashboard()
    st.divider()
    section_run_matching(threshold)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🕵️ Reviewer Login", "🕵️ Reviewer Register", "➕ Auto-Append Login", "💾 Export"]
    )
    with tab1:
        section_reviewer(
            "Login",
            config.SS_DUPLICATE_PAIRS_LOGIN,
            config.SS_REVIEW_DECISIONS_LOGIN,
            config.SS_LOGIN_DF,
        )
    with tab2:
        section_reviewer(
            "Register",
            config.SS_DUPLICATE_PAIRS_REGISTER,
            config.SS_REVIEW_DECISIONS_REGISTER,
            config.SS_REGISTER_DF,
        )
    with tab3:
        section_auto_append()
    with tab4:
        section_export()


if __name__ == "__main__":
    main()
