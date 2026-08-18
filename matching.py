"""
matching.py
-----------
Inti dari mesin deduplikasi (record linkage):
  - Pengecekan duplikasi dilakukan SECARA TERPISAH pada masing-masing dataset
    (Login/Absensi DAN Register) — bukan lintas Login vs Register. Fungsi
    `find_duplicate_pairs()` bersifat generik dan dipanggil dua kali oleh
    app.py: sekali untuk df_login, sekali untuk df_register. Ini mengcover
    kasus peserta yang tidak sengaja submit form Register (atau Login) 2x.
  - GATE WAJIB: pasangan hanya dibandingkan jika `judul_kegiatan` SAMA PERSIS
    (case-insensitive, whitespace-trimmed). Jika judul kegiatan berbeda,
    pasangan otomatis di-keep (tidak dianggap duplikat), karena orang yang
    sama hadir di kegiatan berbeda adalah hal yang valid/wajar.
  - Global pairwise matching (TANPA groupby tanggal_lahir) agar typo
    tanggal lahir tetap bisa terdeteksi kemiripannya — groupby yang dilakukan
    hanya berupa GATE judul_kegiatan di atas, bukan tanggal_lahir.
  - Weighted similarity scoring memakai RapidFuzz.
  - Menandai pasangan dengan skor >= threshold sebagai "Potensi Double Count".

Kompleksitas O(N^2) *per judul_kegiatan* disengaja sesuai requirement —
gate judul_kegiatan otomatis mengecilkan jumlah pasangan yang perlu dihitung
similarity-nya dibanding O(N^2) murni atas seluruh dataset.
"""

from __future__ import annotations

import pandas as pd
from itertools import combinations
from rapidfuzz import fuzz

import config


def _clean_text(value) -> str:
    """Normalisasi ringan sebelum fuzzy matching: string, strip, lower, handle NaN."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def compute_pair_score(record_a: dict, record_b: dict) -> dict:
    """
    [DATASET LOGIN] Menghitung skor kemiripan berbobot antara dua record,
    memakai skema bobot Login: Nama, Tanggal Lahir, Kelurahan, Area Program.

    Menggunakan:
      - fuzz.token_sort_ratio untuk Nama (toleran nama terbalik/singkatan)
      - fuzz.ratio untuk Tanggal Lahir, Kelurahan, Area Program

    Returns
    -------
    dict berisi skor per-field + skor akhir (final_score)
    """
    nama_a, nama_b = _clean_text(record_a.get("nama")), _clean_text(record_b.get("nama"))
    dob_a, dob_b = _clean_text(record_a.get("tanggal_lahir")), _clean_text(record_b.get("tanggal_lahir"))
    kel_a, kel_b = _clean_text(record_a.get("kelurahan")), _clean_text(record_b.get("kelurahan"))
    area_a, area_b = _clean_text(record_a.get("area_program")), _clean_text(record_b.get("area_program"))

    score_nama = fuzz.token_sort_ratio(nama_a, nama_b) if nama_a and nama_b else 0.0
    score_dob = fuzz.ratio(dob_a, dob_b) if dob_a and dob_b else 0.0
    score_kelurahan = fuzz.ratio(kel_a, kel_b) if kel_a and kel_b else 0.0
    score_area = fuzz.ratio(area_a, area_b) if area_a and area_b else 0.0

    final_score = (
        score_nama * config.WEIGHT_NAMA
        + score_dob * config.WEIGHT_DOB
        + score_kelurahan * config.WEIGHT_KELURAHAN
        + score_area * config.WEIGHT_AREA
    )

    return {
        "score_nama": round(score_nama, 2),
        "score_dob": round(score_dob, 2),
        "score_kelurahan": round(score_kelurahan, 2),
        "score_area": round(score_area, 2),
        "final_score": round(final_score, 2),
    }


def compute_pair_score_register(record_a: dict, record_b: dict) -> dict:
    """
    [DATASET REGISTER] Menghitung skor kemiripan berbobot antara dua record,
    FOKUS pada Nama, Tanggal Lahir, dan Nama Kepala Keluarga — karena data
    Register harus benar-benar unik secara global (bukan per judul_kegiatan).

    Menggunakan:
      - fuzz.token_sort_ratio untuk Nama & Nama Kepala Keluarga
        (toleran nama terbalik/singkatan)
      - fuzz.ratio untuk Tanggal Lahir (toleran typo angka/format)

    Returns
    -------
    dict berisi skor per-field + skor akhir (final_score)
    """
    nama_a, nama_b = _clean_text(record_a.get("nama")), _clean_text(record_b.get("nama"))
    dob_a, dob_b = _clean_text(record_a.get("tanggal_lahir")), _clean_text(record_b.get("tanggal_lahir"))
    kk_a = _clean_text(record_a.get("nama_kepala_keluarga"))
    kk_b = _clean_text(record_b.get("nama_kepala_keluarga"))

    score_nama = fuzz.token_sort_ratio(nama_a, nama_b) if nama_a and nama_b else 0.0
    score_dob = fuzz.ratio(dob_a, dob_b) if dob_a and dob_b else 0.0
    score_kepala_keluarga = fuzz.token_sort_ratio(kk_a, kk_b) if kk_a and kk_b else 0.0

    final_score = (
        score_nama * config.WEIGHT_NAMA_REG
        + score_dob * config.WEIGHT_DOB_REG
        + score_kepala_keluarga * config.WEIGHT_KEPALA_KELUARGA_REG
    )

    return {
        "score_nama": round(score_nama, 2),
        "score_dob": round(score_dob, 2),
        "score_kepala_keluarga": round(score_kepala_keluarga, 2),
        "final_score": round(final_score, 2),
    }


def find_duplicate_pairs(
    df: pd.DataFrame,
    threshold: float = config.DUPLICATE_THRESHOLD,
    progress_callback=None,
    dataset_label: str = "",
) -> pd.DataFrame:
    """
    Mendeteksi potensi double count DI DALAM SATU dataset (bisa dipanggil untuk
    df_login MAUPUN df_register secara terpisah — keduanya sama-sama bisa
    mengalami submit ganda oleh peserta yang sama).

    Aturan pembanding:
      1. GATE WAJIB: pasangan hanya dibandingkan jika `judul_kegiatan` SAMA PERSIS
         (case-insensitive, trimmed). Jika judul kegiatan berbeda -> otomatis
         di-keep, TIDAK dihitung similarity-nya sama sekali (orang yang sama
         hadir di kegiatan berbeda adalah wajar, bukan double count).
      2. Untuk pasangan yang lolos gate, dilakukan GLOBAL PAIRWISE MATCHING
         (O(N^2) per grup judul_kegiatan) TANPA groupby tanggal_lahir, supaya
         typo tanggal lahir tetap terdeteksi kemiripannya.
      3. Skor akhir dihitung dengan weighted similarity (nama/DOB/kelurahan/area).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset yang ingin dicek duplikasinya — df_login ATAU df_register
        (harus punya kolom REQUIRED_MATCH_COLUMNS, DUPLICATE_GATE_COLUMN, dan 'id_kobo').
    threshold : float
        Ambang batas skor akhir untuk dianggap "Potensi Double Count".
    progress_callback : callable, optional
        Fungsi callback(current, total) untuk update progress bar UI.
    dataset_label : str, optional
        Label dataset ("Login" / "Register") untuk disertakan di kolom hasil,
        supaya mudah dibedakan saat kedua hasil digabung/ditampilkan di UI.

    Returns
    -------
    pd.DataFrame
        Daftar pasangan duplikat dengan kolom:
        id_a, id_b, dataset, judul_kegiatan, <field>_a, <field>_b, score_*, final_score, status
    """
    required = config.REQUIRED_MATCH_COLUMNS + [config.DUPLICATE_GATE_COLUMN]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib untuk matching tidak ditemukan: {missing_cols}")

    if df.empty:
        return pd.DataFrame()

    # ---- STEP 1: GATE berdasarkan judul_kegiatan ----
    # Kelompokkan index berdasarkan judul_kegiatan yang sudah dinormalisasi.
    # Ini BUKAN groupby tanggal_lahir (tetap dilarang), melainkan gate kegiatan
    # yang diminta eksplisit oleh requirement bisnis.
    df_work = df.reset_index(drop=True).copy()
    df_work["_gate_key"] = df_work[config.DUPLICATE_GATE_COLUMN].apply(_clean_text)

    results = []
    total_pairs = 0
    group_indices = {}
    for gate_key, group_df in df_work.groupby("_gate_key"):
        if gate_key == "":
            continue  # judul_kegiatan kosong -> tidak bisa dipastikan sama, skip dari pengecekan
        idx_list = group_df.index.tolist()
        if len(idx_list) > 1:
            group_indices[gate_key] = idx_list
            total_pairs += len(idx_list) * (len(idx_list) - 1) // 2

    if total_pairs == 0:
        return pd.DataFrame()

    records = df_work.to_dict("records")
    processed = 0

    # ---- STEP 2: Pairwise matching HANYA di dalam masing-masing grup judul_kegiatan ----
    for gate_key, idx_list in group_indices.items():
        for i, j in combinations(idx_list, 2):
            rec_a, rec_b = records[i], records[j]

            # Data A = input pertama (berdasarkan timestamp_submit jika tersedia),
            # Data B = input susulan.
            ts_a = rec_a.get("timestamp_submit")
            ts_b = rec_b.get("timestamp_submit")
            if pd.notna(ts_a) and pd.notna(ts_b) and ts_a > ts_b:
                rec_a, rec_b = rec_b, rec_a  # swap supaya A selalu lebih awal

            scores = compute_pair_score(rec_a, rec_b)

            processed += 1
            if progress_callback and (processed % 200 == 0 or processed == total_pairs):
                progress_callback(processed, total_pairs)

            if scores["final_score"] >= threshold:
                results.append({
                    "pair_id": f"{dataset_label}__{rec_a.get('id_kobo')}__{rec_b.get('id_kobo')}"
                               if dataset_label else f"{rec_a.get('id_kobo')}__{rec_b.get('id_kobo')}",
                    "dataset": dataset_label,
                    "id_a": rec_a.get("id_kobo"),
                    "id_b": rec_b.get("id_kobo"),
                    "judul_kegiatan": rec_a.get("judul_kegiatan"),
                    "nama_a": rec_a.get("nama"),
                    "nama_b": rec_b.get("nama"),
                    "tanggal_lahir_a": rec_a.get("tanggal_lahir"),
                    "tanggal_lahir_b": rec_b.get("tanggal_lahir"),
                    "kelurahan_a": rec_a.get("kelurahan"),
                    "kelurahan_b": rec_b.get("kelurahan"),
                    "area_program_a": rec_a.get("area_program"),
                    "area_program_b": rec_b.get("area_program"),
                    "timestamp_submit_a": rec_a.get("timestamp_submit"),
                    "timestamp_submit_b": rec_b.get("timestamp_submit"),
                    **scores,
                    "status": "Potensi Double Count",
                })

    if not results:
        return pd.DataFrame()

    df_result = pd.DataFrame(results).sort_values("final_score", ascending=False).reset_index(drop=True)
    return df_result


def find_duplicate_pairs_register(
    df_register: pd.DataFrame,
    threshold: float = config.DUPLICATE_THRESHOLD,
    progress_callback=None,
) -> pd.DataFrame:
    """
    [DATASET REGISTER] Mendeteksi potensi double count di dataset Register.

    BERBEDA dari Login: pengecekan dilakukan GLOBAL PAIRWISE (O(N^2)) atas
    SELURUH baris Register — TIDAK di-gate oleh judul_kegiatan — karena
    identitas peserta di form Register harus unik secara keseluruhan
    (satu orang seharusnya hanya register sekali, terlepas dari kegiatan
    mana pun yang nanti diikutinya).

    Fokus pembanding: Nama, Tanggal Lahir, Nama Kepala Keluarga
    (lihat compute_pair_score_register & bobot WEIGHT_*_REG di config.py).

    Parameters
    ----------
    df_register : pd.DataFrame
        Dataset Register (harus punya kolom REQUIRED_MATCH_COLUMNS_REGISTER
        dan 'id_kobo').
    threshold : float
        Ambang batas skor akhir untuk dianggap "Potensi Double Count".
    progress_callback : callable, optional
        Fungsi callback(current, total) untuk update progress bar UI.

    Returns
    -------
    pd.DataFrame
        Daftar pasangan duplikat dengan kolom:
        id_a, id_b, <field>_a, <field>_b, score_*, final_score, status
    """
    missing_cols = [c for c in config.REQUIRED_MATCH_COLUMNS_REGISTER if c not in df_register.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib untuk matching Register tidak ditemukan: {missing_cols}")

    if df_register.empty:
        return pd.DataFrame()

    df_work = df_register.reset_index(drop=True).copy()
    records = df_work.to_dict("records")
    n = len(records)
    total_pairs = n * (n - 1) // 2 if n > 1 else 0
    results = []

    if total_pairs == 0:
        return pd.DataFrame()

    processed = 0
    for i, j in combinations(range(n), 2):
        rec_a, rec_b = records[i], records[j]

        # Data A = input pertama (berdasarkan timestamp_submit jika tersedia),
        # Data B = input susulan.
        ts_a = rec_a.get("timestamp_submit")
        ts_b = rec_b.get("timestamp_submit")
        if pd.notna(ts_a) and pd.notna(ts_b) and ts_a > ts_b:
            rec_a, rec_b = rec_b, rec_a

        scores = compute_pair_score_register(rec_a, rec_b)

        processed += 1
        if progress_callback and (processed % 200 == 0 or processed == total_pairs):
            progress_callback(processed, total_pairs)

        if scores["final_score"] >= threshold:
            results.append({
                "pair_id": f"Register__{rec_a.get('id_kobo')}__{rec_b.get('id_kobo')}",
                "dataset": "Register",
                "id_a": rec_a.get("id_kobo"),
                "id_b": rec_b.get("id_kobo"),
                "nama_a": rec_a.get("nama"),
                "nama_b": rec_b.get("nama"),
                "tanggal_lahir_a": rec_a.get("tanggal_lahir"),
                "tanggal_lahir_b": rec_b.get("tanggal_lahir"),
                "nama_kepala_keluarga_a": rec_a.get("nama_kepala_keluarga"),
                "nama_kepala_keluarga_b": rec_b.get("nama_kepala_keluarga"),
                "judul_kegiatan_a": rec_a.get("judul_kegiatan"),
                "judul_kegiatan_b": rec_b.get("judul_kegiatan"),
                "tanggal_kegiatan_a": rec_a.get("tanggal_kegiatan"),
                "tanggal_kegiatan_b": rec_b.get("tanggal_kegiatan"),
                "timestamp_submit_a": rec_a.get("timestamp_submit"),
                "timestamp_submit_b": rec_b.get("timestamp_submit"),
                **scores,
                "status": "Potensi Double Count",
            })

    if not results:
        return pd.DataFrame()

    df_result = pd.DataFrame(results).sort_values("final_score", ascending=False).reset_index(drop=True)
    return df_result


def check_person_logged_in_for_event(
    person_record: dict,
    df_login: pd.DataFrame,
    threshold: float = config.DUPLICATE_THRESHOLD,
) -> bool:
    """
    Mengecek apakah `person_record` (biasanya berasal dari baris Register)
    SUDAH memiliki entri di dataset Login UNTUK KEGIATAN YANG SAMA
    (judul_kegiatan sama persis, dinormalisasi).

    Ini adalah versi "event-aware" dari pengecekan sudah-login-atau-belum —
    dipakai untuk mengimplementasikan langkah flowchart:
    "cek apakah di absensi datanya ada dengan acara terbaru?"

    Seseorang yang sudah login di Event 1 TIDAK otomatis dianggap sudah login
    di Event 3 — harus dicek per acara.

    Parameters
    ----------
    person_record : dict
        Baris data (biasanya dari Register) yang ingin dicek, harus punya
        'judul_kegiatan' dan field REQUIRED_MATCH_COLUMNS.
    df_login : pd.DataFrame
        Dataset Login untuk dibandingkan.
    threshold : float
        Ambang batas skor kemiripan.

    Returns
    -------
    bool
        True jika ditemukan baris Login dengan judul_kegiatan sama DAN
        skor kemiripan orang >= threshold. False jika tidak ditemukan.
    """
    if df_login.empty:
        return False

    target_event = _clean_text(person_record.get("judul_kegiatan"))
    if target_event == "":
        # Tidak ada info kegiatan -> tidak bisa dipastikan, anggap belum login
        # (lebih aman: lebih baik ditawarkan untuk di-append daripada data hilang)
        return False

    for login_row in df_login.to_dict("records"):
        if _clean_text(login_row.get("judul_kegiatan")) != target_event:
            continue  # beda kegiatan, skip (bukan pembanding yang valid)
        scores = compute_pair_score(person_record, login_row)
        if scores["final_score"] >= threshold:
            return True

    return False


def find_registered_not_logged_in(
    df_login: pd.DataFrame,
    df_register: pd.DataFrame,
    threshold: float = config.DUPLICATE_THRESHOLD,
) -> pd.DataFrame:
    """
    Mencari peserta yang sudah mengisi Register tetapi belum Login
    UNTUK KEGIATAN YANG SAMA (event-aware — lihat check_person_logged_in_for_event).

    PENTING: seseorang yang sudah login di kegiatan lain TIDAK dianggap sudah
    login di kegiatan saat ini. Setiap baris Register dicek terhadap Login
    dengan judul_kegiatan yang sama persis, bukan lintas semua kegiatan.

    Returns
    -------
    pd.DataFrame
        Subset df_register yang belum memiliki pasangan Login untuk
        kegiatan yang sama.
    """
    if df_register.empty:
        return df_register.copy()
    if df_login.empty:
        return df_register.copy()

    not_logged_in_rows = []
    for reg_dict in df_register.to_dict("records"):
        if not check_person_logged_in_for_event(reg_dict, df_login, threshold):
            not_logged_in_rows.append(reg_dict)

    if not not_logged_in_rows:
        return pd.DataFrame(columns=df_register.columns)

    return pd.DataFrame(not_logged_in_rows).reset_index(drop=True)
