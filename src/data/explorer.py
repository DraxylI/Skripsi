from typing import Dict

import pandas as pd

KUNCI_PRESENTASI = ["code_module", "code_presentation"]
KUNCI_SISWA = KUNCI_PRESENTASI + ["id_student"]
JENIS_PENILAIAN = {"TMA": "penilaian_tma", "CMA": "penilaian_cma", "Exam": "penilaian_exam"}
HASIL_AKHIR = ["pass", "distinction", "fail", "withdrawn"]


def _persen_kosong(jumlah_kosong: pd.Series, jumlah_baris: int) -> pd.Series:
    if jumlah_baris == 0:
        return pd.Series(0.0, index=jumlah_kosong.index)
    return (jumlah_kosong / jumlah_baris * 100).round(2)


def _ringkasan_kolom(tabel: str, tipe_data: pd.Series, jumlah_baris: int, jumlah_kosong: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "tabel": tabel,
        "kolom": tipe_data.index,
        "tipe_data": tipe_data.astype(str).to_numpy(),
        "jumlah_baris": jumlah_baris,
        "jumlah_kosong": jumlah_kosong.to_numpy(),
        "persen_kosong": _persen_kosong(jumlah_kosong, jumlah_baris).to_numpy(),
    })


def _ringkasan_student_vle(student_vle_path: str, chunksize: int) -> pd.DataFrame:
    tipe_data = None
    jumlah_kosong = None
    jumlah_baris = 0
    for chunk in pd.read_csv(student_vle_path, chunksize=chunksize):
        if tipe_data is None:
            tipe_data = chunk.dtypes
            jumlah_kosong = chunk.isna().sum()
        else:
            jumlah_kosong = jumlah_kosong + chunk.isna().sum()
        jumlah_baris += len(chunk)
    return _ringkasan_kolom("studentVle", tipe_data, jumlah_baris, jumlah_kosong)


def summarize_structure(tables: Dict[str, pd.DataFrame], student_vle_path: str, chunksize: int = 2000000) -> pd.DataFrame:
    ringkasan = [
        _ringkasan_kolom(nama, df.dtypes, len(df), df.isna().sum())
        for nama, df in tables.items()
    ]
    ringkasan.append(_ringkasan_student_vle(student_vle_path, chunksize))
    return pd.concat(ringkasan, ignore_index=True)


def summarize_presentations(courses_df: pd.DataFrame, student_info_df: pd.DataFrame, assessments_df: pd.DataFrame) -> pd.DataFrame:
    per_jenis = (
        assessments_df.groupby(KUNCI_PRESENTASI + ["assessment_type"]).size()
        .unstack("assessment_type", fill_value=0)
        .reindex(columns=list(JENIS_PENILAIAN), fill_value=0)
        .rename(columns=JENIS_PENILAIAN)
    )
    bertenggat = assessments_df.merge(courses_df[KUNCI_PRESENTASI + ["cutoff_day"]], on=KUNCI_PRESENTASI)
    bertenggat = bertenggat[(bertenggat["assessment_type"] != "Exam") & (bertenggat["date"] <= bertenggat["cutoff_day"])]
    per_bertenggat = bertenggat.groupby(KUNCI_PRESENTASI).agg(
        penilaian_non_exam_bertenggat_sampai_cutoff=("id_assessment", "size"),
        bobot_non_exam_bertenggat_sampai_cutoff=("weight", "sum"),
    )
    return (
        courses_df.rename(columns={"module_presentation_length": "panjang_presentasi"})
        .set_index(KUNCI_PRESENTASI)[["panjang_presentasi", "cutoff_day"]]
        .join(student_info_df.groupby(KUNCI_PRESENTASI).size().rename("jumlah_siswa"))
        .join(per_jenis)
        .join(per_bertenggat)
        .fillna(0)
        .astype(int)
        .reset_index()
        .sort_values(KUNCI_PRESENTASI, ignore_index=True)
    )


def summarize_targets(student_info_df: pd.DataFrame, student_registration_df: pd.DataFrame, courses_df: pd.DataFrame) -> pd.DataFrame:
    per_hasil = (
        pd.crosstab(student_info_df["code_module"], student_info_df["final_result"])
        .rename(columns=str.lower)
        .reindex(columns=HASIL_AKHIR, fill_value=0)
    )
    withdrawn_batal = (
        student_info_df[student_info_df["final_result"] == "Withdrawn"]
        .merge(student_registration_df[KUNCI_SISWA + ["date_unregistration"]], on=KUNCI_SISWA, how="left")
        .merge(courses_df[KUNCI_PRESENTASI + ["cutoff_day"]], on=KUNCI_PRESENTASI, how="left")
    )
    withdrawn_batal = withdrawn_batal[
        withdrawn_batal["date_unregistration"].notna() & (withdrawn_batal["date_unregistration"] <= withdrawn_batal["cutoff_day"])
    ]
    per_batal = withdrawn_batal.groupby("code_module").size().rename("withdrawn_batal_sampai_cutoff")
    hasil = (
        student_info_df.groupby("code_module").size().rename("jumlah_siswa_presentasi").to_frame()
        .join(per_hasil)
        .join(per_batal)
        .fillna(0)
        .astype(int)
    )
    hasil["berisiko"] = hasil["fail"] + hasil["withdrawn"]
    hasil["tidak_berisiko"] = hasil["pass"] + hasil["distinction"]
    hasil["proporsi_berisiko"] = (hasil["berisiko"] / hasil["jumlah_siswa_presentasi"]).round(4)
    return hasil.reset_index()
