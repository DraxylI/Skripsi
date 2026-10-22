import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class FeatureExtractor:
    """Ekstraksi delapan varian fitur (Tabel 3.8) pada jendela hari 0..cutoff_day."""

    def __init__(self, dataset_dir: str, cutoff_map: Dict[tuple, int], week_divisor: float = 7.0):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.cutoff_map = cutoff_map
        self.week_divisor = week_divisor

    def extract_vle_features(self, student_vle_filename: str = "studentVle.csv") -> pd.DataFrame:
        """Konsep A: A1 total klik, A2 = A1 / ((cutoff_day + 1) / 7), A3 jumlah hari dengan klik > 0."""
        vle_path = os.path.join(self.dataset_dir, student_vle_filename)
        usecols = ["code_module", "code_presentation", "id_student", "date", "sum_click"]

        chunks = []
        # Dibaca per chunk karena studentVle ~10 juta baris
        for chunk in pd.read_csv(vle_path, usecols=usecols, chunksize=2000000):
            chunk = chunk[chunk["date"] >= 0]
            chunk["cutoff_day"] = chunk.set_index(["code_module", "code_presentation"]).index.map(self.cutoff_map)
            chunk = chunk[chunk["date"] <= chunk["cutoff_day"]]
            chunks.append(chunk)

        if not chunks:
            return pd.DataFrame(columns=["code_module", "code_presentation", "id_student", "A1", "A2", "A3"])

        vle_filtered = pd.concat(chunks, ignore_index=True)

        vle_active = vle_filtered[vle_filtered["sum_click"] > 0]
        a3_series = vle_active.groupby(["code_module", "code_presentation", "id_student"])["date"].nunique().rename("A3")

        a1_series = vle_filtered.groupby(["code_module", "code_presentation", "id_student"])["sum_click"].sum().rename("A1")

        vle_feats = pd.concat([a1_series, a3_series], axis=1).reset_index()
        vle_feats["A3"] = vle_feats["A3"].fillna(0).astype(int)

        vle_feats["cutoff_day"] = vle_feats.set_index(["code_module", "code_presentation"]).index.map(self.cutoff_map)
        vle_feats["weeks"] = (vle_feats["cutoff_day"] + 1.0) / self.week_divisor
        vle_feats["A2"] = vle_feats["A1"] / vle_feats["weeks"]
        vle_feats.drop(columns=["cutoff_day", "weeks"], inplace=True)

        return vle_feats

    def extract_assessment_features(
        self,
        assessments_df: pd.DataFrame,
        student_assessment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Konsep B dan C dari penilaian non-Exam dan non-banked: B1 rata-rata skor, B2 rata-rata terbobot, B3 skor pertama, C1 rasio tepat waktu, C2 rata-rata (date_submitted - due), C2_due untuk analisis sensitivitas."""
        clean_st_ass = student_assessment_df[student_assessment_df["is_banked"] == 0].copy()
        clean_ass = assessments_df[assessments_df["assessment_type"] != "Exam"].copy()

        merged = pd.merge(clean_st_ass, clean_ass, on="id_assessment", how="inner")
        merged["cutoff_day"] = merged.set_index(["code_module", "code_presentation"]).index.map(self.cutoff_map)

        sub_valid = merged[merged["date_submitted"] <= merged["cutoff_day"]].copy()
        sub_with_score = sub_valid[sub_valid["score"].notna()].copy()

        ass_due = clean_ass.copy()
        ass_due["cutoff_day"] = ass_due.set_index(["code_module", "code_presentation"]).index.map(self.cutoff_map)
        ass_due = ass_due[ass_due["date"] <= ass_due["cutoff_day"]]
        due_counts = ass_due.groupby(["code_module", "code_presentation"])["id_assessment"].count().to_dict()
        # Dipakai builder untuk memberi C1 = 0 bagi siswa tanpa pengumpulan (R15)
        self.due_counts_ = due_counts

        def compute_student_metrics(group: pd.DataFrame) -> pd.Series:
            n_sub = len(group)
            if n_sub == 0:
                return pd.Series({
                    "B1": np.nan, "B2": np.nan, "B3": np.nan,
                    "C2": np.nan, "C2_due": np.nan, "ontime_count": 0
                })

            b1 = group["score"].mean()

            weight_sum = group["weight"].sum()
            b2 = (group["score"] * group["weight"]).sum() / weight_sum if weight_sum > 0 else np.nan

            first_sub = group.sort_values(["date_submitted", "date", "id_assessment"]).iloc[0]
            b3 = first_sub["score"]

            c2 = (group["date_submitted"] - group["date"]).mean()

            # C2_due: C2 pada himpunan penilaian yang sama dengan C1 (analisis sensitivitas, bukan varian utama)
            due_group = group[group["date"] <= group["cutoff_day"]]
            c2_due = (due_group["date_submitted"] - due_group["date"]).mean() if len(due_group) > 0 else np.nan

            # Pembilang C1 dibatasi pada tenggat <= cutoff_day agar sehimpunan dengan N_due (R13)
            ontime_count = (
                (group["date_submitted"] <= group["date"]) & (group["date"] <= group["cutoff_day"])
            ).sum()

            return pd.Series({
                "B1": b1,
                "B2": b2,
                "B3": b3,
                "C2": c2,
                "C2_due": c2_due,
                "ontime_count": ontime_count
            })

        metrics_df = sub_with_score.groupby(
            ["code_module", "code_presentation", "id_student"]
        ).apply(compute_student_metrics, include_groups=False).reset_index()

        metrics_df["N_due"] = metrics_df.set_index(["code_module", "code_presentation"]).index.map(due_counts).fillna(0)
        metrics_df["C1"] = np.where(metrics_df["N_due"] > 0, metrics_df["ontime_count"] / metrics_df["N_due"], np.nan)
        metrics_df.drop(columns=["ontime_count", "N_due"], inplace=True)

        return metrics_df
