import os
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


class DatasetBuilder:
    """Menggabungkan fitur ke kohort, menyaring sampel analisis bersama (complete cases), melengkapi tabel atrisi, dan menghitung korelasi antarvarian."""

    def __init__(self, ggg_omit_concepts: List[str] = None):
        self.ggg_omit_concepts = ggg_omit_concepts or ["B"]

    def build_module_datasets(
        self,
        cohort_df: pd.DataFrame,
        vle_df: pd.DataFrame,
        ass_df: pd.DataFrame,
        attrition_df: pd.DataFrame,
        due_counts: Dict[tuple, int] = None
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        """Mengembalikan (dataset per modul, tabel atrisi lengkap, tabel korelasi)."""
        merged = pd.merge(
            cohort_df,
            vle_df[["code_module", "code_presentation", "id_student", "A1", "A2", "A3"]],
            on=["code_module", "code_presentation", "id_student"],
            how="left"
        )
        tanpa_catatan_klik = merged["A1"].isna()
        # Tanpa catatan klik = 0 (R04)
        merged["A1"] = merged["A1"].fillna(0.0)
        merged["A2"] = merged["A2"].fillna(0.0)
        merged["A3"] = merged["A3"].fillna(0).astype(int)

        merged = pd.merge(
            merged,
            ass_df[["code_module", "code_presentation", "id_student", "B1", "B2", "B3", "C1", "C2", "C2_due"]],
            on=["code_module", "code_presentation", "id_student"],
            how="left"
        )

        c1_diisi_nol = pd.Series(False, index=merged.index)
        # C1 = 0 bagi siswa dengan penilaian jatuh tempo tetapi tanpa pengumpulan (R15)
        if due_counts is not None:
            n_due = merged.set_index(["code_module", "code_presentation"]).index.map(due_counts)
            has_due = pd.Series(n_due, index=merged.index).fillna(0) > 0
            c1_diisi_nol = merged["C1"].isna() & has_due
            merged.loc[c1_diisi_nol, "C1"] = 0.0

        module_datasets = {}
        attrition_updates = []
        correlation_records = []
        missing_records = []
        feature_records = []
        varian_dicatat = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2"]

        modules = sorted(merged["code_module"].unique())

        for mod in modules:
            modul_mask = merged["code_module"] == mod
            mdf = merged[modul_mask].copy()
            base_count = len(mdf)
            base_risk_rate = mdf["target"].mean()

            # GGG tanpa konsep B (Subbab 3.5.4)
            if mod == "GGG":
                required_cols = ["A1", "A2", "A3", "C1", "C2"]
            else:
                required_cols = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2"]

            jumlah_tanpa_klik = int((tanpa_catatan_klik & modul_mask).sum())
            diisi_nol_per_varian = {
                "A1": jumlah_tanpa_klik,
                "A2": jumlah_tanpa_klik,
                "A3": jumlah_tanpa_klik,
                "C1": int((c1_diisi_nol & modul_mask).sum()),
            }
            for varian in varian_dicatat:
                missing_records.append({
                    "code_module": mod,
                    "varian": varian,
                    "dipakai_di_modul": varian in required_cols,
                    "kohort_dasar": base_count,
                    "diisi_nol": diisi_nol_per_varian.get(varian, 0),
                    "tetap_kosong": int(mdf[varian].isna().sum()),
                })

            valid_mask = mdf[required_cols].notna().all(axis=1)
            valid_mdf = mdf[valid_mask].copy().reset_index(drop=True)

            for varian in required_cols:
                fitur = valid_mdf[varian]
                feature_records.append({
                    "code_module": mod,
                    "varian": varian,
                    "n": int(fitur.count()),
                    "rata_rata": round(fitur.mean(), 4),
                    "simpangan_baku": round(fitur.std(), 4),
                    "minimum": round(fitur.min(), 4),
                    "kuartil_1": round(fitur.quantile(0.25), 4),
                    "median": round(fitur.median(), 4),
                    "kuartil_3": round(fitur.quantile(0.75), 4),
                    "maksimum": round(fitur.max(), 4),
                    "rata_rata_berisiko": round(fitur[valid_mdf["target"] == 1].mean(), 4),
                    "rata_rata_tidak_berisiko": round(fitur[valid_mdf["target"] == 0].mean(), 4),
                })

            final_count = len(valid_mdf)
            final_risk_rate = valid_mdf["target"].mean() if final_count > 0 else 0.0
            dropped_penilaian = base_count - final_count

            attrition_updates.append({
                "code_module": mod,
                "tanpa_penilaian_lengkap": dropped_penilaian,
                "sampel_analisis_akhir": final_count,
                "proporsi_berisiko_kohort_dasar": round(base_risk_rate, 4),
                "proporsi_berisiko_sampel_akhir": round(final_risk_rate, 4),
                "penurunan_proporsi_berisiko": round(base_risk_rate - final_risk_rate, 4)
            })

            module_datasets[mod] = valid_mdf

            for pair in [("A1", "A2"), ("A1", "A3"), ("A2", "A3")]:
                r_pearson = valid_mdf[pair[0]].corr(valid_mdf[pair[1]], method="pearson")
                r_spearman = valid_mdf[pair[0]].corr(valid_mdf[pair[1]], method="spearman")
                correlation_records.append({
                    "code_module": mod,
                    "konsep": "Aktivitas VLE",
                    "varian_1": pair[0],
                    "varian_2": pair[1],
                    "pearson": round(r_pearson, 4),
                    "spearman": round(r_spearman, 4)
                })

            if mod != "GGG":
                for pair in [("B1", "B2"), ("B1", "B3"), ("B2", "B3")]:
                    r_pearson = valid_mdf[pair[0]].corr(valid_mdf[pair[1]], method="pearson")
                    r_spearman = valid_mdf[pair[0]].corr(valid_mdf[pair[1]], method="spearman")
                    correlation_records.append({
                        "code_module": mod,
                        "konsep": "Skor penilaian berjalan",
                        "varian_1": pair[0],
                        "varian_2": pair[1],
                        "pearson": round(r_pearson, 4),
                        "spearman": round(r_spearman, 4)
                    })

            r_pearson = valid_mdf["C1"].corr(valid_mdf["C2"], method="pearson")
            r_spearman = valid_mdf["C1"].corr(valid_mdf["C2"], method="spearman")
            correlation_records.append({
                "code_module": mod,
                "konsep": "Penyelesaian penilaian",
                "varian_1": "C1",
                "varian_2": "C2",
                "pearson": round(r_pearson, 4),
                "spearman": round(r_spearman, 4)
            })

        update_df = pd.DataFrame(attrition_updates)
        final_attrition_df = pd.merge(attrition_df, update_df, on="code_module", how="inner")
        correlations_df = pd.DataFrame(correlation_records)
        self.missing_summary_ = pd.DataFrame(missing_records)
        self.feature_stats_ = pd.DataFrame(feature_records)

        return module_datasets, final_attrition_df, correlations_df
