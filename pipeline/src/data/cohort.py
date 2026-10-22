from typing import Dict, Tuple
import pandas as pd


class CohortFilter:
    """Penetapan kohort (Subbab 3.8.3): percobaan pertama per modul, lalu keluarkan yang batal registrasi <= cutoff_day."""

    def __init__(self, presentation_order: Dict[str, int]):
        self.presentation_order = presentation_order

    def filter_cohort(
        self,
        student_info_df: pd.DataFrame,
        student_registration_df: pd.DataFrame,
        courses_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Mengembalikan (kohort, tabel atrisi per modul)."""
        df = pd.merge(
            student_info_df,
            courses_df[["code_module", "code_presentation", "module_presentation_length", "cutoff_day"]],
            on=["code_module", "code_presentation"],
            how="inner"
        )

        df = pd.merge(
            df,
            student_registration_df[["code_module", "code_presentation", "id_student", "date_registration", "date_unregistration"]],
            on=["code_module", "code_presentation", "id_student"],
            how="inner"
        )

        modules = sorted(df["code_module"].unique())
        attrition_records = []

        initial_counts = df.groupby("code_module").size().to_dict()

        df["pres_order"] = df["code_presentation"].map(self.presentation_order).fillna(99)
        df = df.sort_values(["code_module", "id_student", "pres_order"])
        df_first = df.drop_duplicates(subset=["code_module", "id_student"], keep="first").copy()
        first_counts = df_first.groupby("code_module").size().to_dict()

        unregistered_before_cutoff = (
            df_first["date_unregistration"].notna() & (df_first["date_unregistration"] <= df_first["cutoff_day"])
        )
        df_cohort = df_first[~unregistered_before_cutoff].copy()
        cohort_counts = df_cohort.groupby("code_module").size().to_dict()

        for mod in modules:
            init_c = initial_counts.get(mod, 0)
            first_c = first_counts.get(mod, 0)
            cohort_c = cohort_counts.get(mod, 0)
            attrition_records.append({
                "code_module": mod,
                "total_registrasi_awal": init_c,
                "bukan_percobaan_pertama": init_c - first_c,
                "setelah_percobaan_pertama": first_c,
                "batal_sebelum_cutoff": first_c - cohort_c,
                "kohort_dasar": cohort_c
            })

        attrition_df = pd.DataFrame(attrition_records)
        return df_cohort, attrition_df
