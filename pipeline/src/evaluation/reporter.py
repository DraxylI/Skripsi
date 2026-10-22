import os
from typing import Dict, List, Any
import pandas as pd


class ReportGenerator:
    """Ekspor tabel hasil ke CSV dan ringkasan konsol."""

    def __init__(self, tables_dir: str = "outputs/tables"):
        self.tables_dir = os.path.abspath(tables_dir)
        os.makedirs(self.tables_dir, exist_ok=True)

    def save_reports(
        self,
        eval_dfs: List[pd.DataFrame],
        sensitivity_records: List[Dict[str, Any]],
        attrition_df: pd.DataFrame,
        correlation_df: pd.DataFrame,
        supplementary_dfs: List[pd.DataFrame] = None
    ) -> Dict[str, str]:
        saved_paths = {}

        attrition_path = os.path.join(self.tables_dir, "tabel_atrisi_kohort.csv")
        attrition_df.to_csv(attrition_path, index=False)
        saved_paths["attrition"] = attrition_path

        full_eval_df = pd.concat(eval_dfs, ignore_index=True)
        eval_path = os.path.join(self.tables_dir, "tabel_evaluasi_lengkap.csv")
        full_eval_df.to_csv(eval_path, index=False)
        saved_paths["full_eval"] = eval_path

        sens_df = pd.DataFrame(sensitivity_records)
        sens_path = os.path.join(self.tables_dir, "tabel_sensitivitas_konsep.csv")
        sens_df.to_csv(sens_path, index=False)
        saved_paths["sensitivity"] = sens_path

        pivot_records = []
        for mod, group in full_eval_df.groupby("code_module"):
            row = {"code_module": mod}
            baseline = group["baseline_zeror"].values[0]
            row["Baseline_ZeroR"] = baseline
            for _, r in group.iterrows():
                conf = r["konfigurasi"]
                mean_acc = r["rata_akurasi"]
                se = r["standard_error"]
                status = r["status_1_se"]
                row[f"{conf}_Akurasi"] = f"{mean_acc:.4f} ± {se:.4f}"
                if conf != "K0":
                    row[f"{conf}_Delta"] = f"{r['selisih_delta']:+.4f} ({status})"
            pivot_records.append(row)

        pivot_df = pd.DataFrame(pivot_records)
        pivot_path = os.path.join(self.tables_dir, "tabel_ringkasan_per_modul.csv")
        pivot_df.to_csv(pivot_path, index=False)
        saved_paths["summary"] = pivot_path

        corr_path = os.path.join(self.tables_dir, "tabel_korelasi_fitur.csv")
        correlation_df.to_csv(corr_path, index=False)
        saved_paths["correlations"] = corr_path

        if supplementary_dfs:
            supp_df = pd.concat(supplementary_dfs, ignore_index=True)
            supp_path = os.path.join(self.tables_dir, "tabel_metrik_pelengkap.csv")
            supp_df.to_csv(supp_path, index=False)
            saved_paths["supplementary"] = supp_path

            comp_df = self.build_sensitivity_comparison(full_eval_df, supp_df)
            comp_path = os.path.join(self.tables_dir, "tabel_perbandingan_sensitivitas_metrik.csv")
            comp_df.to_csv(comp_path, index=False)
            saved_paths["sensitivity_comparison"] = comp_path

        return saved_paths

    @staticmethod
    def build_sensitivity_comparison(full_eval_df: pd.DataFrame, supp_df: pd.DataFrame) -> pd.DataFrame:
        """Jumlah modul sensitif per konsep menurut tiap metrik; baris Akurasi adalah hasil utama."""
        acc_df = full_eval_df.rename(columns={"status_1_se": "status"})[
            ["code_module", "konfigurasi", "konsep_diuji", "status"]
        ].assign(metrik="Akurasi")
        other_df = supp_df.rename(columns={"status_1_se_informatif": "status"})[
            ["code_module", "konfigurasi", "konsep_diuji", "status", "metrik"]
        ]
        long_df = pd.concat([acc_df, other_df], ignore_index=True)
        long_df = long_df[long_df["konfigurasi"] != "K0"]

        records = []
        for metric in long_df["metrik"].unique():
            row = {"metrik": metric}
            mdf = long_df[long_df["metrik"] == metric]
            for concept in ["Aktivitas VLE", "Skor penilaian berjalan", "Penyelesaian penilaian"]:
                cdf = mdf[mdf["konsep_diuji"] == concept]
                per_module = cdf.groupby("code_module")["status"].apply(lambda s: (s == "Bermakna").any())
                row[f"{concept}_sensitif"] = int(per_module.sum())
                row[f"{concept}_diuji"] = int(len(per_module))
                row[f"{concept}_modul_sensitif"] = ",".join(per_module[per_module].index)
            records.append(row)
        return pd.DataFrame(records)

    def generate_console_summary(self, full_eval_df: pd.DataFrame, sens_df: pd.DataFrame) -> str:
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("RANGKUMAN HASIL EKSPERIMEN: ANALISIS DAMPAK VARIASI DEFINISI FITUR")
        summary_lines.append("=" * 80)

        summary_lines.append("\n1. Ringkasan Status Sensitivitas per Konsep Fitur:")
        for col, label in [
            ("sensitivitas_aktivitas_vle", "Konsep A (Aktivitas VLE)"),
            ("sensitivitas_skor_penilaian", "Konsep B (Skor Penilaian Berjalan)"),
            ("sensitivitas_penyelesaian_penilaian", "Konsep C (Penyelesaian Penilaian)")
        ]:
            vc = sens_df[col].value_counts().to_dict()
            sens_count = vc.get("Sensitif", 0)
            stab_count = vc.get("Stabil", 0)
            na_count = vc.get("Tidak Diuji", 0)
            summary_lines.append(f"   • {label}: Sensitif={sens_count} modul, Stabil={stab_count} modul (Tidak Diuji={na_count})")

        summary_lines.append("\n2. Detail Akurasi per Modul dan Konfigurasi:")
        summary_lines.append(
            f"{'Modul':<7} {'Konf':<5} {'Rata Akurasi ± SE':<20} {'Delta':<10} {'|Delta|':<10} {'SE Acuan':<10} {'Status 1-SE':<15}"
        )
        summary_lines.append("-" * 80)
        for _, row in full_eval_df.iterrows():
            summary_lines.append(
                f"{row['code_module']:<7} {row['konfigurasi']:<5} "
                f"{row['rata_akurasi']:.4f} ± {row['standard_error']:.4f}    "
                f"{row['selisih_delta']:+.4f}    "
                f"{row['abs_delta']:.4f}    "
                f"{row['se_acuan']:.4f}    "
                f"{row['status_1_se']:<15}"
            )

        summary_lines.append("=" * 80)
        return "\n".join(summary_lines)
