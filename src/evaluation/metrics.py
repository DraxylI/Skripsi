from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Label metrik pelengkap untuk tabel laporan (kelas positif = berisiko)
SUPPLEMENTARY_METRIC_LABELS = {
    "precision": "Presisi",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "AUC-ROC"
}


class Evaluator:
    """Kriteria one-standard-error (Persamaan 3.1) dan status sensitif/stabil per konsep (Subbab 3.8.5)."""

    def __init__(self, concept_mapping: Dict[str, str] = None):
        self.concept_mapping = concept_mapping or {
            "K0": "Acuan",
            "K1": "Aktivitas VLE",
            "K2": "Aktivitas VLE",
            "K3": "Skor penilaian berjalan",
            "K4": "Skor penilaian berjalan",
            "K5": "Penyelesaian penilaian"
        }

    def evaluate_module(
        self,
        module_code: str,
        training_results: Dict[str, Any],
        actual_targets: np.ndarray
    ) -> Dict[str, Any]:
        """Tabel evaluasi akurasi, tabel metrik pelengkap, dan status sensitivitas konsep untuk satu modul."""
        p_risk = float(np.mean(actual_targets))
        baseline_zeror = max(p_risk, 1.0 - p_risk)

        configs_data = training_results["configs"]
        if "K0" not in configs_data:
            raise ValueError(f"Konfigurasi acuan K0 tidak ditemukan pada modul {module_code}")

        k0_mean = configs_data["K0"]["mean_accuracy"]
        k0_se = configs_data["K0"]["se_accuracy"]

        eval_records = []
        for config_name, cdata in configs_data.items():
            mean_acc = cdata["mean_accuracy"]
            std_acc = cdata["std_accuracy"]
            se_acc = cdata["se_accuracy"]
            features_str = "+".join(cdata["features"])

            if config_name == "K0":
                delta = 0.0
                abs_delta = 0.0
                status_1_se = "Acuan"
            else:
                delta = mean_acc - k0_mean
                abs_delta = abs(delta)
                status_1_se = "Bermakna" if abs_delta > k0_se else "Tidak Bermakna"

            eval_records.append({
                "code_module": module_code,
                "konfigurasi": config_name,
                "konsep_diuji": self.concept_mapping.get(config_name, "Acuan"),
                "fitur_masukan": features_str,
                "rata_akurasi": round(mean_acc, 4),
                "std_akurasi": round(std_acc, 4),
                "standard_error": round(se_acc, 4),
                "se_acuan": round(k0_se, 4),
                "selisih_delta": round(delta, 4),
                "abs_delta": round(abs_delta, 4),
                "rasio_delta_se": round(abs_delta / k0_se, 2) if k0_se > 0 else np.nan,
                "status_1_se": status_1_se,
                "baseline_zeror": round(baseline_zeror, 4)
            })

        eval_df = pd.DataFrame(eval_records)

        # status_1_se_informatif hanya informatif; status sensitif/stabil tetap dari akurasi
        supp_records = []
        for metric, label in SUPPLEMENTARY_METRIC_LABELS.items():
            if f"mean_{metric}" not in configs_data["K0"]:
                continue
            ref_mean = configs_data["K0"][f"mean_{metric}"]
            ref_se = configs_data["K0"][f"se_{metric}"]
            for config_name, cdata in configs_data.items():
                mean_val = cdata[f"mean_{metric}"]
                if config_name == "K0":
                    delta = 0.0
                    status = "Acuan"
                else:
                    delta = mean_val - ref_mean
                    status = "Bermakna" if abs(delta) > ref_se else "Tidak Bermakna"
                supp_records.append({
                    "code_module": module_code,
                    "konfigurasi": config_name,
                    "konsep_diuji": self.concept_mapping.get(config_name, "Acuan"),
                    "fitur_masukan": "+".join(cdata["features"]),
                    "metrik": label,
                    "rata_rata": round(mean_val, 4),
                    "standard_error": round(cdata[f"se_{metric}"], 4),
                    "se_acuan": round(ref_se, 4),
                    "selisih_delta": round(delta, 4),
                    "rasio_delta_se": round(abs(delta) / ref_se, 2) if ref_se > 0 else np.nan,
                    "status_1_se_informatif": status
                })
        supp_df = pd.DataFrame(supp_records)

        konsep_a_status = "Stabil"
        for conf in ["K1", "K2"]:
            if conf in configs_data:
                row = eval_df[eval_df["konfigurasi"] == conf]
                if not row.empty and row["status_1_se"].values[0] == "Bermakna":
                    konsep_a_status = "Sensitif"

        if module_code == "GGG":
            konsep_b_status = "Tidak Diuji"
        else:
            konsep_b_status = "Stabil"
            for conf in ["K3", "K4"]:
                if conf in configs_data:
                    row = eval_df[eval_df["konfigurasi"] == conf]
                    if not row.empty and row["status_1_se"].values[0] == "Bermakna":
                        konsep_b_status = "Sensitif"

        konsep_c_status = "Stabil"
        if "K5" in configs_data:
            row = eval_df[eval_df["konfigurasi"] == "K5"]
            if not row.empty and row["status_1_se"].values[0] == "Bermakna":
                konsep_c_status = "Sensitif"

        sensitivity = {
            "code_module": module_code,
            "sensitivitas_aktivitas_vle": konsep_a_status,
            "sensitivitas_skor_penilaian": konsep_b_status,
            "sensitivitas_penyelesaian_penilaian": konsep_c_status,
            "se_acuan": round(k0_se, 4),
            "baseline_zeror": round(baseline_zeror, 4),
            "acuan_melampaui_baseline": bool((k0_mean - baseline_zeror) > k0_se)
        }

        return {
            "evaluation_df": eval_df,
            "supplementary_df": supp_df,
            "sensitivity": sensitivity
        }
