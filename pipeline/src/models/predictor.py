import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class ModelPredictor:
    """Menyimpan prediksi out-of-fold per modul dan konfigurasi ke CSV."""

    def __init__(self, predictions_dir: str = "outputs/predictions"):
        self.predictions_dir = os.path.abspath(predictions_dir)
        os.makedirs(self.predictions_dir, exist_ok=True)

    def format_predictions(
        self,
        module_code: str,
        module_df: pd.DataFrame,
        training_results: Dict[str, Any]
    ) -> List[pd.DataFrame]:
        saved_dfs = []
        base_cols = ["code_module", "code_presentation", "id_student", "target"]

        for config_name, conf_data in training_results["configs"].items():
            pred_df = module_df[base_cols].copy()
            pred_df["konfigurasi"] = config_name
            pred_df["prediksi_label"] = conf_data["oof_preds"]
            pred_df["prediksi_probabilitas"] = np.round(conf_data["oof_probs"], 5)
            pred_df["prediksi_benar"] = (pred_df["prediksi_label"] == pred_df["target"]).astype(int)

            out_filename = f"prediksi_{module_code}_{config_name}.csv"
            out_path = os.path.join(self.predictions_dir, out_filename)
            pred_df.to_csv(out_path, index=False)
            saved_dfs.append(pred_df)

        return saved_dfs
