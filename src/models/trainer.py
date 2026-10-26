import os
from typing import Dict, List, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Metrik pelengkap kelas berisiko (target = 1); tidak menentukan status sensitif/stabil
SUPPLEMENTARY_METRICS = ["precision", "recall", "f1", "roc_auc"]


class ModelTrainer:
    """Stratified 10-fold CV dengan fold identik antar konfigurasi dan StandardScaler di-fit per fold latih (Subbab 3.8.4)."""

    def __init__(
        self,
        random_state: int = 42,
        n_splits: int = 10,
        penalty: str = "l2",
        solver: str = "lbfgs",
        C: float = 1.0,
        max_iter: int = 1000,
        models_dir: str = "outputs/models",
        model_type: str = "logistic_regression",
        save_models: bool = True
    ):
        self.random_state = random_state
        self.n_splits = n_splits
        self.penalty = penalty
        self.solver = solver
        self.C = C
        self.max_iter = max_iter
        self.models_dir = os.path.abspath(models_dir)
        self.model_type = model_type
        self.save_models = save_models
        if self.save_models:
            os.makedirs(self.models_dir, exist_ok=True)

    def _make_model(self):
        """Regresi logistik (alat ukur utama) atau Random Forest bawaan (uji ketahanan)."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(random_state=self.random_state, n_jobs=1)

        kwargs = {
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state
        }
        # penalty hanya dikirim bila bukan L2 (bawaan); parameter ini usang sejak scikit-learn 1.8
        if self.penalty != "l2":
            kwargs["penalty"] = self.penalty
        return LogisticRegression(**kwargs)

    def train_module_configurations(
        self,
        module_code: str,
        module_df: pd.DataFrame,
        configurations: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Melatih semua konfigurasi satu modul; mengembalikan metrik per fold, rata-rata, SE, dan prediksi out-of-fold."""
        y = module_df["target"].values
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        # Fold sama untuk K0-K5
        fold_splits = list(skf.split(module_df, y))

        results = {
            "module_code": module_code,
            "total_samples": len(module_df),
            "configs": {}
        }

        for config_name, feature_cols in configurations.items():
            X = module_df[feature_cols].values
            oof_preds = np.zeros(len(module_df), dtype=int)
            oof_probs = np.zeros(len(module_df), dtype=float)
            fold_metrics = []

            for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                model = self._make_model()
                model.fit(X_train_scaled, y_train)

                preds = model.predict(X_val_scaled)
                probs = model.predict_proba(X_val_scaled)[:, 1] if model.classes_.shape[0] > 1 else np.zeros(len(y_val))

                oof_preds[val_idx] = preds
                oof_probs[val_idx] = probs

                acc = (preds == y_val).mean()
                fold_metrics.append({
                    "fold": fold_idx + 1,
                    "accuracy": float(acc),
                    "precision": float(precision_score(y_val, preds, pos_label=1, zero_division=0)),
                    "recall": float(recall_score(y_val, preds, pos_label=1, zero_division=0)),
                    "f1": float(f1_score(y_val, preds, pos_label=1, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_val, probs)),
                    "n_val": len(val_idx)
                })

                if self.save_models:
                    model_filename = f"{module_code}_{config_name}_fold_{fold_idx + 1}.joblib"
                    scaler_filename = f"{module_code}_{config_name}_scaler_fold_{fold_idx + 1}.joblib"
                    joblib.dump(model, os.path.join(self.models_dir, model_filename))
                    joblib.dump(scaler, os.path.join(self.models_dir, scaler_filename))

            # Model penuh hanya untuk arsip, tidak dipakai evaluasi
            if self.save_models:
                full_scaler = StandardScaler()
                X_full_scaled = full_scaler.fit_transform(X)
                full_model = self._make_model()
                full_model.fit(X_full_scaled, y)
                joblib.dump(full_model, os.path.join(self.models_dir, f"{module_code}_{config_name}_full.joblib"))
                joblib.dump(full_scaler, os.path.join(self.models_dir, f"{module_code}_{config_name}_scaler_full.joblib"))

            accuracies = [m["accuracy"] for m in fold_metrics]
            mean_acc = float(np.mean(accuracies))
            std_acc = float(np.std(accuracies, ddof=1))
            se_acc = float(std_acc / np.sqrt(self.n_splits))

            config_result = {
                "features": feature_cols,
                "fold_metrics": fold_metrics,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "se_accuracy": se_acc,
                "oof_preds": oof_preds,
                "oof_probs": oof_probs
            }

            for metric in SUPPLEMENTARY_METRICS:
                values = [m[metric] for m in fold_metrics]
                std_val = float(np.std(values, ddof=1))
                config_result[f"mean_{metric}"] = float(np.mean(values))
                config_result[f"se_{metric}"] = float(std_val / np.sqrt(self.n_splits))

            results["configs"][config_name] = config_result

        return results
