import os
from typing import Dict, List, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class PlotGenerator:
    """Grafik 300 DPI untuk Bab 4."""

    def __init__(self, figures_dir: str = "outputs/figures"):
        self.figures_dir = os.path.abspath(figures_dir)
        os.makedirs(self.figures_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", font="sans-serif")

    def plot_all(
        self,
        full_eval_df: pd.DataFrame,
        sens_df: pd.DataFrame,
        correlations_df: pd.DataFrame,
        supp_df: pd.DataFrame = None,
        comparison_df: pd.DataFrame = None,
        rf_comparison_df: pd.DataFrame = None
    ) -> List[str]:
        """Menyimpan semua grafik yang datanya tersedia dan mengembalikan daftar path file keluaran."""
        saved_plots = []
        p1 = self.plot_accuracy_per_configuration(full_eval_df)
        p2 = self.plot_delta_vs_se(full_eval_df)
        p3 = self.plot_concept_sensitivity_matrix(sens_df)
        p4 = self.plot_feature_correlations(correlations_df)

        saved_plots.extend([p1, p2, p3, p4])

        if supp_df is not None and not supp_df.empty:
            metrics_long = self._combine_metrics(full_eval_df, supp_df)
            saved_plots.append(self.plot_all_metrics_per_configuration(metrics_long, full_eval_df))
            saved_plots.append(self.plot_delta_heatmap_all_metrics(metrics_long))
        if comparison_df is not None and not comparison_df.empty:
            saved_plots.append(self.plot_sensitivity_count_by_metric(comparison_df))
        if rf_comparison_df is not None and not rf_comparison_df.empty:
            saved_plots.append(self.plot_lr_vs_rf_delta(rf_comparison_df))
        return saved_plots

    def plot_accuracy_per_configuration(self, full_eval_df: pd.DataFrame) -> str:
        """grafik_akurasi_per_konfigurasi.png: akurasi ± SE per konfigurasi dengan garis baseline."""
        modules = sorted(full_eval_df["code_module"].unique())
        n_mods = len(modules)

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9), sharey=True)
        axes = axes.flatten()

        palette = {
            "K0": "#2b5c8f",
            "K1": "#4682b4",
            "K2": "#5f9ea0",
            "K3": "#e67e22",
            "K4": "#f39c12",
            "K5": "#27ae60"
        }

        for idx, mod in enumerate(modules):
            ax = axes[idx]
            mdf = full_eval_df[full_eval_df["code_module"] == mod]
            configs = mdf["konfigurasi"].tolist()
            means = mdf["rata_akurasi"].values
            ses = mdf["standard_error"].values
            colors = [palette.get(c, "#7f8c8d") for c in configs]
            baseline = mdf["baseline_zeror"].values[0]

            x_pos = np.arange(len(configs))
            ax.bar(x_pos, means, yerr=ses, capsize=4, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
            ax.axhline(baseline, color="#c0392b", linestyle="--", linewidth=1.5, label=f"ZeroR: {baseline:.3f}")

            ax.set_title(f"Modul {mod}", fontsize=12, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(configs, fontsize=9)
            ax.set_ylim([0.45, 0.95])
            ax.legend(loc="lower right", fontsize=8)

        if n_mods < len(axes):
            fig.delaxes(axes[7])

        fig.text(0.5, 0.02, "Konfigurasi Varian Definisi Fitur", ha="center", fontsize=12, fontweight="bold")
        fig.text(0.01, 0.5, "Rata-rata Akurasi (Stratified 10-Fold CV)", va="center", rotation="vertical", fontsize=12, fontweight="bold")
        plt.suptitle("Perbandingan Akurasi Model Prediksi per Konfigurasi pada Setiap Modul OULAD", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])

        out_path = os.path.join(self.figures_dir, "grafik_akurasi_per_konfigurasi.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_delta_vs_se(self, full_eval_df: pd.DataFrame) -> str:
        """grafik_selisih_akurasi_vs_se.png: |Delta| per varian dibandingkan ambang SE acuan (One-Standard-Error Rule)."""
        comp_df = full_eval_df[full_eval_df["konfigurasi"] != "K0"].copy()

        plt.figure(figsize=(14, 7))
        comp_df["mod_conf"] = comp_df["code_module"] + " - " + comp_df["konfigurasi"] + " (" + comp_df["konsep_diuji"] + ")"

        comp_df.sort_values(["code_module", "konfigurasi"], inplace=True)

        y_pos = np.arange(len(comp_df))
        bar_colors = ["#e74c3c" if status == "Bermakna" else "#3498db" for status in comp_df["status_1_se"]]

        plt.barh(y_pos, comp_df["abs_delta"], color=bar_colors, alpha=0.85, edgecolor="black", label="|Delta| Varian")
        plt.plot(comp_df["se_acuan"], y_pos, "k|", markersize=14, markeredgewidth=2.5, label="Ambang Batas SE Acuan")

        plt.yticks(y_pos, comp_df["mod_conf"], fontsize=9)
        plt.xlabel("Besaran Selisih Akurasi Terhadap Acuan (|Delta|)", fontsize=11, fontweight="bold")
        plt.title("Evaluasi Besaran Dampak Variasi Fitur Berdasarkan One-Standard-Error Rule\n(|Delta| > SE_acuan menandakan dampak bermakna/sensitif)", fontsize=13, fontweight="bold")

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor="#e74c3c", edgecolor="black", label="Bermakna (|Delta| > SE acuan)"),
            Patch(facecolor="#3498db", edgecolor="black", label="Tidak Bermakna (|Delta| <= SE acuan)"),
            Line2D([0], [0], color="black", marker="|", linestyle="None", markersize=12, markeredgewidth=2, label="Ambang SE Acuan")
        ]
        plt.legend(handles=legend_elements, loc="lower right", fontsize=10)
        plt.tight_layout()

        out_path = os.path.join(self.figures_dir, "grafik_selisih_akurasi_vs_se.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_concept_sensitivity_matrix(self, sens_df: pd.DataFrame) -> str:
        """grafik_sensitivitas_konsep.png: matriks sensitif/stabil konsep fitur per modul."""
        matrix_data = sens_df.set_index("code_module")[
            ["sensitivitas_aktivitas_vle", "sensitivitas_skor_penilaian", "sensitivitas_penyelesaian_penilaian"]
        ].rename(columns={
            "sensitivitas_aktivitas_vle": "Konsep A\n(Aktivitas VLE)",
            "sensitivitas_skor_penilaian": "Konsep B\n(Skor Penilaian)",
            "sensitivitas_penyelesaian_penilaian": "Konsep C\n(Penyelesaian Penilaian)"
        })

        numeric_map = {"Stabil": 0, "Sensitif": 1, "Tidak Diuji": -1}
        encoded = matrix_data.map(lambda x: numeric_map.get(x, 0))

        plt.figure(figsize=(9, 6))
        cmap = matplotlib.colors.ListedColormap(["#bdc3c7", "#2ecc71", "#e74c3c"])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        ax = sns.heatmap(
            encoded,
            annot=matrix_data.values,
            fmt="",
            cmap=cmap,
            norm=norm,
            cbar=False,
            linewidths=1.5,
            linecolor="white",
            annot_kws={"fontsize": 11, "fontweight": "bold", "color": "white"}
        )

        plt.title("Matriks Sensitivitas Konsep Fitur Terhadap Variasi Definisi per Modul", fontsize=13, fontweight="bold", pad=15)
        plt.ylabel("Modul Kursus", fontsize=11, fontweight="bold")
        plt.xlabel("Konsep Fitur Dinamis", fontsize=11, fontweight="bold")
        plt.tight_layout()

        out_path = os.path.join(self.figures_dir, "grafik_sensitivitas_konsep.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_feature_correlations(self, correlations_df: pd.DataFrame) -> str:
        """grafik_korelasi_fitur.png: korelasi Pearson dan Spearman antarvarian dalam satu konsep per modul."""
        plt.figure(figsize=(12, 6))

        correlations_df["pasangan"] = correlations_df["code_module"] + " : " + correlations_df["varian_1"] + "-" + correlations_df["varian_2"]

        correlations_df.sort_values(["konsep", "code_module"], inplace=True)

        x = np.arange(len(correlations_df))
        width = 0.35

        plt.bar(x - width/2, correlations_df["pearson"], width, label="Pearson r", color="#3498db", alpha=0.85)
        plt.bar(x + width/2, correlations_df["spearman"], width, label="Spearman rho", color="#9b59b6", alpha=0.85)

        plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        plt.xticks(x, correlations_df["pasangan"], rotation=90, fontsize=8)
        plt.ylabel("Koefisien Korelasi", fontsize=11, fontweight="bold")
        plt.title("Korelasi Antarvarian Definisi Fitur dalam Satu Konsep per Modul\n(Memvalidasi kesetaraan definisional dan multikolinieritas antarvarian)", fontsize=13, fontweight="bold")
        plt.ylim([-1.05, 1.05])
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()

        out_path = os.path.join(self.figures_dir, "grafik_korelasi_fitur.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    # Palet tetap per entitas (CVD-safe); divergen biru = lebih baik, merah = lebih buruk
    CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    CONFIG_ORDER = ["K0", "K1", "K2", "K3", "K4", "K5"]
    CONFIG_LABELS = {
        "K0": "K0 (acuan A1+B1+C1)", "K1": "K1 (A2)", "K2": "K2 (A3)",
        "K3": "K3 (B2)", "K4": "K4 (B3)", "K5": "K5 (C2)"
    }
    VARIANT_SWAP = {"K1": "A2", "K2": "A3", "K3": "B2", "K4": "B3", "K5": "C2"}
    METRIC_ORDER = ["Akurasi", "Presisi", "Recall", "F1", "AUC-ROC"]
    TEXT_PRIMARY = "#0b0b0b"
    TEXT_SECONDARY = "#52514e"
    GRID = "#e4e3df"
    SURFACE = "#fcfcfb"

    @staticmethod
    def _combine_metrics(full_eval_df: pd.DataFrame, supp_df: pd.DataFrame) -> pd.DataFrame:
        acc = full_eval_df.rename(columns={
            "rata_akurasi": "rata_rata",
            "status_1_se": "status"
        })[["code_module", "konfigurasi", "rata_rata", "standard_error", "selisih_delta", "status"]].assign(metrik="Akurasi")
        supp = supp_df.rename(columns={"status_1_se_informatif": "status"})[
            ["code_module", "konfigurasi", "rata_rata", "standard_error", "selisih_delta", "status", "metrik"]
        ]
        return pd.concat([acc, supp], ignore_index=True)

    def _style_axis(self, ax) -> None:
        ax.set_facecolor(self.SURFACE)
        ax.grid(axis="y", color=self.GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(self.TEXT_SECONDARY)
        ax.tick_params(colors=self.TEXT_SECONDARY, labelsize=9)

    def plot_all_metrics_per_configuration(self, metrics_long: pd.DataFrame, full_eval_df: pd.DataFrame) -> str:
        """grafik_semua_metrik_per_konfigurasi.png: kelima metrik per konfigurasi dan modul; konfigurasi tak dijalankan tampil sebagai celah."""
        modules = sorted(metrics_long["code_module"].unique())
        colors = dict(zip(self.CONFIG_ORDER, self.CATEGORICAL))
        n_conf = len(self.CONFIG_ORDER)
        width = 0.8 / n_conf
        x = np.arange(len(modules))

        fig, axes = plt.subplots(nrows=len(self.METRIC_ORDER), ncols=1, figsize=(14, 17), sharex=True)
        fig.patch.set_facecolor(self.SURFACE)

        for ax, metric in zip(axes, self.METRIC_ORDER):
            mdf = metrics_long[metrics_long["metrik"] == metric]
            self._style_axis(ax)
            for c_idx, conf in enumerate(self.CONFIG_ORDER):
                cdf = mdf[mdf["konfigurasi"] == conf].set_index("code_module").reindex(modules)
                offset = (c_idx - (n_conf - 1) / 2) * width
                ax.bar(
                    x + offset, cdf["rata_rata"].values, width=width,
                    yerr=cdf["standard_error"].values,
                    error_kw={"elinewidth": 0.8, "capsize": 0, "ecolor": self.TEXT_SECONDARY},
                    color=colors[conf], edgecolor=self.SURFACE, linewidth=0.8,
                    label=self.CONFIG_LABELS[conf]
                )
            if metric == "Akurasi":
                baselines = full_eval_df.groupby("code_module")["baseline_zeror"].first().reindex(modules)
                for i, b in enumerate(baselines.values):
                    ax.hlines(b, i - 0.42, i + 0.42, colors=self.TEXT_PRIMARY, linestyles="--", linewidth=1.2,
                              label="Baseline kelas mayoritas" if i == 0 else None)
                title = "Akurasi (metrik utama)"
            elif metric == "AUC-ROC":
                title = "AUC-ROC (metrik pelengkap)"
            else:
                title = f"{metric} kelas berisiko (metrik pelengkap)"
            ax.set_ylim(0, 1.0)
            ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=self.TEXT_PRIMARY)
            ax.set_ylabel("Rata-rata 10 fold", fontsize=10, color=self.TEXT_SECONDARY)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([f"Modul {m}" for m in modules], fontsize=10, color=self.TEXT_PRIMARY)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 0.965), labelcolor=self.TEXT_PRIMARY)
        fig.suptitle("Akurasi dan Metrik Pelengkap per Konfigurasi Varian pada Setiap Modul",
                     fontsize=15, fontweight="bold", color=self.TEXT_PRIMARY, y=0.985)
        fig.text(0.5, 0.005,
                 "Garis tipis pada batang = ±1 SE. Modul GGG tidak menjalankan K3 dan K4 (konsep skor tidak diuji).",
                 ha="center", fontsize=9, color=self.TEXT_SECONDARY)
        plt.tight_layout(rect=[0, 0.015, 1, 0.95])

        out_path = os.path.join(self.figures_dir, "grafik_semua_metrik_per_konfigurasi.png")
        plt.savefig(out_path, dpi=300, facecolor=self.SURFACE)
        plt.close()
        return out_path

    def plot_delta_heatmap_all_metrics(self, metrics_long: pd.DataFrame, clip_points: float = 10.0) -> str:
        """grafik_selisih_semua_metrik.png: selisih tiap varian terhadap K0 dalam poin persen untuk kelima metrik; * = melampaui 1 SE acuan."""
        comp = metrics_long[metrics_long["konfigurasi"] != "K0"].copy()
        comp["baris"] = (comp["code_module"] + "  " + comp["konfigurasi"]
                         + " (" + comp["konfigurasi"].map(self.VARIANT_SWAP) + ")")
        rows = comp.sort_values(["code_module", "konfigurasi"])["baris"].unique()

        values = comp.pivot_table(index="baris", columns="metrik", values="selisih_delta") \
            .reindex(index=rows, columns=self.METRIC_ORDER) * 100
        status = comp.pivot_table(index="baris", columns="metrik", values="status", aggfunc="first") \
            .reindex(index=rows, columns=self.METRIC_ORDER)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("divergen", ["#e34948", "#f0efec", "#2a78d6"])
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-clip_points, vcenter=0.0, vmax=clip_points)

        fig, ax = plt.subplots(figsize=(10, 15))
        fig.patch.set_facecolor(self.SURFACE)
        ax.imshow(values.clip(-clip_points, clip_points).values, cmap=cmap, norm=norm, aspect="auto")

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values.iat[i, j]
                if pd.isna(v):
                    continue
                mark = "*" if status.iat[i, j] == "Bermakna" else ""
                # Hindari label -0,0
                text = ("0,0" if round(v, 1) == 0 else f"{v:+.1f}".replace(".", ",")) + mark
                ink = "white" if abs(v) >= 0.6 * clip_points else self.TEXT_PRIMARY
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=ink,
                        fontweight="bold" if mark else "normal")

        modules = [r.split()[0] for r in rows]
        for i in range(1, len(modules)):
            if modules[i] != modules[i - 1]:
                ax.axhline(i - 0.5, color=self.SURFACE, linewidth=3)
        for j in range(1, len(self.METRIC_ORDER)):
            ax.axvline(j - 0.5, color=self.SURFACE, linewidth=2)
        ax.axvline(0.5, color=self.TEXT_PRIMARY, linewidth=1.5)

        # Matikan grid seaborn agar tidak menembus sel heatmap
        ax.grid(False)
        ax.set_xticks(range(len(self.METRIC_ORDER)))
        ax.set_xticklabels(["Akurasi\n(utama)", "Presisi", "Recall", "F1", "AUC-ROC"],
                           fontsize=10, color=self.TEXT_PRIMARY)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=9, color=self.TEXT_PRIMARY)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.025, pad=0.02, aspect=40)
        cbar.set_label(f"Selisih terhadap K0 dalam poin persen (biru = lebih baik, merah = lebih buruk; "
                       f"warna dipotong pada ±{clip_points:.0f})", fontsize=9, color=self.TEXT_SECONDARY)
        cbar.ax.tick_params(labelsize=8, colors=self.TEXT_SECONDARY)
        cbar.outline.set_visible(False)

        ax.set_title("Selisih Setiap Varian Pembanding terhadap Acuan pada Kelima Metrik\n"
                     "* = selisih > 1 SE acuan (akurasi: status resmi; metrik lain: informatif)",
                     fontsize=12, fontweight="bold", color=self.TEXT_PRIMARY, pad=40)
        plt.tight_layout()

        out_path = os.path.join(self.figures_dir, "grafik_selisih_semua_metrik.png")
        plt.savefig(out_path, dpi=300, facecolor=self.SURFACE)
        plt.close()
        return out_path

    def plot_sensitivity_count_by_metric(self, comparison_df: pd.DataFrame) -> str:
        """grafik_sensitivitas_per_metrik.png: jumlah modul sensitif per konsep menurut tiap metrik; akurasi sebagai hasil utama."""
        concepts = ["Aktivitas VLE", "Skor penilaian berjalan", "Penyelesaian penilaian"]
        metrics = [m for m in self.METRIC_ORDER if m in comparison_df["metrik"].values]
        colors = dict(zip(self.METRIC_ORDER, self.CATEGORICAL))
        cdf = comparison_df.set_index("metrik")
        n_m = len(metrics)
        width = 0.8 / n_m
        x = np.arange(len(concepts))

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(self.SURFACE)
        self._style_axis(ax)

        for m_idx, metric in enumerate(metrics):
            counts = [int(cdf.loc[metric, f"{c}_sensitif"]) for c in concepts]
            tested = [int(cdf.loc[metric, f"{c}_diuji"]) for c in concepts]
            offset = (m_idx - (n_m - 1) / 2) * width
            label = f"{metric} (utama)" if metric == "Akurasi" else metric
            bars = ax.bar(x + offset, counts, width=width, color=colors[metric],
                          edgecolor=self.SURFACE, linewidth=0.8, label=label)
            for bar, cnt, tot in zip(bars, counts, tested):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{cnt}/{tot}",
                        ha="center", va="bottom", fontsize=8.5, color=self.TEXT_PRIMARY)

        ax.set_xticks(x)
        ax.set_xticklabels(concepts, fontsize=11, color=self.TEXT_PRIMARY)
        ax.set_ylim(0, 8.2)
        ax.set_yticks(range(0, 8))
        ax.set_ylabel("Jumlah modul sensitif", fontsize=10, color=self.TEXT_SECONDARY)
        ax.set_title("Jumlah Modul Sensitif per Konsep Fitur menurut Setiap Metrik\n"
                     "(sensitif bila sedikitnya satu varian pembanding berselisih > 1 SE acuan)",
                     fontsize=12, fontweight="bold", color=self.TEXT_PRIMARY, loc="left")
        ax.legend(loc="upper left", ncol=5, frameon=False, fontsize=9, labelcolor=self.TEXT_PRIMARY)
        plt.tight_layout()

        out_path = os.path.join(self.figures_dir, "grafik_sensitivitas_per_metrik.png")
        plt.savefig(out_path, dpi=300, facecolor=self.SURFACE)
        plt.close()
        return out_path

    def plot_lr_vs_rf_delta(self, rf_comparison_df: pd.DataFrame) -> str:
        """grafik_uji_ketahanan_rf.png: selisih akurasi varian terhadap K0, regresi logistik vs Random Forest; * = RF tidak informatif."""
        df = rf_comparison_df.copy().sort_values(["code_module", "konfigurasi"])
        df["baris"] = df["code_module"] + "  " + df["konfigurasi"] + " (" + df["konfigurasi"].map(self.VARIANT_SWAP) + ")"
        not_informative = ~df["informatif_rf"].astype(bool).values if "informatif_rf" in df.columns else np.zeros(len(df), dtype=bool)
        df.loc[not_informative, "baris"] = df.loc[not_informative, "baris"] + "  *"
        rows = list(df["baris"])
        y = np.arange(len(rows))[::-1]

        fig, ax = plt.subplots(figsize=(10, 13))
        fig.patch.set_facecolor(self.SURFACE)
        ax.set_facecolor(self.SURFACE)

        modules = list(df["code_module"])
        for i in range(1, len(modules)):
            if modules[i] != modules[i - 1]:
                ax.axhline(y[i] + 0.5, color=self.GRID, linewidth=1)
        ax.axvline(0, color=self.TEXT_SECONDARY, linewidth=1)

        series = [
            ("lr", "Regresi logistik (alat ukur utama)", self.CATEGORICAL[0], "o", 0.15),
            ("rf", "Random Forest (uji ketahanan)", self.CATEGORICAL[1], "D", -0.15),
        ]
        for key, label, color, marker, offset in series:
            vals = df[f"selisih_delta_{key}"].values * 100
            sig = (df[f"status_1_se_{key}"] == "Bermakna").values
            ax.scatter(vals[sig], y[sig] + offset, s=64, marker=marker, color=color,
                       edgecolor=self.SURFACE, linewidth=1.2, zorder=3)
            ax.scatter(vals[~sig], y[~sig] + offset, s=64, marker=marker, facecolor=self.SURFACE,
                       edgecolor=color, linewidth=1.8, zorder=3)
            ax.scatter([], [], s=64, marker=marker, color=color, label=label)

        ax.scatter([], [], s=64, marker="o", color=self.TEXT_SECONDARY, label="Penuh = bermakna (> 1 SE acuan)")
        ax.scatter([], [], s=64, marker="o", facecolor=self.SURFACE, edgecolor=self.TEXT_SECONDARY,
                   linewidth=1.8, label="Kosong = tidak bermakna")
        if not_informative.any():
            ax.plot([], [], " ", label="* = hasil Random Forest tidak informatif (akurasi K0 tidak melampaui baseline > 1 SE)")

        ax.set_yticks(y)
        ax.set_yticklabels(rows, fontsize=9, color=self.TEXT_PRIMARY)
        for tick, flag in zip(ax.get_yticklabels(), not_informative):
            if flag:
                tick.set_color(self.TEXT_SECONDARY)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.grid(axis="x", color=self.GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(self.TEXT_SECONDARY)
        ax.tick_params(axis="x", colors=self.TEXT_SECONDARY, labelsize=9)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Selisih akurasi terhadap K0 (poin persen); kanan = lebih baik, kiri = lebih buruk",
                      fontsize=10, color=self.TEXT_SECONDARY)
        ax.set_title("Uji Ketahanan Alat Ukur: Selisih Akurasi per Varian\n"
                     "Regresi Logistik vs Random Forest pada sampel dan fold yang sama",
                     fontsize=12, fontweight="bold", color=self.TEXT_PRIMARY, loc="left")
        # Legenda di bawah sumbu agar tidak menutupi titik data
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.05), ncol=2, fontsize=9,
                  frameon=False, labelcolor=self.TEXT_PRIMARY)
        plt.tight_layout()

        out_path = os.path.join(self.figures_dir, "grafik_uji_ketahanan_rf.png")
        plt.savefig(out_path, dpi=300, facecolor=self.SURFACE)
        plt.close()
        return out_path
