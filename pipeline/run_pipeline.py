import os
import sys
import argparse
import time
import yaml
import joblib
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.data.loader import OULADLoader
from src.data.cohort import CohortFilter
from src.features.extractor import FeatureExtractor
from src.features.builder import DatasetBuilder
from src.models.trainer import ModelTrainer
from src.models.predictor import ModelPredictor
from src.evaluation.metrics import Evaluator
from src.evaluation.reporter import ReportGenerator
from src.evaluation.register import ReconstructionRegister
from src.visualization.plots import PlotGenerator


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_preprocess(cfg: dict, target_module: str = None) -> dict:
    print("\n" + "=" * 60)
    print(">>> CHECKPOINT 1: PRAPROSES DATA & EKSTRAKSI FITUR")
    print("=" * 60)
    t0 = time.time()

    dataset_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["dataset_dir"]))
    cutoff_ratio = cfg["cutoff"]["ratio"]
    week_divisor = cfg["cutoff"]["week_days"]
    pres_order = cfg["cohort"]["presentation_order"]
    proc_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["processed_dir"]))
    tables_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["tables_dir"]))
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print(f"1. Memuat dataset OULAD dari: {dataset_dir} ...")
    loader = OULADLoader(dataset_dir, cutoff_ratio=cutoff_ratio)
    courses_df = loader.load_courses(cfg["paths"]["courses_file"])
    student_info_df = loader.load_student_info(cfg["paths"]["student_info_file"])
    student_reg_df = loader.load_student_registration(cfg["paths"]["student_registration_file"])
    assessments_df = loader.load_assessments(cfg["paths"]["assessments_file"])
    st_ass_df = loader.load_student_assessment(cfg["paths"]["student_assessment_file"])
    cutoff_map = loader.get_cutoff_map(courses_df)

    print(f"   • Total baris courses: {len(courses_df)}")
    print(f"   • Total baris studentInfo: {len(student_info_df)}")
    print(f"   • Total baris studentRegistration: {len(student_reg_df)}")
    print(f"   • Total baris assessments: {len(assessments_df)}")
    print(f"   • Total baris studentAssessment: {len(st_ass_df)}")

    print("\n2. Menjalankan penyaringan kohort (percobaan pertama, lalu batal registrasi <= cutoff)...")
    cohort_filter = CohortFilter(pres_order)
    cohort_df, attrition_df = cohort_filter.filter_cohort(student_info_df, student_reg_df, courses_df)
    print(f"   • Kohort dasar tersaring: {len(cohort_df)} siswa-presentasi")

    print("\n3. Mengekstraksi 8 varian definisi fitur...")
    extractor = FeatureExtractor(dataset_dir, cutoff_map, week_divisor=week_divisor)
    print("   • Mengekstrak Konsep A (Aktivitas VLE: A1, A2, A3) dari studentVle.csv...")
    vle_df = extractor.extract_vle_features(cfg["paths"]["student_vle_file"])
    print(f"     Aktivitas VLE terekstraksi untuk {len(vle_df)} siswa.")

    print("   • Mengekstrak Konsep B & C (Skor & Penyelesaian: B1, B2, B3, C1, C2)...")
    ass_feats_df = extractor.extract_assessment_features(assessments_df, st_ass_df)
    print(f"     Fitur penilaian terekstraksi untuk {len(ass_feats_df)} siswa.")

    print("\n4. Menyusun dataset per modul (Complete Cases pada varian yang diuji)...")
    builder = DatasetBuilder(ggg_omit_concepts=cfg["modules"]["ggg_omit_concepts"])
    module_datasets, final_attrition_df, correlations_df = builder.build_module_datasets(
        cohort_df, vle_df, ass_feats_df, attrition_df, due_counts=extractor.due_counts_
    )

    for mod, mdf in module_datasets.items():
        if target_module and mod != target_module:
            continue
        save_path = os.path.join(proc_dir, f"module_{mod}.csv")
        mdf.to_csv(save_path, index=False)
        print(f"   • Modul {mod}: {len(mdf)} sampel tersimpan di {save_path}")

    attrition_path = os.path.join(tables_dir, "tabel_atrisi_kohort.csv")
    final_attrition_df.to_csv(attrition_path, index=False)
    corr_path = os.path.join(tables_dir, "tabel_korelasi_fitur.csv")
    correlations_df.to_csv(corr_path, index=False)
    print(f"   • Tabel atrisi kohort tersimpan di: {attrition_path}")
    print(f"   • Tabel korelasi fitur tersimpan di: {corr_path}")

    register_path = os.path.join(tables_dir, "tabel_register_keputusan_rekonstruksi.csv")
    ReconstructionRegister(cfg).build().to_csv(register_path, index=False)
    print(f"   • Register keputusan rekonstruksi tersimpan di: {register_path}")

    print(f"\n[OK] Tahap Preprocessing selesai dalam {time.time()-t0:.2f} detik.")
    return {
        "module_datasets": module_datasets,
        "attrition_df": final_attrition_df,
        "correlations_df": correlations_df
    }


def run_train(cfg: dict, target_module: str = None) -> dict:
    print("\n" + "=" * 60)
    print(">>> CHECKPOINT 2: PELATIHAN MODEL & CROSS-VALIDATION")
    print("=" * 60)
    t0 = time.time()

    proc_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["processed_dir"]))
    models_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["models_dir"]))
    os.makedirs(models_dir, exist_ok=True)

    modules_to_run = [target_module] if target_module else cfg["modules"]["all"]

    trainer = ModelTrainer(
        random_state=cfg["modeling"]["random_state"],
        n_splits=cfg["modeling"]["n_splits"],
        penalty=cfg["modeling"]["penalty"],
        solver=cfg["modeling"]["solver"],
        C=cfg["modeling"]["C"],
        max_iter=cfg["modeling"]["max_iter"],
        models_dir=models_dir
    )

    all_training_results = {}

    for mod in modules_to_run:
        mod_file = os.path.join(proc_dir, f"module_{mod}.csv")
        if not os.path.exists(mod_file):
            raise FileNotFoundError(
                f"Data praproses modul {mod} belum ditemukan ({mod_file}). Jalankan checkpoint 'preprocess' terlebih dahulu!"
            )
        
        module_df = pd.read_csv(mod_file)
        confs = cfg["configurations"]["ggg"] if mod == "GGG" else cfg["configurations"]["standard"]

        print(f"\nMelatih Modul {mod} ({len(module_df)} siswa, {len(confs)} konfigurasi)...")
        mod_results = trainer.train_module_configurations(mod, module_df, confs)
        all_training_results[mod] = mod_results

        for cname, cdata in mod_results["configs"].items():
            print(f"   • {cname} ({'+'.join(cdata['features'])}): Akurasi = {cdata['mean_accuracy']:.4f} ± {cdata['se_accuracy']:.4f}")

    summary_path = os.path.join(models_dir, "training_summary.joblib")
    joblib.dump(all_training_results, summary_path)
    print(f"\n[OK] Model pelatihan & checkpoint tersimpan di: {models_dir}")
    print(f"Selesai dalam {time.time()-t0:.2f} detik.")
    return all_training_results


def run_predict(cfg: dict, target_module: str = None) -> list:
    print("\n" + "=" * 60)
    print(">>> CHECKPOINT 3: PEMBANGKITAN PREDIKSI (OUT-OF-FOLD)")
    print("=" * 60)
    t0 = time.time()

    proc_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["processed_dir"]))
    models_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["models_dir"]))
    pred_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["predictions_dir"]))
    os.makedirs(pred_dir, exist_ok=True)

    summary_path = os.path.join(models_dir, "training_summary.joblib")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"File riwayat pelatihan {summary_path} belum ada. Jalankan 'train' terlebih dahulu!")

    training_results = joblib.load(summary_path)
    predictor = ModelPredictor(predictions_dir=pred_dir)

    all_preds = []
    modules_to_run = [target_module] if target_module else list(training_results.keys())

    for mod in modules_to_run:
        mod_file = os.path.join(proc_dir, f"module_{mod}.csv")
        module_df = pd.read_csv(mod_file)
        mod_res = training_results[mod]
        dfs = predictor.format_predictions(mod, module_df, mod_res)
        all_preds.extend(dfs)
        print(f"   • Prediksi out-of-fold untuk Modul {mod} tersimpan ({len(dfs)} file konfigurasi)")

    if all_preds:
        consolidated_df = pd.concat(all_preds, ignore_index=True)
        cons_path = os.path.join(pred_dir, "semua_prediksi.csv")
        consolidated_df.to_csv(cons_path, index=False)
        print(f"   • Konsolidasi seluruh prediksi tersimpan di: {cons_path}")

    print(f"\n[OK] Tahap Predict selesai dalam {time.time()-t0:.2f} detik.")
    return all_preds


def run_evaluate(cfg: dict, target_module: str = None) -> dict:
    print("\n" + "=" * 60)
    print(">>> CHECKPOINT 4: EVALUASI ONE-STANDARD-ERROR RULE & LAPORAN")
    print("=" * 60)
    t0 = time.time()

    proc_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["processed_dir"]))
    models_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["models_dir"]))
    tables_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["tables_dir"]))
    os.makedirs(tables_dir, exist_ok=True)

    summary_path = os.path.join(models_dir, "training_summary.joblib")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"File {summary_path} belum ada. Jalankan 'train' terlebih dahulu!")

    training_results = joblib.load(summary_path)
    evaluator = Evaluator(concept_mapping=cfg.get("concept_mapping"))
    reporter = ReportGenerator(tables_dir=tables_dir)

    eval_dfs = []
    supplementary_dfs = []
    sensitivity_records = []
    modules_to_run = [target_module] if target_module else list(training_results.keys())

    for mod in modules_to_run:
        mod_file = os.path.join(proc_dir, f"module_{mod}.csv")
        module_df = pd.read_csv(mod_file)
        targets = module_df["target"].values
        mod_res = training_results[mod]

        eval_out = evaluator.evaluate_module(mod, mod_res, targets)
        eval_dfs.append(eval_out["evaluation_df"])
        supplementary_dfs.append(eval_out["supplementary_df"])
        sensitivity_records.append(eval_out["sensitivity"])

    attrition_path = os.path.join(tables_dir, "tabel_atrisi_kohort.csv")
    corr_path = os.path.join(tables_dir, "tabel_korelasi_fitur.csv")
    attrition_df = pd.read_csv(attrition_path) if os.path.exists(attrition_path) else pd.DataFrame()
    correlation_df = pd.read_csv(corr_path) if os.path.exists(corr_path) else pd.DataFrame()

    saved_paths = reporter.save_reports(
        eval_dfs, sensitivity_records, attrition_df, correlation_df, supplementary_dfs
    )
    full_eval_df = pd.concat(eval_dfs, ignore_index=True)
    sens_df = pd.DataFrame(sensitivity_records)

    console_summary = reporter.generate_console_summary(full_eval_df, sens_df)
    print(console_summary)
    not_informative = sens_df.loc[~sens_df["acuan_melampaui_baseline"], "code_module"].tolist()
    print(f"Modul dengan akurasi K0 tidak melampaui baseline > 1 SE: {', '.join(not_informative) or '-'}")
    print(f"\n[OK] Tabel laporan CSV tersimpan di {tables_dir}")
    print(f"Selesai dalam {time.time()-t0:.2f} detik.")
    return {
        "full_eval_df": full_eval_df,
        "sens_df": sens_df,
        "saved_paths": saved_paths
    }


def run_sensitivity(cfg: dict, target_module: str = None) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(">>> CHECKPOINT 5: ANALISIS SENSITIVITAS KEPUTUSAN REKONSTRUKSI")
    print("=" * 60)
    t0 = time.time()

    proc_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["processed_dir"]))
    models_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["models_dir"], "sensitivitas"))
    tables_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["tables_dir"]))
    os.makedirs(tables_dir, exist_ok=True)

    sens_cfg = cfg["sensitivity_analysis"]["c2_tenggat_sampai_cutoff"]
    main_var, sens_var = sens_cfg["varian_utama"], sens_cfg["varian_sensitivitas"]
    print(f"Analisis: {sens_cfg['deskripsi']}")

    trainer = ModelTrainer(
        random_state=cfg["modeling"]["random_state"],
        n_splits=cfg["modeling"]["n_splits"],
        penalty=cfg["modeling"]["penalty"],
        solver=cfg["modeling"]["solver"],
        C=cfg["modeling"]["C"],
        max_iter=cfg["modeling"]["max_iter"],
        models_dir=models_dir
    )

    records = []
    modules_to_run = [target_module] if target_module else cfg["modules"]["all"]
    for mod in modules_to_run:
        module_df = pd.read_csv(os.path.join(proc_dir, f"module_{mod}.csv"))
        # K0, K5, dan K5_sens dibandingkan pada siswa yang sama
        subset = module_df[module_df[sens_var].notna()].reset_index(drop=True)
        confs = sens_cfg["ggg"] if mod == "GGG" else sens_cfg["standard"]
        res = trainer.train_module_configurations(mod, subset, confs)["configs"]

        k0 = res["K0"]
        row = {
            "code_module": mod,
            "sampel_sensitivitas": len(subset),
            "dikeluarkan_varian_tak_terdefinisi": len(module_df) - len(subset),
            "akurasi_K0": round(k0["mean_accuracy"], 4),
            "se_K0": round(k0["se_accuracy"], 4),
        }
        for name, label in [("K5", main_var), ("K5_sens", sens_var)]:
            delta = res[name]["mean_accuracy"] - k0["mean_accuracy"]
            row[f"selisih_akurasi_{label}"] = round(delta, 4)
            row[f"rasio_delta_se_{label}"] = round(abs(delta) / k0["se_accuracy"], 2)
            row[f"status_1_se_{label}"] = "Bermakna" if abs(delta) > k0["se_accuracy"] else "Tidak Bermakna"
            row[f"selisih_recall_{label}"] = round(res[name]["mean_recall"] - k0["mean_recall"], 4)
        row["status_sama"] = row[f"status_1_se_{main_var}"] == row[f"status_1_se_{sens_var}"]
        records.append(row)
        print(f"   • {mod}: selisih {main_var} = {row[f'selisih_akurasi_{main_var}']:+.4f} ({row[f'status_1_se_{main_var}']}), "
              f"{sens_var} = {row[f'selisih_akurasi_{sens_var}']:+.4f} ({row[f'status_1_se_{sens_var}']})")

    sens_df = pd.DataFrame(records)
    out_path = os.path.join(tables_dir, "tabel_sensitivitas_c2.csv")
    sens_df.to_csv(out_path, index=False)
    n_same = int(sens_df["status_sama"].sum())
    print(f"\nStatus 1-SE sama pada {n_same} dari {len(sens_df)} modul. Tabel tersimpan di: {out_path}")
    print(f"[OK] Tahap Sensitivitas selesai dalam {time.time()-t0:.2f} detik.")
    return sens_df


def _direction(delta: float, tol: float = 0.0005) -> str:
    """Arah selisih terhadap K0; selisih di bawah 0,05 poin dianggap tetap."""
    if delta > tol:
        return "naik"
    if delta < -tol:
        return "turun"
    return "tetap"


def run_robustness(cfg: dict, target_module: str = None) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(">>> CHECKPOINT 6: UJI KETAHANAN ALAT UKUR (MODEL KEDUA)")
    print("=" * 60)
    t0 = time.time()

    proc_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["processed_dir"]))
    tables_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["tables_dir"]))
    os.makedirs(tables_dir, exist_ok=True)

    rob_cfg = cfg["robustness_model"]
    label = rob_cfg["label"]
    print(f"Model uji ketahanan: {label} (parameter bawaan, random_state={cfg['modeling']['random_state']})")
    print("Sampel, pembagian fold, dan konfigurasi K0-K5 identik dengan regresi logistik.")

    # Fold identik dengan tahap train: seed dan data modul sama
    trainer = ModelTrainer(
        random_state=cfg["modeling"]["random_state"],
        n_splits=cfg["modeling"]["n_splits"],
        model_type=rob_cfg["model_type"],
        save_models=False
    )
    evaluator = Evaluator(concept_mapping=cfg.get("concept_mapping"))

    eval_dfs, supp_dfs, sens_records = [], [], []
    modules_to_run = [target_module] if target_module else cfg["modules"]["all"]
    for mod in modules_to_run:
        module_df = pd.read_csv(os.path.join(proc_dir, f"module_{mod}.csv"))
        confs = cfg["configurations"]["ggg"] if mod == "GGG" else cfg["configurations"]["standard"]
        res = trainer.train_module_configurations(mod, module_df, confs)
        out = evaluator.evaluate_module(mod, res, module_df["target"].values)
        eval_dfs.append(out["evaluation_df"])
        supp_dfs.append(out["supplementary_df"])
        sens_records.append(out["sensitivity"])
        print(f"   • {mod}: " + ", ".join(
            f"{c}={d['mean_accuracy']:.4f}" for c, d in res["configs"].items()
        ))

    rf_eval = pd.concat(eval_dfs, ignore_index=True)
    rf_supp = pd.concat(supp_dfs, ignore_index=True)
    rf_sens = pd.DataFrame(sens_records)
    rf_eval.to_csv(os.path.join(tables_dir, "tabel_rf_evaluasi_lengkap.csv"), index=False)
    rf_supp.to_csv(os.path.join(tables_dir, "tabel_rf_metrik_pelengkap.csv"), index=False)
    rf_sens.to_csv(os.path.join(tables_dir, "tabel_rf_sensitivitas_konsep.csv"), index=False)

    lr_eval = pd.read_csv(os.path.join(tables_dir, "tabel_evaluasi_lengkap.csv"))
    cols = ["code_module", "konfigurasi", "konsep_diuji", "rata_akurasi", "selisih_delta", "rasio_delta_se", "status_1_se"]
    comp = lr_eval[cols].merge(rf_eval[cols], on=["code_module", "konfigurasi", "konsep_diuji"], suffixes=("_lr", "_rf"))
    comp = comp[comp["konfigurasi"] != "K0"].copy()
    comp["arah_lr"] = comp["selisih_delta_lr"].apply(_direction)
    comp["arah_rf"] = comp["selisih_delta_rf"].apply(_direction)
    comp["arah_sama"] = comp["arah_lr"] == comp["arah_rf"]
    comp["status_sama"] = comp["status_1_se_lr"] == comp["status_1_se_rf"]
    informative = rf_sens.set_index("code_module")["acuan_melampaui_baseline"]
    comp["informatif_rf"] = comp["code_module"].map(informative)
    comp_path = os.path.join(tables_dir, "tabel_rf_perbandingan_lr.csv")
    comp.to_csv(comp_path, index=False)

    lr_sens = pd.read_csv(os.path.join(tables_dir, "tabel_sensitivitas_konsep.csv"))
    concept_cols = ["sensitivitas_aktivitas_vle", "sensitivitas_skor_penilaian", "sensitivitas_penyelesaian_penilaian"]
    print("\nJumlah modul sensitif (regresi logistik vs " + label + "):")
    for col in concept_cols:
        n_lr = int((lr_sens[col] == "Sensitif").sum())
        n_rf = int((rf_sens[col] == "Sensitif").sum())
        print(f"   • {col.replace('sensitivitas_', '')}: {n_lr} vs {n_rf}")
    print(f"Status 1-SE sama pada {int(comp['status_sama'].sum())} dari {len(comp)} perbandingan; "
          f"arah selisih sama pada {int(comp['arah_sama'].sum())} dari {len(comp)}.")
    not_informative = rf_sens.loc[~rf_sens["acuan_melampaui_baseline"], "code_module"].tolist()
    print(f"Modul dengan hasil {label} tidak informatif (akurasi K0 tidak melampaui baseline > 1 SE): {', '.join(not_informative) or '-'}")
    print(f"Tabel tersimpan di: {tables_dir} (awalan tabel_rf_)")
    print(f"[OK] Tahap Uji Ketahanan selesai dalam {time.time()-t0:.2f} detik.")
    return comp


def run_visualize(cfg: dict) -> list:
    print("\n" + "=" * 60)
    print(">>> CHECKPOINT 7: PEMBANGKITAN GRAFIK VISUALISASI")
    print("=" * 60)
    t0 = time.time()

    tables_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["tables_dir"]))
    figures_dir = os.path.normpath(os.path.join(script_dir, cfg["paths"]["figures_dir"]))
    os.makedirs(figures_dir, exist_ok=True)

    eval_path = os.path.join(tables_dir, "tabel_evaluasi_lengkap.csv")
    sens_path = os.path.join(tables_dir, "tabel_sensitivitas_konsep.csv")
    corr_path = os.path.join(tables_dir, "tabel_korelasi_fitur.csv")

    if not os.path.exists(eval_path) or not os.path.exists(sens_path):
        raise FileNotFoundError("Tabel evaluasi belum ditemukan. Jalankan 'evaluate' terlebih dahulu!")

    full_eval_df = pd.read_csv(eval_path)
    sens_df = pd.read_csv(sens_path)
    correlation_df = pd.read_csv(corr_path) if os.path.exists(corr_path) else pd.DataFrame()
    supp_path = os.path.join(tables_dir, "tabel_metrik_pelengkap.csv")
    comp_path = os.path.join(tables_dir, "tabel_perbandingan_sensitivitas_metrik.csv")
    supp_df = pd.read_csv(supp_path) if os.path.exists(supp_path) else pd.DataFrame()
    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame()
    rf_comp_path = os.path.join(tables_dir, "tabel_rf_perbandingan_lr.csv")
    rf_comp_df = pd.read_csv(rf_comp_path) if os.path.exists(rf_comp_path) else pd.DataFrame()

    plotter = PlotGenerator(figures_dir=figures_dir)
    plots = plotter.plot_all(full_eval_df, sens_df, correlation_df, supp_df, comp_df, rf_comp_df)

    print(f"Grafik publikasi berhasil dibuat ({len(plots)} file):")
    for p in plots:
        print(f"   • {p}")

    print(f"\n[OK] Tahap Visualisasi selesai dalam {time.time()-t0:.2f} detik.")
    return plots


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Penelitian Analisis Dampak Variasi Definisi Fitur pada OULAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh Penggunaan:
  python run_pipeline.py all                   # Menjalankan seluruh pipeline end-to-end
  python run_pipeline.py preprocess            # Menjalankan praproses & ekstraksi fitur
  python run_pipeline.py train                 # Melatih model Stratified 10-Fold CV
  python run_pipeline.py predict               # Menghasilkan prediksi out-of-fold
  python run_pipeline.py evaluate              # Evaluasi One-Standard-Error Rule & ekspor tabel CSV
  python run_pipeline.py sensitivity           # Analisis sensitivitas keputusan rekonstruksi (C2)
  python run_pipeline.py robustness            # Uji ketahanan alat ukur dengan Random Forest
  python run_pipeline.py visualize             # Menghasilkan grafik visualisasi perbandingan
  python run_pipeline.py all --module AAA      # Menjalankan pipeline hanya untuk Modul AAA
        """
    )

    parser.add_argument(
        "stage",
        choices=["preprocess", "train", "predict", "evaluate", "sensitivity", "robustness", "visualize", "all"],
        help="Tahap/checkpoint pipeline yang akan dijalankan."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path ke file konfigurasi YAML (default: config/config.yaml)."
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Modul spesifik yang ingin dijalankan (misal: AAA). Default: semua modul."
    )

    args = parser.parse_args()

    config_full_path = os.path.normpath(os.path.join(script_dir, args.config))
    if not os.path.exists(config_full_path):
        print(f"Error: File konfigurasi tidak ditemukan di {config_full_path}")
        sys.exit(1)

    cfg = load_config(config_full_path)

    stage = args.stage.lower()
    mod = args.module.upper() if args.module else None

    start_all = time.time()
    if stage == "preprocess":
        run_preprocess(cfg, target_module=mod)
    elif stage == "train":
        run_train(cfg, target_module=mod)
    elif stage == "predict":
        run_predict(cfg, target_module=mod)
    elif stage == "evaluate":
        run_evaluate(cfg, target_module=mod)
    elif stage == "sensitivity":
        run_sensitivity(cfg, target_module=mod)
    elif stage == "robustness":
        run_robustness(cfg, target_module=mod)
    elif stage == "visualize":
        run_visualize(cfg)
    elif stage == "all":
        print("================================================================================")
        print("MEMULAI EKSEKUSI PIPELINE END-TO-END PENELITIAN SKRIPSI OULAD")
        print("================================================================================")
        run_preprocess(cfg, target_module=mod)
        run_train(cfg, target_module=mod)
        run_predict(cfg, target_module=mod)
        run_evaluate(cfg, target_module=mod)
        run_sensitivity(cfg, target_module=mod)
        run_robustness(cfg, target_module=mod)
        run_visualize(cfg)
        print("================================================================================")
        print(f"SELURUH PIPELINE SELESAI DIEKSEKUSI DALAM {time.time()-start_all:.2f} DETIK")
        print("================================================================================")


if __name__ == "__main__":
    main()

