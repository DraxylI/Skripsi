import unittest

import numpy as np

from src.evaluation.metrics import Evaluator

TARGETS = np.array([0] * 80 + [1] * 20)


def cfg(mean, se, feats):
    return {"features": feats, "mean_accuracy": mean, "std_accuracy": se * 10 ** 0.5, "se_accuracy": se}


def run_evaluation(module_code, configs):
    return Evaluator().evaluate_module(module_code, {"configs": configs}, TARGETS)


class EvaluateModuleTest(unittest.TestCase):
    def test_k1_bermakna_membuat_konsep_aktivitas_sensitif(self):
        configs = {
            "K0": cfg(0.81, 0.02, ["A1", "B1", "C1"]),
            "K1": cfg(0.835, 0.02, ["A2", "B1", "C1"]),
            "K5": cfg(0.80, 0.02, ["A1", "B1", "C2"]),
        }
        result = run_evaluation("AAA", configs)
        eval_df = result["evaluation_df"].set_index("konfigurasi")
        sensitivity = result["sensitivity"]

        self.assertEqual(eval_df.loc["K0", "status_1_se"], "Acuan")
        self.assertEqual(eval_df.loc["K1", "status_1_se"], "Bermakna")
        self.assertEqual(eval_df.loc["K5", "status_1_se"], "Tidak Bermakna")
        self.assertEqual(sensitivity["sensitivitas_aktivitas_vle"], "Sensitif")
        self.assertEqual(sensitivity["sensitivitas_skor_penilaian"], "Stabil")
        self.assertEqual(sensitivity["sensitivitas_penyelesaian_penilaian"], "Stabil")
        self.assertAlmostEqual(sensitivity["baseline_zeror"], 0.8)
        self.assertIs(sensitivity["acuan_melampaui_baseline"], False)

    def test_acuan_melampaui_baseline_jika_selisih_lebih_dari_se(self):
        configs = {
            "K0": cfg(0.83, 0.02, ["A1", "B1", "C1"]),
            "K1": cfg(0.835, 0.02, ["A2", "B1", "C1"]),
            "K5": cfg(0.80, 0.02, ["A1", "B1", "C2"]),
        }
        sensitivity = run_evaluation("AAA", configs)["sensitivity"]

        self.assertIs(sensitivity["acuan_melampaui_baseline"], True)

    def test_selisih_sama_dengan_se_tidak_bermakna(self):
        configs = {
            "K0": cfg(0.5, 0.25, ["A1", "B1", "C1"]),
            "K1": cfg(0.75, 0.25, ["A2", "B1", "C1"]),
        }
        eval_df = run_evaluation("AAA", configs)["evaluation_df"].set_index("konfigurasi")

        self.assertEqual(eval_df.loc["K1", "status_1_se"], "Tidak Bermakna")
        self.assertAlmostEqual(eval_df.loc["K1", "rasio_delta_se"], 1.0)

    def test_modul_ggg_konsep_skor_penilaian_tidak_diuji(self):
        configs = {
            "K0": cfg(0.81, 0.02, ["A1", "C1"]),
            "K1": cfg(0.835, 0.02, ["A2", "C1"]),
            "K2": cfg(0.82, 0.02, ["A3", "C1"]),
            "K5": cfg(0.80, 0.02, ["A1", "C2"]),
        }
        sensitivity = run_evaluation("GGG", configs)["sensitivity"]

        self.assertEqual(sensitivity["sensitivitas_skor_penilaian"], "Tidak Diuji")

    def test_baris_k0_berlabel_konsep_acuan(self):
        configs = {
            "K0": cfg(0.81, 0.02, ["A1", "B1", "C1"]),
            "K1": cfg(0.835, 0.02, ["A2", "B1", "C1"]),
        }
        eval_df = run_evaluation("AAA", configs)["evaluation_df"].set_index("konfigurasi")

        self.assertEqual(eval_df.loc["K0", "konsep_diuji"], "Acuan")

    def test_tanpa_konfigurasi_k0_melempar_value_error(self):
        configs = {
            "K1": cfg(0.835, 0.02, ["A2", "B1", "C1"]),
        }

        with self.assertRaises(ValueError):
            run_evaluation("AAA", configs)
