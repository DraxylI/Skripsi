import unittest
import warnings

import pandas as pd

from src.features.builder import DatasetBuilder

NAN = float("nan")


class BuildModuleDatasetsTest(unittest.TestCase):
    def setUp(self):
        cohort_df = pd.DataFrame(
            [
                ("AAA", "2013J", 1, 0),
                ("AAA", "2013J", 2, 1),
                ("AAA", "2013J", 3, 1),
                ("AAA", "2013J", 6, 0),
                ("GGG", "2013J", 4, 0),
                ("GGG", "2013J", 5, 1),
                ("GGG", "2013J", 7, 1),
            ],
            columns=["code_module", "code_presentation", "id_student", "target"],
        )
        vle_df = pd.DataFrame(
            [
                ("AAA", "2013J", 1, 120.0, 60.0, 8),
                ("GGG", "2013J", 4, 40.0, 20.0, 3),
                ("GGG", "2013J", 7, 10.0, 5.0, 1),
            ],
            columns=["code_module", "code_presentation", "id_student", "A1", "A2", "A3"],
        )
        ass_df = pd.DataFrame(
            [
                ("AAA", "2013J", 1, 70.0, 72.0, 65.0, 0.5, -2.0, -1.0),
                ("AAA", "2013J", 3, 50.0, 55.0, 40.0, 1.0, -5.0, -4.0),
                ("AAA", "2013J", 6, 60.0, NAN, 60.0, 0.5, 1.0, 1.0),
                ("GGG", "2013J", 4, NAN, NAN, NAN, 0.5, -1.0, -1.0),
                ("GGG", "2013J", 7, NAN, NAN, NAN, 1.0, -3.0, -3.0),
            ],
            columns=["code_module", "code_presentation", "id_student", "B1", "B2", "B3", "C1", "C2", "C2_due"],
        )
        attrition_df = pd.DataFrame({"code_module": ["AAA", "GGG"]})
        due_counts = {("AAA", "2013J"): 2, ("GGG", "2013J"): 3}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.datasets, self.attrition, self.correlations = DatasetBuilder().build_module_datasets(
                cohort_df, vle_df, ass_df, attrition_df, due_counts
            )

    def test_dataset_aaa_hanya_siswa_dengan_penilaian_lengkap(self):
        self.assertEqual(set(self.datasets["AAA"]["id_student"].tolist()), {1, 3})

    def test_siswa_tanpa_catatan_klik_diisi_nol(self):
        row = self.datasets["AAA"].set_index("id_student").loc[3]

        self.assertAlmostEqual(row["A1"], 0)
        self.assertEqual(row["A3"], 0)

    def test_dataset_ggg_tanpa_konsep_b(self):
        self.assertEqual(set(self.datasets["GGG"]["id_student"].tolist()), {4, 7})

    def test_atrisi_mencatat_jumlah_sampel_akhir_per_modul(self):
        attrition = self.attrition.set_index("code_module")

        self.assertEqual(attrition.loc["AAA", "tanpa_penilaian_lengkap"], 2)
        self.assertEqual(attrition.loc["AAA", "sampel_analisis_akhir"], 2)
        self.assertEqual(attrition.loc["GGG", "tanpa_penilaian_lengkap"], 1)
        self.assertEqual(attrition.loc["GGG", "sampel_analisis_akhir"], 2)

    def test_korelasi_ggg_tanpa_konsep_skor_penilaian_berjalan(self):
        ggg_concepts = self.correlations[self.correlations["code_module"] == "GGG"]["konsep"].tolist()

        self.assertNotIn("Skor penilaian berjalan", ggg_concepts)
