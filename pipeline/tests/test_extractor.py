import os
import tempfile
import unittest

import pandas as pd

from src.features.extractor import FeatureExtractor

NAN = float("nan")
CUTOFF_MAP = {("AAA", "2013J"): 100}


def row_of(df, id_student):
    return df[df["id_student"] == id_student].iloc[0]


class AssessmentFeaturesTest(unittest.TestCase):
    def setUp(self):
        assessments_df = pd.DataFrame(
            [
                ("AAA", "2013J", 1, "TMA", 20, 10),
                ("AAA", "2013J", 2, "CMA", 50, 30),
                ("AAA", "2013J", 3, "TMA", 150, 60),
                ("AAA", "2013J", 4, "Exam", 200, 100),
            ],
            columns=["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"],
        )
        student_assessment_df = pd.DataFrame(
            [
                (1, 10, 18, 0, 60.0),
                (2, 10, 55, 0, 80.0),
                (3, 10, 90, 0, 100.0),
                (4, 10, 95, 0, 0.0),
                (1, 11, 10, 1, 90.0),
                (2, 11, 50, 0, 70.0),
                (1, 12, 18, 0, NAN),
                (2, 13, 18, 0, 40.0),
                (1, 13, 18, 0, 50.0),
            ],
            columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
        )
        self.extractor = FeatureExtractor(".", CUTOFF_MAP)
        self.features = self.extractor.extract_assessment_features(assessments_df, student_assessment_df)

    def test_fitur_penilaian_siswa_10_sesuai_rumus(self):
        row = row_of(self.features, 10)

        self.assertAlmostEqual(row["B1"], 80)
        self.assertAlmostEqual(row["B2"], 90)
        self.assertAlmostEqual(row["B3"], 60)
        self.assertAlmostEqual(row["C2"], -19)
        self.assertAlmostEqual(row["C2_due"], 1.5)
        self.assertAlmostEqual(row["C1"], 0.5)

    def test_siswa_11_mengabaikan_nilai_banked(self):
        row = row_of(self.features, 11)

        self.assertAlmostEqual(row["B1"], 70)
        self.assertAlmostEqual(row["B2"], 70)
        self.assertAlmostEqual(row["B3"], 70)
        self.assertAlmostEqual(row["C2"], 0)
        self.assertAlmostEqual(row["C2_due"], 0)
        self.assertAlmostEqual(row["C1"], 0.5)

    def test_siswa_dengan_skor_kosong_tidak_masuk_hasil(self):
        self.assertNotIn(12, self.features["id_student"].tolist())

    def test_seri_tanggal_kumpul_memakai_tenggat_paling_awal(self):
        row = row_of(self.features, 13)

        self.assertAlmostEqual(row["B3"], 50)

    def test_due_counts_hanya_penilaian_non_exam_sampai_cutoff(self):
        self.assertEqual(self.extractor.due_counts_, {("AAA", "2013J"): 2})


class VleFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        student_vle = pd.DataFrame(
            [
                ("AAA", "2013J", 1, 100, -5, 100),
                ("AAA", "2013J", 1, 100, 0, 3),
                ("AAA", "2013J", 1, 101, 0, 2),
                ("AAA", "2013J", 1, 100, 13, 5),
                ("AAA", "2013J", 1, 100, 14, 50),
                ("AAA", "2013J", 2, 100, 3, 0),
            ],
            columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
        )
        student_vle.to_csv(os.path.join(self.tmp.name, "studentVle.csv"), index=False)
        self.features = FeatureExtractor(self.tmp.name, {("AAA", "2013J"): 13}).extract_vle_features()

    def test_fitur_vle_siswa_1_sesuai_rumus(self):
        row = row_of(self.features, 1)

        self.assertAlmostEqual(row["A1"], 10)
        self.assertAlmostEqual(row["A2"], 5.0)
        self.assertEqual(row["A3"], 2)

    def test_siswa_tanpa_klik_aktif_bernilai_nol(self):
        row = row_of(self.features, 2)

        self.assertAlmostEqual(row["A1"], 0)
        self.assertAlmostEqual(row["A2"], 0.0)
        self.assertEqual(row["A3"], 0)
