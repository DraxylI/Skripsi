import os
import tempfile
import unittest

import pandas as pd

from src.data.explorer import summarize_presentations, summarize_structure, summarize_targets

NAN = float("nan")


class StructureSummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        student_vle_path = os.path.join(self.tmp.name, "studentVle.csv")
        pd.DataFrame(
            [("AAA", 1, 5), ("AAA", 2, NAN), ("AAA", 3, 1), ("BBB", 4, 2), ("BBB", 5, 7)],
            columns=["code_module", "id_student", "sum_click"],
        ).to_csv(student_vle_path, index=False)
        tables = {"courses": pd.DataFrame({"a": [1, NAN, 3], "b": ["x", "y", "z"]})}
        self.structure = summarize_structure(tables, student_vle_path, chunksize=2).set_index(["tabel", "kolom"])

    def test_tabel_biasa_menghitung_baris_dan_kosong(self):
        row = self.structure.loc[("courses", "a")]

        self.assertEqual(row["jumlah_baris"], 3)
        self.assertEqual(row["jumlah_kosong"], 1)
        self.assertAlmostEqual(row["persen_kosong"], 33.33)

    def test_student_vle_diakumulasi_dari_beberapa_chunk(self):
        row = self.structure.loc[("studentVle", "sum_click")]

        self.assertEqual(row["jumlah_baris"], 5)
        self.assertEqual(row["jumlah_kosong"], 1)
        self.assertAlmostEqual(row["persen_kosong"], 20.0)


class PresentationSummaryTest(unittest.TestCase):
    def setUp(self):
        courses_df = pd.DataFrame(
            [("AAA", "2013J", 100, 50), ("AAA", "2014J", 200, 100)],
            columns=["code_module", "code_presentation", "module_presentation_length", "cutoff_day"],
        )
        assessments_df = pd.DataFrame(
            [
                ("AAA", "2013J", 1, "TMA", 20.0, 10),
                ("AAA", "2013J", 2, "TMA", 60.0, 20),
                ("AAA", "2013J", 3, "CMA", 30.0, 5),
                ("AAA", "2013J", 4, "Exam", NAN, 100),
            ],
            columns=["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"],
        )
        student_info_df = pd.DataFrame(
            [("AAA", "2013J", 1, "Pass"), ("AAA", "2013J", 2, "Fail"), ("AAA", "2013J", 3, "Withdrawn")],
            columns=["code_module", "code_presentation", "id_student", "final_result"],
        )
        self.presentations = summarize_presentations(courses_df, student_info_df, assessments_df).set_index(
            ["code_module", "code_presentation"]
        )

    def test_penilaian_exam_tanpa_tanggal_tidak_dihitung_bertenggat(self):
        row = self.presentations.loc[("AAA", "2013J")]

        self.assertEqual(row["penilaian_exam"], 1)
        self.assertEqual(row["penilaian_non_exam_bertenggat_sampai_cutoff"], 2)

    def test_bobot_hanya_dari_non_exam_bertenggat_sampai_cutoff(self):
        row = self.presentations.loc[("AAA", "2013J")]

        self.assertEqual(row["bobot_non_exam_bertenggat_sampai_cutoff"], 15)

    def test_presentasi_tanpa_siswa_dan_penilaian_bernilai_nol(self):
        row = self.presentations.loc[("AAA", "2014J")]

        self.assertEqual(row["panjang_presentasi"], 200)
        self.assertEqual(row["cutoff_day"], 100)
        self.assertEqual(row["jumlah_siswa"], 0)
        self.assertEqual(row["penilaian_tma"] + row["penilaian_cma"] + row["penilaian_exam"], 0)


class TargetSummaryTest(unittest.TestCase):
    def setUp(self):
        student_info_df = pd.DataFrame(
            [
                ("AAA", "2013J", 1, "Pass"),
                ("AAA", "2013J", 2, "Pass"),
                ("AAA", "2013J", 3, "Fail"),
                ("AAA", "2013J", 4, "Withdrawn"),
                ("AAA", "2013J", 5, "Distinction"),
                ("BBB", "2013J", 6, "Fail"),
                ("BBB", "2013J", 7, "Pass"),
                ("CCC", "2013J", 10, "Withdrawn"),
                ("CCC", "2013J", 11, "Withdrawn"),
                ("CCC", "2013J", 12, "Withdrawn"),
            ],
            columns=["code_module", "code_presentation", "id_student", "final_result"],
        )
        student_registration_df = pd.DataFrame(
            [
                ("AAA", "2013J", 4, 30.0),
                ("CCC", "2013J", 10, 30.0),
                ("CCC", "2013J", 11, 60.0),
                ("CCC", "2013J", 12, NAN),
            ],
            columns=["code_module", "code_presentation", "id_student", "date_unregistration"],
        )
        courses_df = pd.DataFrame(
            [("AAA", "2013J", 50), ("BBB", "2013J", 40), ("CCC", "2013J", 40)],
            columns=["code_module", "code_presentation", "cutoff_day"],
        )
        self.targets = summarize_targets(student_info_df, student_registration_df, courses_df).set_index("code_module")

    def test_hitungan_hasil_akhir_per_modul(self):
        row = self.targets.loc["AAA"]

        self.assertEqual(row["jumlah_siswa_presentasi"], 5)
        self.assertEqual(row["pass"], 2)
        self.assertEqual(row["distinction"], 1)
        self.assertEqual(row["fail"], 1)
        self.assertEqual(row["withdrawn"], 1)
        self.assertEqual(row["berisiko"], 2)
        self.assertEqual(row["tidak_berisiko"], 3)

    def test_withdrawn_batal_sampai_cutoff_hanya_batal_tidak_melewati_cutoff(self):
        row = self.targets.loc["CCC"]

        self.assertEqual(row["withdrawn"], 3)
        self.assertEqual(row["withdrawn_batal_sampai_cutoff"], 1)
        self.assertEqual(self.targets.loc["AAA", "withdrawn_batal_sampai_cutoff"], 1)
        self.assertEqual(self.targets.loc["BBB", "withdrawn_batal_sampai_cutoff"], 0)

    def test_proporsi_berisiko_dibulatkan_empat_desimal(self):
        self.assertAlmostEqual(self.targets.loc["AAA", "proporsi_berisiko"], 0.4)
        self.assertAlmostEqual(self.targets.loc["BBB", "proporsi_berisiko"], 0.5)
