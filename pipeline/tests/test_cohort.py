import unittest

import pandas as pd

from src.data.cohort import CohortFilter

NAN = float("nan")


class FilterCohortTest(unittest.TestCase):
    def setUp(self):
        courses_df = pd.DataFrame(
            [
                ("AAA", "2013J", 200, 100),
                ("AAA", "2014J", 200, 100),
            ],
            columns=["code_module", "code_presentation", "module_presentation_length", "cutoff_day"],
        )
        student_info_df = pd.DataFrame(
            [
                ("AAA", "2013J", 1, 0),
                ("AAA", "2014J", 1, 0),
                ("AAA", "2014J", 2, 1),
                ("AAA", "2013J", 3, 1),
                ("AAA", "2013J", 4, 0),
            ],
            columns=["code_module", "code_presentation", "id_student", "target"],
        )
        student_registration_df = pd.DataFrame(
            [
                ("AAA", "2013J", 1, -10, 50.0),
                ("AAA", "2014J", 1, -10, NAN),
                ("AAA", "2014J", 2, -5, NAN),
                ("AAA", "2013J", 3, -5, 150.0),
                ("AAA", "2013J", 4, -5, 100.0),
            ],
            columns=["code_module", "code_presentation", "id_student", "date_registration", "date_unregistration"],
        )
        self.cohort, self.attrition = CohortFilter({"2013J": 1, "2014J": 2}).filter_cohort(
            student_info_df, student_registration_df, courses_df
        )

    def test_kohort_hanya_siswa_yang_tidak_batal_sebelum_cutoff(self):
        self.assertEqual(set(self.cohort["id_student"].tolist()), {2, 3})

    def test_siswa_batal_di_percobaan_pertama_tidak_digantikan_presentasi_berikutnya(self):
        self.assertNotIn(1, self.cohort["id_student"].tolist())

    def test_baris_kohort_siswa_yang_pindah_presentasi_berasal_dari_presentasi_berikutnya(self):
        row = self.cohort[self.cohort["id_student"] == 2].iloc[0]

        self.assertEqual(row["code_presentation"], "2014J")

    def test_tabel_atrisi_mencatat_setiap_tahap(self):
        row = self.attrition.set_index("code_module").loc["AAA"]

        self.assertEqual(row["total_registrasi_awal"], 5)
        self.assertEqual(row["bukan_percobaan_pertama"], 1)
        self.assertEqual(row["setelah_percobaan_pertama"], 4)
        self.assertEqual(row["batal_sebelum_cutoff"], 2)
        self.assertEqual(row["kohort_dasar"], 2)
