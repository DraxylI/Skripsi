from typing import Any, Dict, List
import pandas as pd


class ReconstructionRegister:
    """Register keputusan rekonstruksi (Subbab 3.7); kolom tercantum_di_naskah mengacu pada draf revisi 22-10-2026."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def build(self) -> pd.DataFrame:
        ratio = self.cfg["cutoff"]["ratio"]
        week_days = self.cfg["cutoff"]["week_days"]
        pres_order = " < ".join(
            k for k, _ in sorted(self.cfg["cohort"]["presentation_order"].items(), key=lambda kv: kv[1])
        )
        mdl = self.cfg["modeling"]

        records: List[Dict[str, str]] = [
            self._row("R01", "Batas waktu", "Semua varian", "Hari batas waktu",
                      f"cutoff_day = floor({ratio} x module_presentation_length); catatan pada hari cutoff_day ikut dihitung (<=)",
                      "Subbab 3.8.3 (contoh 269 hari -> hari ke-134)", "Ya", "src/data/loader.py"),
            self._row("R02", "Fitur", "A1, A2, A3", "Awal jendela klik",
                      "Hanya catatan studentVle dengan date >= 0; klik sebelum presentasi dimulai tidak dihitung",
                      "Subbab 3.8.1 dan Tabel 3.8: jendela dari hari ke-0", "Ya", "src/features/extractor.py"),
            self._row("R03", "Fitur", "A1, A2, A3", "Cakupan jenis aktivitas VLE",
                      "Seluruh activity_type dijumlahkan tanpa penyaringan jenis aktivitas",
                      "Subbab 3.8.1; artikel sumber tidak menyebut cakupan jenis aktivitas", "Ya", "src/features/extractor.py"),
            self._row("R04", "Fitur", "A1, A2, A3", "Siswa tanpa catatan klik",
                      "Diisi 0",
                      "Ketiadaan klik bermakna tidak aktif (Subbab 3.8.1)", "Ya", "src/features/builder.py"),
            self._row("R05", "Fitur", "A2", "Pembagi rata-rata mingguan",
                      f"A1 / ((cutoff_day + 1) / {week_days}); pembagi sama untuk semua siswa dalam satu presentasi, bukan jumlah minggu aktif siswa",
                      "Subbab 3.8.1; Shou et al. (2024) tidak menyebut pembagi", "Ya", "src/features/extractor.py"),
            self._row("R06", "Fitur", "A3", "Definisi hari aktif",
                      "Jumlah tanggal unik dengan sum_click > 0",
                      "Subbab 3.8.1; Wang (2025) hanya menyebut nama fitur", "Ya", "src/features/extractor.py"),
            self._row("R07", "Fitur", "B1-B3, C1, C2", "Jenis penilaian",
                      "TMA dan CMA; Exam dikeluarkan",
                      "Ujian akhir berada di luar batas waktu (Subbab 3.8.1)", "Ya", "src/features/extractor.py"),
            self._row("R08", "Fitur", "B1-B3, C1, C2", "Nilai bawaan presentasi sebelumnya",
                      "Catatan dengan is_banked = 1 dikeluarkan",
                      "Subbab 3.8.1", "Ya", "src/features/extractor.py"),
            self._row("R09", "Fitur", "B1-B3", "Ketersediaan skor",
                      "Skor tersedia bila date_submitted <= cutoff_day, tanpa syarat tenggat <= cutoff_day",
                      "Subbab 3.8.3", "Ya", "src/features/extractor.py"),
            self._row("R10", "Fitur", "B1-B3, C1, C2", "Pengumpulan tanpa skor",
                      "Pengumpulan dengan score kosong diabaikan",
                      "Subbab 3.8.1; skor kosong tidak dapat dirata-ratakan dan tidak diimputasi", "Ya", "src/features/extractor.py"),
            self._row("R11", "Fitur", "B2", "Bobot penilaian",
                      "Kolom weight tabel assessments; jumlah bobot 0 membuat B2 tidak terdefinisi",
                      "Subbab 3.8.1; penyebab GGG dikeluarkan dari konsep skor (Subbab 3.5.4)", "Ya", "src/features/extractor.py"),
            self._row("R12", "Fitur", "B3", "Penilaian yang dianggap pertama",
                      "Pengumpulan dengan date_submitted paling awal; bila seri, tenggat paling awal, lalu id_assessment terkecil",
                      "Subbab 3.8.1; Wang (2025) tidak mendefinisikan penilaian pertama", "Ya", "src/features/extractor.py"),
            self._row("R13", "Fitur", "C1", "Pembilang rasio tepat waktu",
                      "Penilaian bertenggat <= cutoff_day yang dikumpulkan pada atau sebelum tenggat (date_submitted <= date)",
                      "Subbab 3.8.1; pembilang dan penyebut dari himpunan penilaian yang sama agar rasio <= 1", "Ya", "src/features/extractor.py"),
            self._row("R14", "Fitur", "C1", "Penyebut rasio tepat waktu",
                      "Jumlah penilaian non-Exam bertenggat <= cutoff_day pada presentasi siswa",
                      "Subbab 3.8.3", "Ya", "src/features/extractor.py"),
            self._row("R15", "Fitur", "C1", "Siswa tanpa pengumpulan",
                      "C1 = 0 bagi siswa yang memiliki penilaian jatuh tempo tetapi tidak mengumpulkan satu pun",
                      "Tidak mengumpulkan berarti tidak ada penilaian yang tepat waktu (Subbab 3.8.2)", "Ya", "src/features/builder.py"),
            self._row("R16", "Fitur", "C2", "Himpunan penilaian C2",
                      "Seluruh penilaian yang dikumpulkan <= cutoff_day, termasuk yang bertenggat setelah cutoff_day; negatif = lebih awal. Diuji ulang dengan C2_due (hanya tenggat <= cutoff_day)",
                      "Subbab 3.8.1 dan 3.9; Althibyani (2024) tidak menyebut himpunan penilaian; lihat tabel_sensitivitas_c2.csv", "Ya", "src/features/extractor.py"),
            self._row("R17", "Kohort", "Semua konfigurasi", "Batal registrasi sebelum batas waktu",
                      "Percobaan pertama dengan date_unregistration <= cutoff_day dikeluarkan; batal setelahnya tetap masuk dan berlabel berisiko",
                      "Subbab 3.8.3: pembatalan dianggap sebelum batas waktu bila tanggalnya tidak melewati hari batas waktu", "Ya", "src/data/cohort.py"),
            self._row("R18", "Kohort", "Semua konfigurasi", "Percobaan pertama dan urutan aturan",
                      f"Percobaan pertama ditentukan LEBIH DULU dari seluruh registrasi siswa pada modul yang sama (presentasi paling awal, urutan {pres_order}); baru kemudian aturan batal registrasi diterapkan. Siswa yang percobaan pertamanya batal sebelum batas waktu keluar sepenuhnya, tidak digantikan percobaan berikutnya",
                      "Subbab 3.8.3: percobaan pertama ditentukan dari seluruh registrasi sebelum aturan batal registrasi diterapkan", "Ya", "src/data/cohort.py"),
            self._row("R19", "Sampel", "Semua konfigurasi", "Sampel analisis bersama",
                      "Siswa dengan nilai pada seluruh varian yang dipakai di modul (GGG tanpa B1-B3)",
                      "Subbab 3.8.1", "Ya", "src/features/builder.py"),
            self._row("R20", "Model", "Semua konfigurasi", "Parameter regresi logistik",
                      f"Penalti L2, C={mdl['C']}, solver={mdl['solver']}, max_iter={mdl['max_iter']} (bawaan scikit-learn 100), random_state={mdl['random_state']}, {mdl['n_splits']}-fold stratified dengan shuffle",
                      "Tabel 3.2 dan Subbab 3.8.4: parameter bawaan kecuali max_iter dinaikkan menjadi 1.000 untuk konvergensi", "Ya", "src/models/trainer.py"),
            self._row("R21", "Fitur", "A3", "Padanan number of logins",
                      "Dibaca sebagai jumlah hari aktif (definisi R06); OULAD tidak memiliki tabel atau kolom login, dan studentVle mencatat jumlah klik per siswa, sumber daya, dan hari",
                      "Subbab 3.8.1; Wang (2025) hanya menyebut nama fitur Number of logins; hari adalah satuan waktu terkecil studentVle sehingga sesi login tidak dapat dibedakan", "Ya", "src/features/extractor.py"),
            self._row("R22", "Fitur", "C2", "Padanan number of days",
                      "Dibaca sebagai rata-rata selisih hari bertanda date_submitted - date per penilaian; negatif = lebih awal dari tenggat, positif = terlambat; pengumpulan terlambat tetap dihitung",
                      "Subbab 3.8.1; Althibyani (2024) menulis number of days a student submitted their coursework early tanpa aturan agregasi; satuan hari menunjuk besaran selisih, bukan jumlah kejadian, dan hitungan pengumpulan lebih awal tumpang tindih dengan pembilang C1", "Ya", "src/features/extractor.py"),
        ]
        return pd.DataFrame(records)

    @staticmethod
    def _row(code, stage, variants, parameter, value, basis, in_manuscript, location) -> Dict[str, str]:
        return {
            "kode": code,
            "tahap": stage,
            "varian_terdampak": variants,
            "parameter": parameter,
            "nilai_ditetapkan": value,
            "dasar_atau_alasan": basis,
            "tercantum_di_naskah": in_manuscript,
            "lokasi_kode": location
        }
