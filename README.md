# Pipeline Skripsi OULAD

Pipeline ini menjawab **rumusan masalah 2**: seberapa besar selisih akurasi model ketika satu konsep fitur dihitung dengan definisi yang berbeda. Regresi logistik dipakai sebagai **alat ukur** yang dibekukan. Yang dilaporkan adalah selisih akurasi antarvarian definisi, bukan model terbaik.

Detail metrik pelengkap: [`DOKUMENTASI_METRIK_PELENGKAP.md`](DOKUMENTASI_METRIK_PELENGKAP.md).

---

## 1. Persiapan

### Data
Unduh OULAD (CC-BY 4.0) dari [OU Analyse](https://analyse.kmi.open.ac.uk/open_dataset) atau [UCI ML Repository](https://archive.ics.uci.edu/dataset/349), lalu ekstrak ke folder `dataset/` yang sejajar dengan folder repository ini (`../dataset/`, diatur di `config/config.yaml`). Dataset tidak ikut repository karena `studentVle.csv` berukuran ±433 MB.

```text
<folder kerja>/
├── dataset/      courses.csv, studentInfo.csv, studentRegistration.csv,
│                 assessments.csv, studentAssessment.csv, studentVle.csv, vle.csv
└── pipeline/     (folder repository ini, nama folder bebas)
```

### Environment (Python 3.11)
```powershell
cd pipeline
.\setup_env.ps1          # atau setup_env.bat di CMD
.\.venv\Scripts\Activate.ps1
```
Versi pustaka dikunci di `requirements.txt` (pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2) agar hasil dapat direproduksi persis.

---

## 2. Menjalankan

```powershell
python run_pipeline.py all                 # seluruh tahap (±3 menit)
python run_pipeline.py <tahap>             # satu tahap
python run_pipeline.py all --module AAA    # satu modul saja
```

| Tahap | Isi | Butuh | Keluaran utama |
| :--- | :--- | :--- | :--- |
| `explore` | Ringkasan struktur tabel, presentasi, dan target dari data mentah | dataset | `tabel_eksplorasi_*.csv` |
| `preprocess` | Cutoff, kohort, 8 varian fitur, complete cases, korelasi, register | dataset | `processed/module_*.csv`, `tabel_atrisi_kohort.csv`, `tabel_korelasi_fitur.csv`, `tabel_register_keputusan_rekonstruksi.csv`, `tabel_persiapan_*.csv` |
| `train` | Regresi logistik, stratified 10-fold CV, K0–K5 per modul | preprocess | `models/*.joblib` |
| `predict` | Prediksi out-of-fold | train | `predictions/semua_prediksi.csv` |
| `evaluate` | Kriteria 1-SE, status sensitif/stabil, metrik pelengkap | train | `tabel_evaluasi_lengkap.csv`, `tabel_sensitivitas_konsep.csv`, `tabel_metrik_pelengkap.csv`, dll. |
| `sensitivity` | Uji ulang C2 dengan `C2_due` | preprocess | `tabel_sensitivitas_c2.csv` |
| `robustness` | Uji ketahanan dengan Random Forest | evaluate | `tabel_rf_*.csv` |
| `visualize` | Grafik 300 DPI | evaluate | `figures/*.png` |

Semua path diatur di `config/config.yaml`.

### Pengujian
```powershell
python -m unittest discover -s tests -t . -v
```
Ada 34 test (`unittest` bawaan, tanpa dependency tambahan). Test ini memakai data sintetis kecil untuk memeriksa rumus delapan varian, aturan kohort, complete cases, kriteria 1-SE, dan penanda baseline. Dataset OULAD tidak diperlukan.

---

## 3. Metodologi yang Diimplementasikan

| Aspek | Implementasi | Naskah | Kode |
| :--- | :--- | :--- | :--- |
| Target | Fail/Withdrawn = 1 (berisiko), Pass/Distinction = 0 | 3.4 | `data/loader.py` |
| Batas waktu | `cutoff_day = floor(0,5 × module_presentation_length)`, hari 0 s.d. cutoff (inklusif) | 3.8.3 | `data/loader.py` |
| Kohort | (a) percobaan pertama per modul ditentukan dari **seluruh** registrasi; (b) percobaan pertama yang batal registrasi ≤ cutoff dikeluarkan dan tidak digantikan | 3.8.3 | `data/cohort.py` |
| Sampel analisis | Complete cases pada semua varian yang dipakai di modul (GGG tanpa B) | 3.8.1 | `features/builder.py` |
| Penilaian | TMA dan CMA saja (Exam dikeluarkan), `is_banked = 0`, skor kosong diabaikan | 3.8.1 | `features/extractor.py` |
| Model | `LogisticRegression` L2 bawaan, `max_iter=1000`, `random_state=42` | 3.8.4 | `models/trainer.py` |
| Evaluasi | Stratified 10-fold (shuffle, seed 42), fold identik untuk K0–K5, `StandardScaler` di-fit di fold latih | 3.8.4 | `models/trainer.py` |
| Kriteria | Bermakna bila \|Ā<sub>v</sub> − Ā<sub>acuan</sub>\| > SE<sub>acuan</sub>, SE = s/√10 (Pers. 2.5, 3.1) | 3.8.5 | `evaluation/metrics.py` |
| Status konsep | Sensitif bila ≥ 1 varian pembanding bermakna; selain itu stabil | 3.8.5 | `evaluation/metrics.py` |

### Varian fitur (Tabel 3.8) dan konfigurasi (Tabel 3.10)

| Konsep | Kode | Definisi | Peran |
| :--- | :--- | :--- | :--- |
| Aktivitas VLE | A1 | Total klik hari 0 s.d. cutoff | Acuan |
| | A2 | A1 / ((cutoff + 1) / 7) | Pembanding |
| | A3 | Jumlah hari dengan klik > 0 | Pembanding |
| Skor penilaian berjalan | B1 | Rata-rata skor yang dikumpulkan ≤ cutoff | Acuan |
| | B2 | Rata-rata terbobot (Σ skor × bobot / Σ bobot) | Pembanding |
| | B3 | Skor dengan tanggal pengumpulan paling awal | Pembanding |
| Penyelesaian penilaian | C1 | Tepat waktu / jumlah penilaian bertenggat ≤ cutoff (0 bila tidak mengumpulkan) | Acuan |
| | C2 | Rata-rata (tanggal kumpul − tenggat) | Pembanding |

| Konfigurasi | K0 | K1 | K2 | K3 | K4 | K5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fitur | A1+B1+C1 | **A2**+B1+C1 | **A3**+B1+C1 | A1+**B2**+C1 | A1+**B3**+C1 | A1+B1+**C2** |

Modul GGG hanya menjalankan K0, K1, K2, K5 tanpa konsep B karena semua penilaian bertenggat ≤ cutoff berbobot 0 (Subbab 3.5.4).

### Analisis pelengkap (tidak mengubah status sensitif/stabil)

| Analisis | Isi | Naskah |
| :--- | :--- | :--- |
| Baseline kelas mayoritas | Akurasi ZeroR per modul; kolom `acuan_melampaui_baseline` = (Ā<sub>K0</sub> − baseline) > SE<sub>K0</sub> | 2.2.6, 3.8.5 |
| Rasio \|Δ\|/SE | Membedakan selisih tipis dan kuat | 3.8.5 |
| Metrik pelengkap | Presisi, recall, F1 (kelas berisiko), AUC-ROC per fold yang sama | 2.2.6, 3.8.5 |
| Uji ulang C2 | `C2_due`: C2 hanya dari penilaian bertenggat ≤ cutoff (bukan varian ke-9) | 3.9 |
| Uji ketahanan | Random Forest bawaan (seed 42) pada sampel, fold, dan konfigurasi yang sama. Modul dengan `informatif_rf = False` dinyatakan tidak informatif | 3.8.4, 3.8.5 |
| Korelasi antarvarian | Pearson dan Spearman dalam satu konsep | 3.9 |
| Register rekonstruksi | 22 keputusan parameter yang ditetapkan peneliti | 3.7 |

---

## 4. Keluaran untuk Bab 4

Semua tabel ada di `outputs/tables/`. Nilai berbentuk desimal 0–1; kalikan 100 untuk persen. Repository hanya menyertakan tabel CSV; `processed/`, `models/`, `predictions/`, dan `figures/` dibentuk ulang dengan `python run_pipeline.py all`.

| File | Isi |
| :--- | :--- |
| `tabel_eksplorasi_struktur_dataset.csv` | Jumlah baris, tipe data, dan jumlah serta persen nilai kosong per kolom setiap tabel OULAD |
| `tabel_eksplorasi_presentasi.csv` | Panjang, cutoff, jumlah siswa, dan jumlah penilaian per presentasi, termasuk penilaian non-Exam yang bertenggat sampai cutoff |
| `tabel_eksplorasi_target.csv` | Jumlah siswa per modul menurut hasil akhir, jumlah Withdrawn yang batal registrasi sampai cutoff, serta jumlah dan proporsi berisiko |
| `tabel_persiapan_penyaringan_vle.csv` | Jumlah baris VLE per tahap penyaringan (sebelum hari 0, setelah cutoff, dipakai) dan total klik yang dipakai per modul |
| `tabel_persiapan_penyaringan_penilaian.csv` | Jumlah pengumpulan per tahap penyaringan (banked, Exam, setelah cutoff, skor kosong) dan jumlah yang dipakai per modul |
| `tabel_persiapan_nilai_kosong_varian.csv` | Jumlah siswa tanpa catatan klik, jumlah C1 yang diisi nol, dan nilai kosong tersisa per varian dan modul |
| `tabel_persiapan_statistik_fitur.csv` | Statistik deskriptif setiap varian fitur pada sampel akhir, termasuk rata-rata per kelas target |
| `tabel_atrisi_kohort.csv` | Jumlah unit setelah setiap aturan kohort dan complete cases, serta proporsi berisiko sebelum/sesudah |
| `tabel_evaluasi_lengkap.csv` | Akurasi, SE, baseline, selisih, rasio \|Δ\|/SE, dan status 1-SE per modul × konfigurasi |
| `tabel_ringkasan_per_modul.csv` | Pivot ringkas K0–K5 per modul |
| `tabel_sensitivitas_konsep.csv` | Status sensitif/stabil per konsep × modul (**jawaban RM 2**) |
| `tabel_metrik_pelengkap.csv` | Presisi, recall, F1, AUC-ROC beserta selisih dan status 1-SE informatif |
| `tabel_perbandingan_sensitivitas_metrik.csv` | Jumlah modul sensitif per konsep menurut tiap metrik |
| `tabel_korelasi_fitur.csv` | Korelasi antarvarian dalam satu konsep |
| `tabel_sensitivitas_c2.csv` | Selisih K5 (C2) vs K5 (C2_due) terhadap K0 |
| `tabel_rf_evaluasi_lengkap.csv`, `tabel_rf_metrik_pelengkap.csv`, `tabel_rf_sensitivitas_konsep.csv` | Hasil Random Forest dengan format yang sama |
| `tabel_rf_perbandingan_lr.csv` | Selisih, status, dan arah per varian: regresi logistik vs Random Forest |
| `tabel_register_keputusan_rekonstruksi.csv` | Register keputusan rekonstruksi (R01–R22) |

Grafik di `outputs/figures/`:

| File | Isi |
| :--- | :--- |
| `grafik_akurasi_per_konfigurasi.png` | Akurasi ± SE per konfigurasi dengan garis baseline |
| `grafik_selisih_akurasi_vs_se.png` | \|Δ\| terhadap ambang SE acuan |
| `grafik_sensitivitas_konsep.png` | Matriks status sensitif/stabil |
| `grafik_selisih_semua_metrik.png` | Heatmap selisih kelima metrik terhadap K0 (disarankan sebagai grafik utama pembahasan) |
| `grafik_semua_metrik_per_konfigurasi.png` | Nilai kelima metrik per konfigurasi |
| `grafik_sensitivitas_per_metrik.png` | Jumlah modul sensitif per konsep menurut tiap metrik |
| `grafik_korelasi_fitur.png` | Korelasi antarvarian |
| `grafik_uji_ketahanan_rf.png` | Selisih akurasi per varian: regresi logistik vs Random Forest |

---

## 5. Struktur Folder

```text
pipeline/
├── run_pipeline.py          CLI per tahap
├── config/config.yaml       path, cutoff, kohort, model, konfigurasi K0–K5
├── tests/                   unittest: metrics, cohort, extractor, builder, explorer
├── src/
│   ├── data/                loader.py (tabel + cutoff + target), cohort.py (kohort + atrisi), explorer.py (ringkasan eksplorasi)
│   ├── features/            extractor.py (8 varian), builder.py (complete cases + korelasi)
│   ├── models/              trainer.py (CV + metrik per fold), predictor.py (out-of-fold)
│   ├── evaluation/          metrics.py (1-SE + status), reporter.py (CSV), register.py (R01–R22)
│   └── visualization/       plots.py
└── outputs/                 processed/, models/, predictions/, tables/, figures/
```
