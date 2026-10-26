# Metrik Pelengkap

Presisi, recall, F1, dan AUC-ROC dilaporkan di samping akurasi sebagai **analisis pelengkap** (Subbab 2.2.6 dan 3.8.5). Hasilnya tidak mengubah status bermakna, sensitif, atau stabil. Status itu tetap hanya ditentukan oleh akurasi dengan Persamaan 3.1.

## Ringkasan

| Hal | Ketentuan |
| :--- | :--- |
| Kelas positif | Berisiko (Fail/Withdrawn, `target = 1`) |
| Skema | Dihitung per fold pada stratified 10-fold yang sama dengan akurasi, lalu dirata-ratakan. SE = s/√10 |
| Ambang | 0,5 untuk presisi, recall, dan F1. AUC-ROC dihitung dari `predict_proba` tanpa ambang |
| Status | `status_1_se_informatif`: \|M̄<sub>v</sub> − M̄<sub>acuan</sub>\| > SE<sub>M,acuan</sub>, dihitung per metrik, hanya informatif |
| Alasan | Kelas tidak seimbang (baseline 62–80%), sehingga akurasi dapat menutupi berkurangnya siswa berisiko yang dikenali |

## Definisi

| Metrik | Rumus | Makna |
| :--- | :--- | :--- |
| Presisi | TP / (TP + FP) | Dari siswa yang ditandai berisiko, berapa yang memang berisiko |
| Recall | TP / (TP + FN) | Dari siswa berisiko, berapa yang dikenali. Paling dekat dengan tujuan peringatan dini |
| F1 | 2·P·R / (P + R) | Rata-rata harmonik presisi dan recall |
| AUC-ROC | Luas di bawah kurva TPR–FPR | Peluang siswa berisiko diberi probabilitas lebih tinggi daripada siswa tidak berisiko. 0,5 = acak, 1 = sempurna |

## Catatan Teknis

- Implementasi: `precision_score`, `recall_score`, `f1_score` (`pos_label=1`, `zero_division=0`), dan `roc_auc_score` dari scikit-learn, di `src/models/trainer.py`.
- Metrik dihitung per fold, bukan dari gabungan prediksi out-of-fold, agar rata-rata dan SE-nya sebanding dengan akurasi.
- `zero_division=0` hanya berpengaruh bila sebuah fold tidak memuat satu pun prediksi berisiko.
- Integritas prosedur: metrik ini ditambahkan setelah hasil akurasi pertama terlihat, sehingga di naskah disebut analisis pelengkap, bukan kriteria yang ditetapkan sejak awal (Subbab 3.6).

## Keluaran

| File | Isi |
| :--- | :--- |
| `outputs/tables/tabel_metrik_pelengkap.csv` | Format panjang, satu baris per modul × konfigurasi × metrik: `rata_rata`, `standard_error`, `se_acuan`, `selisih_delta`, `rasio_delta_se`, `status_1_se_informatif` |
| `outputs/tables/tabel_perbandingan_sensitivitas_metrik.csv` | Per metrik (Akurasi = hasil utama): jumlah modul sensitif, jumlah modul diuji, dan daftar modul sensitif per konsep |
| `outputs/figures/grafik_semua_metrik_per_konfigurasi.png` | Lima panel nilai metrik per konfigurasi ±1 SE, dengan baseline pada panel akurasi |
| `outputs/figures/grafik_selisih_semua_metrik.png` | Heatmap selisih terhadap K0 (poin persen). Biru = lebih baik, merah = lebih buruk, `*` = melampaui 1 SE |
| `outputs/figures/grafik_sensitivitas_per_metrik.png` | Jumlah modul sensitif per konsep menurut tiap metrik |

Rujukan: Sokolova and Lapalme (2009), Fawcett (2006).
