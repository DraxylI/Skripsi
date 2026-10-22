# TODO
## Akan Dilakukan
- [ ] git init di D:\skripsi2, commit awal, tambah remote, lalu push (menunggu izin)
- [ ] Naskah 3.8.5: tulis operasionalisasi "RF tidak informatif" = akurasi acuan tidak melampaui baseline > 1 SE (lihat HASIL_RUN_2026-10-22.md §9)
## Sedang Dilakukan
- [ ] Bereskan Debt: RF n_jobs=1, test suite unittest, penanda RF tidak informatif di grafik, teks konsol kohort, banner md usang
## Sudah Dilakukan
- [x] 2026-10-22: Audit pipeline terhadap draf revisi 22-10-2026; tambah kolom acuan_melampaui_baseline dan informatif_rf (Subbab 3.8.5), ganti nama kolom atrisi agar tidak bertentangan dengan penomoran naskah, perbarui tercantum_di_naskah register (semua "Ya")
- [x] 2026-10-22: Ringkas docstring/komentar seluruh src dan run_pipeline.py (diverifikasi AST identik untuk file docs-only)
- [x] 2026-10-22: Tulis ulang README.md, ringkas DOKUMENTASI_METRIK_PELENGKAP.md
- [x] 2026-10-22: Run end-to-end (101 detik); hasil utama identik dengan run 27-09; didokumentasikan di HASIL_RUN_2026-10-22.md
- [x] 2026-10-22: Buat D:\skripsi2\.gitignore berbasis whitelist (hanya pipeline/ dan outputs-nya; backup, dataset, dapus, md naskah, .venv, .claude, graphify-out diabaikan)
- [x] 2026-10-13: Inventarisasi jenis fitur dan varian definisi (konsep A, B, C; K0–K5; C2_due sensitivitas) dari src/features/extractor.py, src/features/builder.py, dan config/config.yaml
## Debt
- D:\skripsi2\.gitignore → dataset/ tidak ikut repo karena studentVle.csv 433 MB melebihi batas 100 MB GitHub; instruksi unduh sudah ada di README §1
- src/models/trainer.py → _make_model: RandomForestClassifier(n_jobs=-1) membuat AUC-ROC RF bergeser ±0,0001 antar-run; status tidak terpengaruh. Pakai n_jobs=1 bila butuh reproduksi bit-identik
- pipeline/ → belum ada test suite; logika Evaluator/register diverifikasi dengan smoke assert ad hoc. Tambah tests/ dengan pytest bila diizinkan
- src/visualization/plots.py → plot_lr_vs_rf_delta: modul RF tidak informatif (AAA, GGG) belum ditandai di grafik; baca kolom informatif_rf di tabel
- run_pipeline.py → run_preprocess: teks konsol "Aturan 1 & Aturan 2" masih memakai penomoran lama (kosmetik)
- D:\skripsi2\PENYESUAIAN_NASKAH.md dan RINGKASAN_HASIL_RUN.md (di luar pipeline) sudah usang terhadap draf revisi; HASIL_RUN_2026-10-22.md menggantikan ringkasan hasil
