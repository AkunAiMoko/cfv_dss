# CFV-DSS

**Causal Financial Vulnerability Decision Support** merupakan prototipe riset untuk prediksi kesulitan pembayaran, penjelasan faktor prediktif, analisis kausal berbatas domain, dan simulasi prioritas intervensi melalui dashboard Streamlit.

## Dokumentasi

- [Panduan tahapan teknis riset berbahasa Indonesia](cfv_dss/TAHAPAN_TEKNIS_RISET.md), termasuk tiga flowchart PNG.
- [Panduan teknis versi Word](cfv_dss/TAHAPAN_TEKNIS_RISET.docx).
- [Naskah artikel](cfv_dss/manuscript/article.md).
- [Naskah artikel versi Word](cfv_dss/manuscript/article.docx).
- [Catatan verifikasi 50 referensi dan DOI](cfv_dss/manuscript/reference_verification.csv).
- [Dokumentasi paket dan artefak](cfv_dss/README.md).

## Isi repository

Repository memuat kode Python, dependensi, dokumentasi, gambar PNG, model tersimpan, dan ringkasan hasil analisis. Dataset mentah, tabel turunan per pemohon, virtual environment, dan cache tidak disertakan.

```text
cfv_dss/
├── app.py                         # Dashboard Streamlit
├── train.py                       # Pipeline utama
├── sensitivity.py                 # Evaluasi sensitivitas
├── requirements.txt
├── artifacts/
│   ├── models/
│   ├── metrics/
│   ├── causal/
│   ├── figures/
│   └── reports/
├── manuscript/
├── TAHAPAN_TEKNIS_RISET.md
└── TAHAPAN_TEKNIS_RISET.docx
```

## Menjalankan dashboard

Jalankan dari akar repository. Artefak model dan ringkasan yang diperlukan dashboard sudah disertakan; dataset mentah tidak diperlukan untuk membuka dashboard dan memasukkan konstruk secara manual.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\cfv_dss\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run .\cfv_dss\app.py
```

Gunakan alamat lokal yang dicetak Streamlit. Versi dependensi mengikuti `requirements.txt`; versi Python pada eksperimen tersimpan adalah 3.14.0.

## Mengulang penelitian dengan dataset lokal

Sediakan dataset Home Credit Default Risk dan Give Me Some Credit sesuai ketentuan sumber masing-masing. Tempatkan berkas berikut pada folder **`Data Set Home Credit` di akar repository**, sejajar dengan folder paket `cfv_dss`:

```text
Data Set Home Credit/
├── application_train.csv
├── bureau.csv
├── previous_application.csv
├── installments_payments.csv
├── credit_card_balance.csv
└── cs-training.csv
```

Kemudian jalankan:

```powershell
.\.venv\Scripts\python.exe -m cfv_dss.train
.\.venv\Scripts\python.exe -m cfv_dss.sensitivity
.\.venv\Scripts\python.exe -m streamlit run .\cfv_dss\app.py
```

Pipeline menghasilkan kembali `cfv_dss/artifacts/data/` secara lokal. Folder dataset dan artefak data tersebut dikecualikan melalui `.gitignore`.

## Status hasil dan batas penggunaan

Model prediktif tersimpan memilih XGBoost dengan ROC-AUC 0,6786. Estimasi kausal dan nilai intervensi bergantung pada asumsi observasional; utility simulasi bukan bukti dampak intervensi aktual.

CSV simulasi kebijakan yang tersedia saat publikasi berisi nilai nol, berbeda dari nilai yang tercantum dalam naskah. Panduan teknis menjelaskan ketidaksesuaian tersebut serta bahwa satu uji sintetis dapat menimpa CSV hasil kebijakan. Periksa konsistensi tabel, grafik, dan naskah sebelum menggunakan hasil untuk pelaporan; jangan menjalankan uji tersebut setelah menghasilkan artefak final tanpa pemisahan atau pemulihan keluarannya.

CFV-DSS merupakan prototipe dukungan keputusan dan memerlukan penilaian manusia serta validasi lokal untuk penggunaan yang berdampak pada pemohon.
