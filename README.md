# Sistem Analisis Sentimen Ulasan Produk UMKM Berbasis NLP

Proyek ini merupakan **proyek mandiri** yang dikembangkan sebagai hasil penerapan pengetahuan dan keterampilan selama mengikuti **Program Studi Independen Asah by Dicoding (association with Accenture)** pada learning path *Machine Learning dan Artificial Intelligence*.

## 📌 Deskripsi Proyek
Sistem ini bertujuan untuk menganalisis sentimen ulasan produk **UMKM** menggunakan pendekatan **Natural Language Processing (NLP)**. Analisis sentimen dilakukan dengan membandingkan performa dua model klasifikasi, yaitu **Logistic Regression** dan **Long Short-Term Memory (LSTM)**, untuk menentukan model yang paling optimal.

Dataset yang digunakan berasal dari **ulasan produk UMKM pada platform Tokopedia**, yang telah melalui tahapan *text preprocessing* dan ekstraksi fitur menggunakan metode **TF-IDF**.

## 🛠️ Teknologi yang Digunakan
- Python  
- Natural Language Processing (NLP)  
- TF-IDF  
- Logistic Regression  
- Long Short-Term Memory (LSTM)  
- Streamlit  
- Git & GitHub  

## ⚙️ Fitur Sistem
- Klasifikasi sentimen ulasan produk (positif/negatif)
- Perbandingan performa model Logistic Regression dan LSTM
- Antarmuka **GUI berbasis Streamlit**
- Input ulasan secara langsung dan hasil klasifikasi secara *real-time*

## 🖥️ Tampilan Aplikasi
Aplikasi dikembangkan menggunakan **Streamlit** sehingga mudah digunakan oleh pengguna umum maupun pelaku UMKM.

🔗 **Link Aplikasi:**  
*(isi link Streamlit kamu di sini)*

## 📂 Struktur Proyek
Berikut adalah struktur direktori dari proyek **mini_project_mbkm**:
mini_project_mbkm/
├── dataset/
│ ├── raw/
│ │ └── tokopedia-product-reviews-2019.csv
│ └── processed/
│ └── data_preprocessed.csv
│
├── models/
│ ├── logistic_regression/
│ │ ├── best_model_logistic_regression.pkl
│ │ └── tfidf_vectorizer.pkl
│ │
│ └── lstm/
│ ├── augmented_minoritas.csv
│ ├── best_model_lstm.h5
│ ├── best_model_lstm_tuned.h5
│ └── tokenizer_sentiment.pkl
│
├── notebooks/
│ ├── 01_preprocessing_eda.ipynb
│ ├── 02_ml_logistic_regression.ipynb
│ └── 03_dl_lstm.ipynb
│
├── src/
│ ├── app.py
│ └── requirements.txt
│
└── README.md

## 📌 Keterangan Singkat
- **dataset/**: Berisi data mentah dan data hasil preprocessing ulasan produk UMKM Tokopedia  
- **models/**: Menyimpan model terlatih Logistic Regression dan LSTM beserta pendukungnya  
- **notebooks/**: Notebook eksplorasi data, preprocessing, serta eksperimen model  
- **src/**: Source code aplikasi GUI berbasis Streamlit


## 📊 Hasil dan Evaluasi
Berdasarkan hasil evaluasi menggunakan metrik **akurasi dan F1-score**, model **Logistic Regression dengan representasi TF-IDF** menunjukkan performa yang lebih stabil dan efisien dibandingkan model LSTM, sehingga dipilih sebagai model utama dalam sistem.

## 📝 Dokumentasi
Seluruh proses pengembangan sistem, mulai dari pengolahan data, pemodelan, evaluasi, hingga implementasi aplikasi, telah **didokumentasikan dalam laporan akhir** sebagai bagian dari penyelesaian program Studi Independen.

## 👤 Penulis
**Devira Mutiara Widya Nugraha**  
Mahasiswa Teknik Informatika  

---
