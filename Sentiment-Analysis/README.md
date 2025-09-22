# 📊 Shopee Sentiment Analysis Chatbot

Proyek ini bertujuan untuk menganalisis sentimen ulasan pengguna di **Shopee** menggunakan **Natural Language Processing (NLP)** dan **Machine Learning**.  
Model yang digunakan adalah **Naive Bayes Classifier** dengan fitur **TF-IDF**.  
Hasil akhirnya berupa aplikasi interaktif berbasis **Streamlit** yang dapat memprediksi sentimen ulasan (Positif, Negatif, atau Netral).

---

## 🚀 Fitur
- **Preprocessing teks otomatis** (lowercasing, hapus URL, karakter non-alfanumerik, dll).
- **Model Machine Learning (Naive Bayes)** untuk klasifikasi sentimen.
- **Evaluasi model** menggunakan metrik seperti akurasi, presisi, recall, dan confusion matrix.
- **Streamlit App** sebagai chatbot interaktif untuk prediksi sentimen.

---

## 📂 Struktur Project
├── sentiment_analysis_ecommerce.py # Script training model & penyimpanan model
├── app.py # Aplikasi Streamlit untuk prediksi sentimen
├── sentiment_model.pkl # Model Machine Learning (hasil training)
├── tfidf_vectorizer.pkl # Vectorizer TF-IDF (hasil training)


---

## 🛠️ Instalasi

1. Clone repository:
   ```bash
   git clone https://github.com/username/shopee-sentiment-analysis.git
   cd shopee-sentiment-analysis
   ```
2. Buat virtual environment (opsional tapi disarankan):
```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
```
4. Install dependencies:
```bash
   pip install -r requirements.txt
```
## 📊 Training Model

Untuk melatih model dan menyimpan hasilnya:
```
python sentiment_analysis_ecommerce.py
```
Hasil training akan menghasilkan:
- sentiment_model.pkl
- tfidf_vectorizer.pkl

## 💻 Menjalankan Aplikasi

Jalankan aplikasi Streamlit:
```bash
streamlit run app.py
```

Kemudian buka browser
```bash
http://localhost:8501
```

## 📝 Contoh Penggunaan

Masukkan ulasan di aplikasi, misalnya:
```bash
"Produk bagus banget, pengiriman cepat!"
```
output:
```bash
Sentimen: Positif 😄
```

## 📦 Dependencies

- Python 3.8+
- pandas
- scikit-learn
- joblib
- streamlit
