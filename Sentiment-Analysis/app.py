import streamlit as st
import joblib
import pandas as pd
import re

# Memuat model dan vectorizer yang sudah disimpan
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Fungsi untuk membersihkan teks ulasan baru
def preprocess_text(text):
    # Ubah teks menjadi huruf kecil
    text = text.lower()
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Hapus karakter non-alfanumerik dan spasi berlebih
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Judul aplikasi
st.title('Shopee Sentiment Analysis Chatbot')
st.write('Masukkan ulasan Anda dan chatbot akan memprediksi sentimennya.')

# Input dari pengguna
user_input = st.text_area("Masukkan ulasan Anda di sini:", "")

if st.button('Prediksi Sentimen'):
    if user_input:
        # Pra-proses input pengguna
        cleaned_input = preprocess_text(user_input)
        
        # Mengubah teks menjadi fitur numerik (vektor TF-IDF)
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Membuat prediksi
        prediction = model.predict(vectorized_input)[0]
        
        # Menampilkan hasil prediksi
        st.write('---')
        st.subheader('Hasil Prediksi:')
        if prediction == 'Positif':
            st.success(f'Sentimen: {prediction} 😄')
        elif prediction == 'Negatif':
            st.error(f'Sentimen: {prediction} 😔')
        else:
            st.info(f'Sentimen: {prediction} 🙂')
    else:
        st.warning('Mohon masukkan ulasan terlebih dahulu!')