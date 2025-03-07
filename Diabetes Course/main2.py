import streamlit as st  # Mengimpor pustaka Streamlit untuk membuat aplikasi web
import pickle  # Untuk memuat model yang disimpan dalam format pickle
import joblib  # Alternatif pustaka untuk memuat model machine learning
import numpy as np  # Digunakan untuk manipulasi array numerik

# Load models
with open("model_rfc.pkl", "rb") as file:
    model_pkl = pickle.load(file)  # Memuat model dari file pickle

model_joblib = joblib.load("model_diabet_rfc.joblib")  # Memuat model dari file joblib

# Streamlit UI
st.title("Klasifikasi Diabetes")  # Menampilkan judul aplikasi
st.write("Masukkan data pasien untuk mengetahui apakah pasien menderita diabetes atau tidak.")  # Deskripsi aplikasi

# Pilih model
model_option = st.selectbox("Pilih Model", ("model_rfc.pkl", "model_diabet_rfc.joblib"))  # Pilihan model untuk digunakan

# Input pengguna
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)  # Input jumlah kehamilan
glucose = st.number_input("Glucose", min_value=0.0, step=0.1)  # Input kadar glukosa dalam darah
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1)  # Input tekanan darah
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=0.1)  # Input ketebalan kulit
insulin = st.number_input("Insulin", min_value=0.0, step=0.1)  # Input kadar insulin
bmi = st.number_input("BMI", min_value=0.0, step=0.1)  # Input indeks massa tubuh (BMI)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)  # Input faktor keturunan diabetes
age = st.number_input("Age", min_value=0, step=1)  # Input usia pasien

# Prediksi
if st.button("Prediksi"):  # Jika tombol ditekan
    model = model_pkl if model_option == "model_rfc.pkl" else model_joblib  # Memilih model yang digunakan
    
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])  # Membentuk array input
    prediction = model.predict(input_data)  # Melakukan prediksi dengan model
    
    if prediction[0] == 1:
        st.error("Pasien terdeteksi mengidap diabetes.")  # Menampilkan pesan jika hasil prediksi = 1 (diabetes)
    else:
        st.success("Pasien tidak mengidap diabetes.")  # Menampilkan pesan jika hasil prediksi = 0 (tidak diabetes)