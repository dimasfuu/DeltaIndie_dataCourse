import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np

#load models


muat_model_rf = joblib.load("model_random_forest.joblib")
muat_model_gb = joblib.load("model_gradient_boosting.joblib")

df = pd.read_csv("df_clean2.csv")

st.header("Final Project 1 Delta Indie Course")
st.subheader("Analisis data tentang Obesitas")

st.title("Dikumpulkan sebagai Final Project 1 untuk Kursus Delta Indie Course")
st.write("Nama :  Dimas Furqon Prawimastoro")


st.dataframe(df)

# streamlit UI
st.title("Klasifikasi Obesitas")
st.write("Masukkan data pasien untuk mengetahui apakah pasien menderita Obesitas tipe tertentu.")

#pilih model
model_option = st.selectbox("Pilih Model",("model_random_forest.joblib","model_gradient_boosting.joblib")) # pilihan model

#input pengguna 
Age = st.number_input("Masukkan usia:", min_value=0, max_value=100, value=25, step=1)
Height = st.number_input("Tinggi Badan", min_value=0.0, step=0.1)
Weight = st.number_input("Berat Badan", min_value=0.0, step=0.1)
FCVC = st.number_input("Konsumsi buah-buahan", min_value=0.0, max_value=3.0, step=0.1)
NCP = st.number_input("Jumlah makan utama per hari", min_value=0.0, step=0.1)
CH2O = st.number_input("Konsumsi minum per hari", min_value=0.0, step=0.1)
FAF = st.number_input("Frekuensi Aktifitas fisik", min_value=0.0, step=0.1)
TUE = st.number_input("penggunaan Teknologi", min_value=0.0, step=0.1)
En_Fam= st.number_input("Riwayat Obesitas", min_value=0,max_value=1, step=1)

En_FAVC= st.number_input("Makan tinggi kalori", min_value=0,max_value=1, step=1)
En_CAEC = st.number_input("jumlah konsumsi cemilan", min_value=0,max_value=3, step=1)
En_SMOKE = st.number_input("apakah merokok", min_value=0,max_value=1, step=1)
En_SCC = st.number_input("Kontrol harian", min_value=0,max_value=1, step=1)
En_CALC = st.number_input("pola konsumsi Alkohol", min_value=0,max_value=3, step=1)
En_MTRANS = st.number_input("pola Transportasi publik", min_value=0,max_value=4,step=1)
En_Gender = st.number_input("jenis kelamin", min_value=0,max_value=1, step=1)



#prediksi
if st.button("Prediksi"): #jika tombol ditekan
    model = muat_model_rf if model_option == "model_random_forest.joblib" else muat_model_gb # memilih model yang digunakan

    input_data = np.array([['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE','En_Fam', 'En_FAVC', 'En_CAEC', 'En_SMOKE', 'En_SCC','En_CALC', 'En_MTRANS', 'En_Gender']])#membentuk array input

    prediction = model.predict(input_data) #melakukan prediksi dengan model

    if prediction[0] == 0:
        st.error("Pasien terdeteksi mengidap diabetes.")
    elif prediction[0] ==1:
        st.error("Pasien Berat Badan Normal")
    elif prediction[0] ==2:
        st.error("Pasien Berat Badan Obesitas Tipe I")
    elif prediction[0] ==3:
        st.error("Pasien Berat Badan Obesitas Tipe II")
    elif prediction[0] ==4:
       st.error("Pasien Berat Badan Obesitas Tipe III")
    elif prediction[0] ==5:
        st.error("Pasien Berat Badan Lebih Tipe I")
    else : 
        st.success("Pasien Berat Badan Lebih Tipe II")