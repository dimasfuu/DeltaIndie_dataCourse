import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np

#load models

muat_model_rf = joblib.load("model_random_forest.joblib")
muat_model_et = joblib.load("model_extra_trees.joblib")
muat_model_gb = joblib.load("model_gradient_boosting.joblib")

df = pd.read_csv("df_clean2.csv")

st.header("Final Project 1 Delta Indie Course !")
st.subheader("Analisis data tentang Obesitas")

st.title("Dikumpulkan sebagai Final Project 1 untuk Kursus Delta Indie Course")
st.write("Nama :  Dimas Furqon Prawimastoro")


st.dataframe(df)

# streamlit UI
st.title("Klasifikasi Obesitas")
st.write("Masukkan data pasien untuk mengetahui apakah pasien menderita Obesitas tipe tertentu.")

#pilih model
model_option = st.selectbox("Pilih Model",("model_random_forest.joblib","model_extra_trees.joblib","model_gradient_boosting.joblib")) # pilihan model

import streamlit as st

col1,col2 = st.columns(2)

with col1:
    Age = st.number_input("Usia", min_value=0, step=1) 
    Height = st.number_input("Tinggi Badan (masukkan dalam satuan meter)", min_value=0.0, step=0.1)  
    Weight = st.number_input("Berat Badan", min_value=0.0, step=0.1)  
    FCVC = st.slider('Frekuensi konsumsi buah-buahan (skala 1 sampai 3):',min_value=1, max_value= 3, step=1)
    NCP = st.number_input("Jumlah makan utama per hari", min_value=0, step=1) 
    CH2O = st.slider("Konsumsi minum per hari (skala 1 sampai 3)", min_value=1, max_value= 3, step=1) 
    FAF = st.slider("Frekuensi Aktivitas fisik (skala 0 sampai 3)", min_value=0, max_value= 3, step=1)  
    TUE = st.slider("Penggunaan Teknologi (skala 0 sampai 3)", min_value=0, max_value= 3, step=1)  

with col2: 
    En_binary_options = {
        1 : "Ya",
        0 : "Tidak"
    }   

    En_Fam = st.selectbox("Apakah memiliki riwayat obesitas pada keluarga : ", options=list(En_binary_options.keys()),format_func=lambda x:En_binary_options[x])  
    selected_value = En_binary_options[En_Fam]

    En_FAVC = st.selectbox("apakah mengkonsumsi makanan tinggi kalori : ", options=list(En_binary_options.keys()),format_func=lambda x:En_binary_options[x]) 
    selected_value = En_binary_options[En_FAVC] 

    En_CAEC = st.number_input("Jumlah konsumsi cemilan", min_value=0, max_value=3, step=1)  

    En_SMOKE = st.selectbox("Apakah merokok : ", options=list(En_binary_options.keys()),format_func=lambda x:En_binary_options[x]) 
    selected_value = En_binary_options[En_SMOKE] 

    En_SCC = st.selectbox("Apakah melakukan kontrol kalori harian : ", options=list(En_binary_options.keys()),format_func=lambda x:En_binary_options[x]) 
    selected_value = En_binary_options[En_SCC]

    En_CALC_options={
        0 : "Selalu",
        1 : "Sering",
        2 : "Jarang",
        3 : "Tidak Pernah",
    }   
    En_CALC = st.selectbox("Pola konsumsi Alkohol : ", options=list(En_CALC_options.keys()),format_func=lambda x:En_CALC_options[x]) 
    selected_value = En_CALC_options[En_CALC]
    
    En_MTRANS_options={
        0 : "Mobil",
        1 : "Sepeda",
        2 : "Sepeda Motor",
        3 : "Transportasi Umum",
        4 : "Jalan Kaki"
    }
    En_MTRANS = st.selectbox("Moda Transportasi yang sering digunakan untuk bepergian ", options=list(En_MTRANS_options.keys()),format_func=lambda x:En_MTRANS_options[x]) 
    selected_value = En_MTRANS_options[En_MTRANS]  

    En_Gender_options={
        0 : "Perempuan",
        1 : "Laki-Laki"
    }
    En_Gender = st.selectbox("Jenis Kelamin ", options=list(En_Gender_options.keys()),format_func=lambda x:En_Gender_options[x]) 
    selected_value = En_Gender_options[En_Gender]  



    # mode radio button 
    # st.write("jenis Kelamin Pasien ")
    # gender = st.radio(
    #     "Jenis Kelamin:",
    #     options=["Laki-laki", "Perempuan"]
    # )
    # gender_value = 1 if gender == "Laki-laki" else 0
    # st.write(f"Jenis Kelamin yang dipilih: {gender} (Value: {gender_value})")

    # mode button
    # col1, col2 = st.columns(2)
    # with col1:
    # if st.button("Laki-laki"):
    #     gender = 1
    # with col2:
    # if st.button("Perempuan"):
    #     gender = 0
    # if 'gender' in locals():
    # st.write(f"Jenis Kelamin yang dipilih: {'Laki-laki' if gender == 1 else 'Perempuan'}")


if st.button("Prediksi"):  # Jika tombol ditekan
    if model_option == "model_random_forest.joblib":
        model = muat_model_rf
    elif model_option == "model_extra_trees.joblib":
        model = muat_model_et
    else:
        model = muat_model_gb

    input_data = np.array([[Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE,En_Fam, En_FAVC, En_CAEC, En_SMOKE, En_SCC,En_CALC, En_MTRANS, En_Gender]])#membentuk array input

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