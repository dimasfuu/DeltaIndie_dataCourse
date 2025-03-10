import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np

#load models
with open("model_rfc.pkl","rb") as file: 
    model_pkl = pickle.load(file)

model_joblib = joblib.load("model_diabet_rfc.joblib")

df = pd.read_csv("Healthcare-Diabetes.csv")

st.header("Final Project 1 Delta Indie Course")
st.subheader("Analisis data tentang Diabetes Course")

st.title("Perkenalan")
st.write("Nama saya Dimas Furqon Prawimastoro")
st.write("saya biasa dipanggil dimas")


st.text_input("Masukan Nama")
st.number_input("Masukan Umur")

st.dataframe(df)

# streamlit UI
st.title("Klasifikasi Diabetes")
st.write("Masukkan data pasien untuk mengetahui apakah pasien menderita diabetes atau tidak.")

#pilih model
model_option = st.selectbox("Pilih Model",("model_rfc_pkl","model_diabet_rfc.joblib")) # pilihan model

#input pengguna 
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0.0, step=0.1)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=0.1)
insulin = st.number_input("insulin", min_value=0.0, step=0.1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.1)
age = st.number_input("Age", min_value=0, step=1)

#prediksi
if st.button("Prediksi"): #jika tombol ditekan
    model = model_pkl if model_option == "model_rfc_pkl" else model_joblib # memilih model yang digunakan

    input_data = np.array([[pregnancies, glucose, blood_pressure,skin_thickness, insulin, bmi, diabetes_pedigree, age]])#membentuk array input

    prediction = model.predict(input_data) #melakukan prediksi dengan model

    if prediction[0] == 1:
        st.error("Pasien terdeteksi mengidap diabetes.")
    else:
        st.success("Pasien tidak mengidap diabetes.")