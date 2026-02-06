import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Set page config
st.set_page_config(page_title="Prediksi Harga Properti", layout="wide")

st.title("🏠 Aplikasi Prediksi Harga Properti")
st.write("Prediksi harga rumah berdasarkan fitur-fitur utama")

# ------------------------
# Load Model dan Scaler
# ------------------------
model = pickle.load(open("model_property.pkl", "rb"))
scaler = pickle.load(open("scaler_property.pkl", "rb"))

# ------------------------
# Input dari User
# ------------------------
st.header("Input Data Properti")

gr_liv_area = st.number_input("Luas Bangunan (sq ft)", min_value=200, max_value=10000, value=1500)
overall_qual = st.slider("Kualitas Bangunan (1–10)", 1, 10, 5)
year_built = st.slider("Tahun Dibangun", 1900, 2022, 2005)
total_bsmt_sf = st.number_input("Luas Basement (sq ft)", min_value=0, max_value=3000, value=800)
garage_cars = st.slider("Kapasitas Mobil Garage", 0, 4, 2)

features = np.array([[gr_liv_area, overall_qual, year_built, total_bsmt_sf, garage_cars]])
features_scaled = scaler.transform(features)

# ------------------------
# Prediksi
# ------------------------
if st.button("Prediksi Harga"):
    price_pred = model.predict(features_scaled)
    st.success(f"📊 Prediksi Harga Properti: ${price_pred[0]:,.2f}")

# ------------------------
# Statistik & Grafik
# ------------------------
st.header("Visualisasi Harga Properti")
data = pd.read_csv("AmesHousing.csv")
st.write(data.head())

st.subheader("Distribusi Harga Rumah")
st.bar_chart(data["SalePrice"])
