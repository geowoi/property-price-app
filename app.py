import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Harga Properti", layout="centered")

st.title("🏠 Aplikasi Prediksi Harga Properti")
st.write("Aplikasi AI sederhana untuk memprediksi harga rumah berdasarkan data sintetis.")

# =============================
# BUAT DATASET SENDIRI (AMAN)
# =============================
@st.cache_data
def load_data():
    np.random.seed(42)
    data = {
        "Luas_Bangunan": np.random.randint(50, 300, 100),
        "Jumlah_Kamar": np.random.randint(1, 6, 100),
        "Harga": np.random.randint(300, 3000, 100) * 1_000_000
    }
    return pd.DataFrame(data)

df = load_data()

# =============================
# TAMPILKAN DATA
# =============================
st.subheader("📊 Statistik Data Properti")
st.dataframe(df.head())

# =============================
# GRAFIK
# =============================
st.subheader("📈 Grafik Harga vs Luas Bangunan")
fig, ax = plt.subplots()
ax.scatter(df["Luas_Bangunan"], df["Harga"])
ax.set_xlabel("Luas Bangunan (m²)")
ax.set_ylabel("Harga (Rp)")
st.pyplot(fig)

# =============================
# TRAIN MODEL (LANGSUNG)
# =============================
X = df[["Luas_Bangunan", "Jumlah_Kamar"]]
y = df["Harga"]

model = LinearRegression()
model.fit(X, y)

# =============================
# INPUT USER
# =============================
st.subheader("🔮 Prediksi Harga Rumah")

luas = st.number_input("Luas Bangunan (m²)", 20, 500, 100)
kamar = st.number_input("Jumlah Kamar", 1, 10, 3)

if st.button("Prediksi Harga"):
    prediksi = model.predict([[luas, kamar]])
    st.success(f"💰 Estimasi Harga Rumah: Rp {int(prediksi[0]):,}")
