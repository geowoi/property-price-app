import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ===============================
# JUDUL
# ===============================
st.set_page_config(page_title="AI Prediksi Harga Properti", layout="wide")
st.title("🏠 AI Prediksi Harga Properti")
st.markdown("Aplikasi kecerdasan buatan untuk memprediksi harga properti menggunakan **Machine Learning (Random Forest)**")

# ===============================
# LOAD DATASET
# ===============================
@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df

df = load_data()

# ===============================
# EDA
# ===============================
st.subheader("📊 Statistik Dataset")
st.write(df.describe())

st.subheader("📈 Grafik Hubungan Median Income vs Harga Rumah")
fig, ax = plt.subplots()
ax.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.3)
ax.set_xlabel("Median Income")
ax.set_ylabel("Harga Rumah")
st.pyplot(fig)

# ===============================
# TRAIN MODEL (AI)
# ===============================
features = df.drop("MedHouseVal", axis=1)
target = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)

st.success(f"✅ Model AI berhasil dilatih | MAE: {mae:.2f}")

# ===============================
# INPUT USER
# ===============================
st.subheader("🧠 Prediksi Harga Properti (AI)")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("Median Income", 0.5, 15.0, 3.0)
    HouseAge = st.slider("Umur Bangunan", 1.0, 50.0, 20.0)
    AveRooms = st.slider("Rata-rata Jumlah Kamar", 1.0, 10.0, 5.0)
    AveBedrms = st.slider("Rata-rata Kamar Tidur", 1.0, 5.0, 1.0)

with col2:
    Population = st.slider("Populasi Area", 100.0, 5000.0, 1000.0)
    AveOccup = st.slider("Rata-rata Penghuni", 1.0, 6.0, 3.0)
    Latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
    Longitude = st.slider("Longitude", -125.0, -114.0, -118.0)

input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                        Population, AveOccup, Latitude, Longitude]])

# ===============================
# PREDIKSI
# ===============================
if st.button("🔮 Prediksi Harga"):
    prediction = model.predict(input_data)[0]
    st.metric(
        label="💰 Perkiraan Harga Properti",
        value=f"${prediction * 100_000:,.0f}"
    )
    st.info("Prediksi ini dihasilkan oleh **AI (Random Forest Regressor)** berdasarkan pola data historis.")

# ===============================
# PENJELASAN AI
# ===============================
st.subheader("🤖 Kenapa ini disebut AI?")
st.markdown("""
- Menggunakan **Machine Learning**
- Model belajar dari **20.000+ data properti**
- Bukan rumus manual
- Model menemukan pola sendiri dari data
""")
