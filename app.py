import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# =====================================
# 1. GENERATE DATA PROPERTI (SIMULASI)
# =====================================
data = []

for _ in range(300):
    area = random.randint(3000, 12000)        # luas bangunan
    bedrooms = random.randint(1, 6)           # kamar tidur
    bathrooms = random.randint(1, 4)          # kamar mandi
    stories = random.randint(1, 4)             # lantai

    # Rumus harga (simulasi AI)
    price = (
        area * 20 +
        bedrooms * 5000 +
        bathrooms * 8000 +
        stories * 7000 +
        random.randint(-10000, 10000)
    )

    data.append([area, bedrooms, bathrooms, stories, price])

df = pd.DataFrame(
    data,
    columns=["area", "bedrooms", "bathrooms", "stories", "price"]
)

print("✅ Data properti berhasil dibuat")

# =====================================
# 2. TRAINING MODEL AI
# =====================================
X = df[["area", "bedrooms", "bathrooms", "stories"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# =====================================
# 3. SIMPAN MODEL & SCALER
# =====================================
pickle.dump(model, open("model_property.pkl", "wb"))
pickle.dump(scaler, open("scaler_property.pkl", "wb"))

print("✅ Model AI berhasil dibuat")
print("📁 File dihasilkan:")
print("- model_property.pkl")
print("- scaler_property.pkl")

