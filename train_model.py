import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("housing.csv")

# =========================
# FITUR & TARGET (PASTI ADA)
# =========================
features = ["area", "bedrooms", "bathrooms", "stories"]
target = "price"

X = df[features]
y = df[target]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# =========================
# TRAIN MODEL
# =========================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# =========================
# SAVE MODEL
# =========================
pickle.dump(model, open("model_property.pkl", "wb"))
pickle.dump(scaler, open("scaler_property.pkl", "wb"))

print("✅ Training selesai. Model & scaler berhasil disimpan.")
