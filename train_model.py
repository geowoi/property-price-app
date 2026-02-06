import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# Upload dataset AmesHousing.csv dulu
df = pd.read_csv("AmesHousing.csv")

features = ["GrLivArea", "OverallQual", "YearBuilt", "TotalBsmtSF", "GarageCars"]
target = "SalePrice"

X = df[features]
y = df[target]

# Handle missing values
X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model & scaler
pickle.dump(model, open("model_property.pkl", "wb"))
pickle.dump(scaler, open("scaler_property.pkl", "wb"))

print("✅ Model & scaler berhasil dibuat")
