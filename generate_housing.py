import pandas as pd
import random

data = []

for _ in range(200):
    area = random.randint(3000, 12000)
    bedrooms = random.randint(1, 6)
    bathrooms = random.randint(1, 4)
    stories = random.randint(1, 4)
    price = area * 20 + bedrooms * 5000 + bathrooms * 8000 + stories * 7000

    data.append([area, bedrooms, bathrooms, stories, price])

df = pd.DataFrame(
    data,
    columns=["area", "bedrooms", "bathrooms", "stories", "price"]
)

df.to_csv("housing.csv", index=False)
print("✅ housing.csv berhasil dibuat")
