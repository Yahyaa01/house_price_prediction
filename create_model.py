import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Make sure we are in the correct folder
print("Current working directory:", os.getcwd())

# Load dataset
df = pd.read_csv("house_prices_dataset.csv")

# Basic preprocessing
df.fillna(df.median(), inplace=True)
df.drop_duplicates(inplace=True)

# Features and target
X = df[['square_feet', 'num_rooms', 'age', 'distance_to_city(km)']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model as a proper pickle
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ house_price_model.pkl created successfully!")
