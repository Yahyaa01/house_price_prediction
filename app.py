from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For server
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "house_price_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv("house_prices_dataset.csv")

# Ensure plots folder exists
plots_dir = os.path.join("static", "plots")
os.makedirs(plots_dir, exist_ok=True)

# Generate and save plots
def generate_plots():
    features = ['square_feet', 'num_rooms', 'age', 'distance_to_city(km)', 'price']
    
    # 1. Feature Distributions
    for col in features:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, color='lightblue')
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{col}.png"))
        plt.close()
    
    # 2. Actual vs Predicted
    X = df[['square_feet', 'num_rooms', 'age', 'distance_to_city(km)']]
    y = df['price']
    y_pred = model.predict(X)
    
    plt.figure(figsize=(6,6))
    plt.scatter(y, y_pred, color='blue', alpha=0.5, label='Predicted')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', label='Perfect Prediction')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "actual_vs_predicted.png"))
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"))
    plt.close()

# Generate plots at startup
generate_plots()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    if request.method == "POST":
        try:
            square_feet = float(request.form["square_feet"])
            num_rooms = float(request.form["num_rooms"])
            age = float(request.form["age"])
            distance = float(request.form["distance_to_city"])
            features = np.array([[square_feet, num_rooms, age, distance]])
            predicted_price = model.predict(features)[0]
            prediction_text = f"Estimated House Price: ${predicted_price:,.2f}"
        except Exception as e:
            prediction_text = f"Error: {str(e)}"
    
    # List of plots to show
    plot_files = [f"plots/{col}.png" for col in ['square_feet','num_rooms','age','distance_to_city(km)','price']]
    plot_files.append("plots/actual_vs_predicted.png")
    plot_files.append("plots/correlation_heatmap.png")
    
    return render_template("index.html", prediction_text=prediction_text, plot_files=plot_files)

if __name__ == "__main__":
    app.run(debug=True)
