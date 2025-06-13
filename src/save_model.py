import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import os
import mlflow

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load the best model from MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Housing Price Prediction")

# Get the latest run with the best model
runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name("Housing Price Prediction").experiment_id])
best_run = runs.loc[runs['metrics.test_r2'].idxmax()]

# Load model from artifacts
model_name = best_run['tags.mlflow.runName'].split('_')[1]  
model_dir = f"{model_name}_model"
model_path = os.path.join(
    "artifacts",
    "1",
    best_run['run_id'],
    "artifacts",
    model_dir
)
model = mlflow.sklearn.load_model(model_path)

# Load and preprocess data
df = pd.read_csv('data/housing.csv')

# Create and fit encoders
categorical_cols = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea',
    'furnishingstatus'
]

encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    encoder.fit(df[col].unique())
    encoders[col] = encoder
    # Save encoder
    with open(f"models/encoder_{col}.pkl", 'wb') as f:
        pickle.dump(encoder, f)

# Transform categorical columns
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = encoders[col].transform(df[col])

X = df_encoded.drop('price', axis=1)
y = df['price'].values.reshape(-1, 1)

# Create and fit scalers
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(X)
scaler_y.fit(y)

# Save model and scalers
with open("models/gradient_boosting_model.pkl", 'wb') as f:
    pickle.dump(model, f)

with open("models/scaler_X.pkl", 'wb') as f:
    pickle.dump(scaler_X, f)

with open("models/scaler_y.pkl", 'wb') as f:
    pickle.dump(scaler_y, f)

print("Model and preprocessors saved successfully!")