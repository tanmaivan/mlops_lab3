#!/bin/bash

echo "Starting Housing Price Prediction API..."
echo "Note: Make sure you have:"
echo "1. MLflow server running (./run_mlflow.sh)"
echo "2. Trained model available in MLflow (python src/housing_flow.py run)"
echo "Starting API server..."

# Start the FastAPI service
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload