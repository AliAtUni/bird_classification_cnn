#!/bin/bash

# Start the MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000 &

# Wait for MLflow to fully start
sleep 10

# Execute the main script that triggers train.py
python main.py
