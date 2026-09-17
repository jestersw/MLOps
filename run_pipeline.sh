#!/bin/bash
echo "Starting pipeline at $(date)"
cd /Users/jester/Desktop/MLOps

# Stage 1: Data Engineering
python code/datasets/process_data.py

# Stage 2: Model Engineering
python code/models/train_model.py

# Stage 3: Deployment
cd code/deployment
docker compose down
docker compose up -d --build

echo "Pipeline finished at $(date)"
