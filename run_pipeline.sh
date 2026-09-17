#!/bin/bash
# Run from the repo root no matter where the script is invoked from.
cd "$(dirname "$0")" || exit 1

# If you use a virtualenv, point PY at its interpreter so cron finds it
# (cron does NOT activate your venv). Example:
#   PY="$(pwd)/venv/bin/python"
PY="${PY:-python}"

echo "Starting pipeline at $(date)"

# Stage 0: Generate raw data.
# Required on a fresh clone because data/raw is gitignored and not committed.
"$PY" setup_data.py

# Stage 1: Data Engineering
"$PY" code/datasets/process_data.py

# Stage 2: Model Engineering
"$PY" code/models/train_model.py

# Stage 3: Deployment (API and app in separate containers)
cd code/deployment || exit 1
docker compose down
docker compose up -d --build

echo "Pipeline finished at $(date)"
