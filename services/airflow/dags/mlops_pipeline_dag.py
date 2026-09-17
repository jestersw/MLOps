from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

# Dynamically find the root repository path
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'mlops_wine_pipeline',
    default_args=default_args,
    description='Automated MLOps pipeline',
    schedule_interval='*/5 * * * *',  # Runs every 5 minutes
    catchup=False,
)

t0 = BashOperator(
    task_id='setup_data',
    bash_command=f'cd {REPO_DIR} && python setup_data.py',
    dag=dag,
)

t1 = BashOperator(
    task_id='data_engineering',
    bash_command=f'cd {REPO_DIR} && python code/datasets/process_data.py',
    dag=dag,
)

t2 = BashOperator(
    task_id='model_engineering',
    bash_command=f'cd {REPO_DIR} && python code/models/train_model.py',
    dag=dag,
)

t3 = BashOperator(
    task_id='deployment',
    bash_command=f'cd {REPO_DIR}/code/deployment && docker compose down && docker compose up -d --build',
    dag=dag,
)

t0 >> t1 >> t2 >> t3