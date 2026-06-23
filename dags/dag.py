import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def branch_setup(**context):
    model_path = '/opt/airflow/model_bank/credit_model_xgb_2024_09_01.pkl'
    if os.path.exists(model_path):
        return 'skip_setup'
    return ['bronze_lms', 'bronze_click', 'bronze_attr', 'bronze_fin']

with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline',
    schedule_interval='0 0 1 * *',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 1, 1),
    catchup=True,
    max_active_runs = 1
) as dag:

    # --- branch: only run setup on first execution ---
    branch = BranchPythonOperator(
        task_id='branch_setup',
        python_callable=branch_setup
    )

    skip_setup = DummyOperator(task_id='skip_setup')

    # BRONZE =============================================
    bronze_lms = BashOperator(
        task_id='bronze_lms',
        bash_command='cd /opt/airflow && python scripts/data_processing_main.py --stage bronze_lms && echo "===== BRONZE LMS COMPLETE ====="'
    )
    bronze_click = BashOperator(
        task_id='bronze_click',
        bash_command='cd /opt/airflow && python scripts/data_processing_main.py --stage bronze_click && echo "===== BRONZE CLICKSTREAM COMPLETE ====="'
    )
    bronze_attr = BashOperator(
        task_id='bronze_attr',
        bash_command='cd /opt/airflow && python scripts/data_processing_main.py --stage bronze_attr && echo "===== BRONZE ATTRIBUTES COMPLETE ====="'
    )
    bronze_fin = BashOperator(
        task_id='bronze_fin',
        bash_command='cd /opt/airflow && python scripts/data_processing_main.py --stage bronze_fin && echo "===== BRONZE FINANCIALS COMPLETE ====="'
    )

    # SILVER =============================================
    silver_lms = BashOperator(
        task_id='silver_lms',
        bash_command='cd /opt/airflow && python scripts/data_processing_main.py --stage silver_lms && echo "===== SILVER LMS COMPLETE ====="'
    )
    silver_click = BashOperator(
        task_id='silver_click',
        bash_command='cd /opt/airflow && python scripts/data_processing_main.py --stage silver_click && echo "===== SILVER CLICKSTREAM COMPLETE ====="'
    )
    silver_attr = BashOperator(
        task_id='silver_attr',
        bash_command='cd /opt/airflow && python scripts/data_processing_main.py --stage silver_attr && echo "===== SILVER ATTRIBUTES COMPLETE ====="'
    )
    silver_fin = BashOperator(
        task_id='silver_fin',
        bash_command='cd /opt/airflow && python scripts/data_processing_main.py --stage silver_fin && echo "===== SILVER FINANCIALS COMPLETE ====="'
    )

    # GOLD =============================================
    gold_label = BashOperator(
        task_id='gold_label',
        bash_command='cd /opt/airflow && python scripts/data_processing_main.py --stage gold_label && echo "===== GOLD LABEL STORE COMPLETE ====="'
    )
    gold_features = BashOperator(
        task_id='gold_features',
        bash_command='cd /opt/airflow && python scripts/data_processing_main.py --stage gold_features && echo "===== GOLD FEATURE STORE COMPLETE ====="'
    )

    # MODEL TRAIN =============================================
    model_train = BashOperator(
        task_id='model_train',
        bash_command='cd /opt/airflow && python scripts/model_train_main.py && echo "===== MODEL TRAINING COMPLETE ====="'
    )

    # --- join after setup branch ---
    join = DummyOperator(
        task_id='join',
        trigger_rule='none_failed_min_one_success',
    )

    # INFERENCE + MONITORING (Monthly) =============================================
    model_inference = BashOperator(
        task_id='model_inference',
        bash_command=(
            'cd /opt/airflow && python scripts/model_inference_main.py '
            '--snapshotdate {{ ds }} '
            '--modelname credit_model_xgb_2024_09_01.pkl '
            '&& echo "===== MODEL INFERENCE COMPLETE FOR {{ ds }} ====="'
        )
    )
    model_monitoring = BashOperator(
        task_id='model_monitoring',
        bash_command=(
            'cd /opt/airflow && python scripts/monitoring.py '
            '--snapshotdate {{ ds }} '
            '&& echo "===== MODEL MONITORING COMPLETE FOR {{ ds }} ====="'
        )
    )

    # DEPENDENCIES =====================================
    branch >> bronze_lms >> silver_lms >> [gold_label, gold_features]
    branch >> bronze_click >> silver_click >> gold_features
    branch >> bronze_attr >> silver_attr >> gold_features
    branch >> bronze_fin >> silver_fin >> gold_features
    [gold_label, gold_features] >> model_train >> join
    branch >> skip_setup >> join
    join >> model_inference >> model_monitoring