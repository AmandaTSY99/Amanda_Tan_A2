import sys
import os
sys.path.insert(0, '/opt/airflow')
import argparse
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pyspark
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, fbeta_score


print(f'\n{"="*60}')
print('MODEL MONITORING')
print(f'{"="*60}')

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


# parse args from DAG
parser = argparse.ArgumentParser()
parser.add_argument('--snapshotdate', required=True)
args = parser.parse_args()
snapshot_date_str = args.snapshotdate

# label date is 6 months after snapshot date
label_date_str = (datetime.strptime(snapshot_date_str, '%Y-%m-%d') + relativedelta(months=6)).strftime('%Y-%m-%d')


# performance metric: ROC AUC and f2 scores =====================
pred_path = (
    'datamart/gold/model_predictions/credit_model_xgb_2024_09_01/'
    'credit_model_xgb_2024_09_01_predictions_'
    + snapshot_date_str.replace('-', '_') + '.parquet'
)
pred = spark.read.parquet(pred_path).toPandas()

label_path = (
    'datamart/gold/label_store/gold_label_store_'
    + label_date_str.replace('-', '_') + '.parquet'
)
label = spark.read.parquet(label_path).toPandas()

merged = pred.merge(label[['Customer_ID', 'label']], on='Customer_ID', how='inner')
auc = roc_auc_score(merged['label'], merged['model_predictions'])
f2 = fbeta_score(merged['label'], (merged['model_predictions'] >= 0.5).astype(int), beta=2)


# stability metric: PSI =======================
def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    current = datetime(start_date.year, start_date.month, 1)
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current = current + relativedelta(months=1)
    return dates

def calculate_psi(reference, comparison, bins=10):
    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)
    
    comparison = np.clip(comparison, breakpoints[0], breakpoints[-1])
    
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    comp_counts, _ = np.histogram(comparison, bins=breakpoints)
    
    ref_pct = ref_counts / len(reference)
    comp_pct = comp_counts / len(comparison)
    
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    comp_pct = np.where(comp_pct == 0, 1e-4, comp_pct)
    
    return ((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)).sum()

train_test = pd.DataFrame()

for date_str in generate_first_of_month_dates('2023-01-01', '2024-03-01'):
    temp = spark.read.parquet(
        'datamart/gold/model_predictions/credit_model_xgb_2024_09_01/'
        'credit_model_xgb_2024_09_01_predictions_'
        + date_str.replace('-', '_') + '.parquet'
    ).toPandas()
    train_test = pd.concat([train_test, temp], ignore_index=True)

reference = train_test['model_predictions']
psi = calculate_psi(reference, pred['model_predictions'])


# write metrics to datamart ========================
metrics = pd.DataFrame([{
    'snapshot_date': pd.Timestamp(snapshot_date_str),
    'model_name': 'credit_model_xgb_2024_09_01',
    'roc_auc': auc,
    'f2_score': f2,
    'psi': psi
}])

output_dir = 'datamart/gold/model_monitoring'
os.makedirs(output_dir, exist_ok=True)
output_path = f'{output_dir}/metrics_{snapshot_date_str.replace("-", "_")}.parquet'
spark.createDataFrame(metrics).write.mode('overwrite').parquet(output_path)

print(f'Snapshot: {snapshot_date_str} | AUC: {auc:.4f} | F2: {f2:.4f} | PSI: {psi:.4f}')


























