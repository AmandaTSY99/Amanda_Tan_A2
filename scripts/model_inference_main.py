import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print(f'\n{"="*60}')
print('INFERENCE PIPELINE')
print(f'{"="*60}')

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


def main(snapshotdate, modelname):
    print('\n\n--- starting job ---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")


    print('1. Extract features and model artefact')
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_bank_directory"] = "model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    
    pprint.pprint(config)
    


    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])


    feature_location = "datamart/gold/feature_store/gold_joined.parquet"
    
    # Load CSV into DataFrame - connect to feature store
    features_store_sdf = spark.read.parquet(feature_location)
    
    
    # extract feature store
    inference_sdf = features_store_sdf.filter((col("snapshot_date") == config["snapshot_date"]))
    print(f"Extracted inference features with {inference_sdf.count()} rows. Date of inference set: {config["snapshot_date"]}" )
    
    inference_pdf = inference_sdf.toPandas()



    print('\n2. Preprocessing')
    
    inference_pdf["snapshot_date"] = pd.to_datetime(inference_pdf["snapshot_date"])
    inference_pdf["snapshot_year"]  = inference_pdf["snapshot_date"].dt.year
    inference_pdf["snapshot_month"] = inference_pdf["snapshot_date"].dt.month
    inference_pdf["snapshot_quarter"] = inference_pdf["snapshot_date"].dt.quarter
    
    # mean imputation
    mean_cols = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 
                 'Num_of_Delayed_Payment', 'Age']
    
    for c in mean_cols:
        mean_val = model_artefact['preprocessing_transformers']['mean_imputations'][c]
        inference_pdf[c] = inference_pdf[c].fillna(mean_val)
    print(f'Mean imputation complete.')
    
    
    
    # median imputation
    median_cols = ['Delay_from_due_date', 'Num_Credit_Inquiries', 
                   'Monthly_Balance', 'Interest_Rate']
    
    for c in median_cols:
        median_val = model_artefact['preprocessing_transformers']['median_imputations'][c]
        inference_pdf[c] = inference_pdf[c].fillna(median_val)
    print(f'Median imputation complete.')
    
    
    # clickstream imputation
    fe_cols = [f'fe_{i}_mean' for i in range(1, 21)]
    
    for c in fe_cols:
        val = model_artefact['preprocessing_transformers']['clickstream_imputations'][c]
        inference_pdf[c] = inference_pdf[c].fillna(val)
    print(f'Mean imputed to clickstream columns.')
    
    
    cat_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 
                'Spending_Behaviour', 'Payments_Size']
    
    ohe = model_artefact['preprocessing_transformers']['one_hot_encoding']
    encoded = ohe.transform(inference_pdf[cat_cols])
    encoded_df = pd.DataFrame(encoded, 
                              columns = ohe.get_feature_names_out(cat_cols),
                              index = inference_pdf.index)
    inference_pdf.drop(columns = cat_cols, inplace = True)
    inference_pdf[encoded_df.columns] = encoded_df
    print(f'OHE applied to cat columns.')
    
    # scaling
    scale_cols = [
        # continuous numeric — large magnitude differences
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary',
        'Interest_Rate', 'Outstanding_Debt', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance',
        'Changed_Credit_Limit', 'Credit_Utilization_Ratio',
        'Credit_History_Age_Years', 'snapshot_year',
        
        # count columns — different ranges
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan',
        'Num_Credit_Inquiries', 'Num_of_Delayed_Payment',
        'Delay_from_due_date', 'Interest_Rate', 'clickstream_days', # to remove interest rate
        
        # engineered features
        'EMI_to_Salary_Ratio', 'Debt_to_Income', 'Disposable_Income',
        'months12_to_annual_income_ratio',
    
        # loan counts
        'Loan_Auto_Loan', 'Loan_Credit_Builder_Loan',
        'Loan_Debt_Consolidation_Loan', 'Loan_Home_Equity_Loan',
        'Loan_Mortgage_Loan', 'Loan_Not_Specified', 'Loan_Payday_Loan',
        'Loan_Personal_Loan', 'Loan_Student_Loan'
    ] + [f'fe_{i}_mean' for i in range(1, 21)]
    
    transformer_scaler = model_artefact['preprocessing_transformers']['stdscaler']
    inference_pdf[scale_cols] = transformer_scaler.transform(inference_pdf[scale_cols])
    print(f'StandardScaler applied to non-binary columns.')
    
    
    exclude_cols = ['Customer_ID', 'snapshot_date']
    feature_cols = [c for c in inference_pdf.columns if c not in exclude_cols]
    print(f'Feature columns: {len(feature_cols)} total.')
    
    
    print('\n3.Model predictions / inference')
    
    
    
    # load model
    model = model_artefact['model']
    
    
    # predict model
    x_inf = inference_pdf[feature_cols]
    y_inf = model.predict_proba(x_inf)[:, 1]
    
    
    # prepare output
    y_inf_pdf = inference_pdf[exclude_cols].copy()
    y_inf_pdf['model_name'] = config['model_name']
    y_inf_pdf['model_predictions'] = y_inf
    y_inf_pdf
    
    gold_directory = f'datamart/gold/model_predictions/{config['model_name'][:-4]}/'
    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save inference to gold table
    partition_name = config["model_name"][:-4] + "_predictions_" + config['snapshot_date_str'].replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inf_pdf).write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)


















