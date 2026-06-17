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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


print(f'\n{"="*60}')
print('MODEL TRAINING')
print(f'{"="*60}')

print('\n--- Join features and labels ---')
# connect to label store
label_folder_path = "datamart/gold/label_store/gold_label_store_*"
label_store_sdf = spark.read.parquet(label_folder_path)
print(f'Loaded from: {label_folder_path}')
print("row_count:",label_store_sdf.count())

# rename snapshot_date col to indicate mob6
label_store_sdf = label_store_sdf.withColumnRenamed('snapshot_date', 'mob6_snapshot_date')


# get features ==============================
feature_folder_path = "datamart/gold/feature_store/gold_joined.parquet"
feature_store_sdf = spark.read.parquet(feature_folder_path)
print(f'Loaded from: {feature_folder_path}')
print("row_count:",feature_store_sdf.count())



# join features and labels ===================================
xy_sdf = feature_store_sdf.join(label_store_sdf, 
                                on='Customer_ID', 
                                how='outer')
print('Feature store joined with label store')

xyclean_sdf = xy_sdf.drop('label_def', 'loan_id', 'mob6_snapshot_date')
print("Columns dropped in joined table: 'label_def', 'loan_id', 'mob6_snapshot_date'")



# split data ===================================
print('\n--- Split train, test, oot ---')

# set up config
model_train_date_str = "2024-09-01" # remaining snapshot months: 2024-09-01 to 2025-01-01 (inference?)
train_test_period_months = 15
oot_period_months = 5
train_test_ratio = 0.8

config = {}
config["model_train_date_str"] = model_train_date_str
config["train_test_period_months"] = train_test_period_months
config["oot_period_months"] =  oot_period_months
config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
config["train_test_ratio"] = train_test_ratio 

print("Split configurations:")
pprint.pprint(config)

oot_sdf = xyclean_sdf.filter(
    (col('snapshot_date')>= config['oot_start_date']) &
    (col('snapshot_date')<= config['oot_end_date'])
)

oot_pdf = oot_sdf.toPandas()
print("OOT row counts:")
print(oot_pdf['snapshot_date'].value_counts().sort_index())


traintest_sdf = xyclean_sdf.filter(
    (col('snapshot_date') <= config['train_test_end_date']) &
    (col('snapshot_date') >= config['train_test_start_date'])
)

traintest_pdf = traintest_sdf.toPandas()


# convert snapshot_date to year and month columns
traintest_pdf["snapshot_date"] = pd.to_datetime(traintest_pdf["snapshot_date"])
traintest_pdf["snapshot_year"]  = traintest_pdf["snapshot_date"].dt.year
traintest_pdf["snapshot_month"] = traintest_pdf["snapshot_date"].dt.month
traintest_pdf["snapshot_quarter"] = traintest_pdf["snapshot_date"].dt.quarter

oot_pdf["snapshot_date"] = pd.to_datetime(oot_pdf["snapshot_date"])
oot_pdf["snapshot_year"]  = oot_pdf["snapshot_date"].dt.year
oot_pdf["snapshot_month"] = oot_pdf["snapshot_date"].dt.month
oot_pdf["snapshot_quarter"] = oot_pdf["snapshot_date"].dt.quarter

# define feature and label columns
exclude_cols = ['Customer_ID', 'label', 'snapshot_date']
feature_cols = [c for c in traintest_pdf.columns if c not in exclude_cols]
print(f'\nFeature columns ({len(feature_cols)} total): {feature_cols}')


# split data
X_temp = traintest_pdf[feature_cols].copy()
y_temp = traintest_pdf['label'].copy()

X_oot = oot_pdf[feature_cols].copy()
y_oot = oot_pdf['label'].copy()

# train/test split — 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.2,
    random_state=55,
    stratify=y_temp
)


sets = [
    ("train", y_train),
    ("test",  y_test),
    ("oot",   y_oot),
]
for name, y in sets:
    print(f"{name:6} | rows: {len(y):>7,} | bad rate: {y.mean():.4f}")


print('\n--- Post-split processing ---') 

# mean imputation ================================
mean_cols = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 
             'Num_of_Delayed_Payment', 'Age']
mean_impute_vals = {}
for c in mean_cols:
    mean_val = X_train[c].mean()
    mean_impute_vals[c] = mean_val
    X_train[c] = X_train[c].fillna(mean_val)
    X_test[c]  = X_test[c].fillna(mean_val)        
    X_oot[c]   = X_oot[c].fillna(mean_val)
print(f'Mean imputation applied to: {mean_cols}')


# median imputation ===================================
median_cols = ['Delay_from_due_date', 'Num_Credit_Inquiries', 
               'Monthly_Balance', 'Interest_Rate']
median_impute_vals = {}
for c in median_cols:
    median_val = X_train[c].median()
    median_impute_vals[c] = median_val
    X_train[c] = X_train[c].fillna(median_val)
    X_test[c]  = X_test[c].fillna(median_val)      
    X_oot[c]   = X_oot[c].fillna(median_val)       
print(f'Median imputation applied to: {median_cols}')


# clickstream fe_ imputation — fit on train clickers only ===================================
fe_cols = [f'fe_{i}_mean' for i in range(1, 21)]
fe_impute_vals = {}
for c in fe_cols:
    mean_val = X_train.loc[X_train['has_clickstream'] == 1, c].mean() 
    fe_impute_vals[c] = mean_val
    X_train[c] = X_train[c].fillna(mean_val)
    X_test[c]  = X_test[c].fillna(mean_val)
    X_oot[c]   = X_oot[c].fillna(mean_val)
print(f'Clickstream mean imputation applied to: {fe_cols}')


# dummy encoding for object columns ===================================
cat_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 
            'Spending_Behaviour', 'Payments_Size']

ohe = OneHotEncoder(drop = 'first', sparse_output = False, handle_unknown = 'ignore')
ohe.fit(X_train[cat_cols])

for X in [X_train, X_test, X_oot]:
    encoded = ohe.transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, 
                              columns = ohe.get_feature_names_out(cat_cols),
                              index = X.index)
    X.drop(columns = cat_cols, inplace=True)
    X[encoded_df.columns] = encoded_df

print(f'OHE encoding applied to: {cat_cols}')
print(f'Feature columns after encoding ({len(X_train.columns)} total): {list(X_train.columns)}')


# scaling ===================================
scaler = StandardScaler()
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
    'Delay_from_due_date', 'Interest_Rate', 'clickstream_days',
    
    # engineered features
    'EMI_to_Salary_Ratio', 'Debt_to_Income', 'Disposable_Income',
    'months12_to_annual_income_ratio',

    # loan counts
    'Loan_Auto_Loan', 'Loan_Credit_Builder_Loan',
    'Loan_Debt_Consolidation_Loan', 'Loan_Home_Equity_Loan',
    'Loan_Mortgage_Loan', 'Loan_Not_Specified', 'Loan_Payday_Loan',
    'Loan_Personal_Loan', 'Loan_Student_Loan'
] + [f'fe_{i}_mean' for i in range(1, 21)]

print(f'Standard scaling applied to {len(scale_cols)} columns: {scale_cols}')

scaler = StandardScaler()
transformer_stdscaler = scaler.fit(X_train[scale_cols])
X_train[scale_cols] = transformer_stdscaler.transform(X_train[scale_cols])
X_test[scale_cols]  = transformer_stdscaler.transform(X_test[scale_cols])
X_oot[scale_cols]   = transformer_stdscaler.transform(X_oot[scale_cols])

print('X_train rows count: ', X_train.shape[0])
print('X_test rows count: ', X_test.shape[0])
print('X_oot rows count: ', X_oot.shape[0])



print('\n--- Train XGB model ---')
# Define the XGBoost classifier
xgb_clf = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight = 3, random_state=55)

# Define the hyperparameter space to search
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 3],  # lower max_depth to simplify the model
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'gamma': [0, 0.1, 0.5],
    'min_child_weight': [5,10,20],
    # 'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [10, 20, 50]
}

# Create a scorer based on AUC score
auc_scorer = make_scorer(roc_auc_score)

# Set up the random search with cross-validation
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    scoring=auc_scorer,
    n_iter=50,  # Number of iterations for random search
    cv=5,       # Number of folds in cross-validation
    verbose=1,
    random_state=55,
    n_jobs=-1   # Use all available cores
)

# Perform the random search
random_search.fit(X_train, y_train)

# Output the best parameters and best score
print("Best parameters found: ", random_search.best_params_)
print("Best AUC score: ", random_search.best_score_)

# Evaluate the model on the train set
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_train)[:, 1]
train_auc_score = roc_auc_score(y_train, y_pred_proba)
print("Train AUC score: ", train_auc_score)

# Evaluate the model on the test set
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
test_auc_score = roc_auc_score(y_test, y_pred_proba)
print("Test AUC score: ", test_auc_score)

# Evaluate the model on the oot set
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_oot)[:, 1]
oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
print("OOT AUC score: ", oot_auc_score)

print("TRAIN GINI score: ", round(2*train_auc_score-1,3))
print("Test GINI score: ", round(2*test_auc_score-1,3))
print("OOT GINI score: ", round(2*oot_auc_score-1,3))




print('\n--- Save model artefacts ---')
# --- prepare model artefact to save ---
model_artefact = {}

model_artefact['model'] = best_model
model_artefact['model_version'] = "credit_model_xgb_"+config["model_train_date_str"].replace('-','_')

model_artefact['preprocessing_transformers'] = {}
model_artefact['preprocessing_transformers']['stdscaler'] = transformer_stdscaler
model_artefact['preprocessing_transformers']['mean_imputations'] = mean_impute_vals
model_artefact['preprocessing_transformers']['median_imputations'] = median_impute_vals
model_artefact['preprocessing_transformers']['clickstream_imputations'] = fe_impute_vals
model_artefact['preprocessing_transformers']['one_hot_encoding'] = ohe

model_artefact['data_dates'] = config

model_artefact['data_stats'] = {}
model_artefact['data_stats']['X_train'] = X_train.shape[0]
model_artefact['data_stats']['X_test'] = X_test.shape[0]
model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)

model_artefact['results'] = {}
model_artefact['results']['auc_train'] = train_auc_score
model_artefact['results']['auc_test'] = test_auc_score
model_artefact['results']['auc_oot'] = oot_auc_score
model_artefact['results']['gini_train'] = round(2*train_auc_score-1,3)
model_artefact['results']['gini_test'] = round(2*test_auc_score-1,3)
model_artefact['results']['gini_oot'] = round(2*oot_auc_score-1,3)

model_artefact['hp_params'] = random_search.best_params_


pprint.pprint(model_artefact)


# --- save artefact to model bank ---
# create model_bank dir
model_bank_directory = "model_bank/"

if not os.path.exists(model_bank_directory):
    os.makedirs(model_bank_directory)

# Full path to the file
file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

# Write the model to a pickle file
with open(file_path, 'wb') as file:
    pickle.dump(model_artefact, file)

print(f"Model saved to {file_path}")




# remaining snapshot months: 2024-09-01 to 2025-01-01 (inference?)
























