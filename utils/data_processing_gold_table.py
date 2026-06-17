import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse
import re

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# label engineering =================================================================================
def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

# loan_dim and loan_prod =====================================================================
def process_gold_copy(silver_filepath, gold_directory, gold_filename, spark):
    # connect to silver layer
    df = spark.read.parquet(silver_filepath)
    
    # create directory path
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save to datamart
    filepath = gold_directory + gold_filename
    df.write.mode('overwrite').parquet(filepath)
    print(f'Saved {gold_filename} to: {filepath}, {df.count()} row(s)')
    
    return df



# attributes =================================================================================
def process_gold_attr(silver_directory, gold_directory, spark):
    
    # connect to silver layer
    df = spark.read.parquet(silver_directory)

    # drop columns - Name, SSN
    df = df.drop('Name', 'SSN')

    # impute 'Unknown' for null occupations
    df = df.fillna({'Occupation': 'Unknown'})

    # # mean imputation for age
    # mean_val = df.select(F.mean(col('Age'))).collect()[0][0]
    # df = df.fillna({'Age': mean_val})
    
    # create directory path
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    # save to datamart
    filepath = gold_directory + 'gold_cust_attr.parquet'
    df.write.mode('overwrite').parquet(filepath)
    print(f'Saved gold attribute table to: {filepath}, {df.count()} row(s), {len(df.columns)} col(s)')
    print(f'Columns: {df.columns}')

    # return df for joining with other dfs
    return df



# financials =================================================================================
def process_gold_fin(silver_directory, gold_directory, spark):
    
    # connect to silver layer
    df = spark.read.parquet(silver_directory)
    
    
    # fillna(0) — null means no credit change made
    df = df.fillna({'Changed_Credit_Limit': 0})
    
    # fillna(unknown) — categorical nulls
    df = df.fillna({'Credit_Mix': 'Unknown',
                    'Spending_Behaviour': 'Unknown',
                    'Payments_Size': 'Unknown'})

    
    
    # feature engineering
    df = df.withColumn('months12_to_annual_income_ratio',
        (col('Monthly_Inhand_Salary') * 12) / col('Annual_Income')) # detect excessively high annual income

    df = df.withColumn('EMI_to_Salary_Ratio',
                       (col('Total_EMI_per_month') / col('Monthly_Inhand_Salary')))
    
    df = df.withColumn('Debt_to_Income',
                       (col('Outstanding_Debt') / col('Annual_Income')))
    
    df = df.withColumn('Disposable_Income',
                       (col('Monthly_Inhand_Salary') - col('Total_EMI_per_month') - col('Amount_invested_monthly')))


    # log transform income-related columns
    income_cols = ['Annual_Income', 'Monthly_Inhand_Salary',
                   'Outstanding_Debt', 'Total_EMI_per_month',
                   'Amount_invested_monthly', 'Monthly_Balance']
    
    for c in income_cols:
        df = df.withColumn(c, F.log1p(col(c)))



    # save to datamart
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    filepath = gold_directory + 'gold_cust_fin.parquet'
    df.write.mode('overwrite').parquet(filepath)
    print(f'Saved gold customer financials table to: {filepath}, {df.count()} row(s), {len(df.columns)} col(s)')
    print(f'Columns: {df.columns}')
    
    return df




# clickstream =================================================================================
def process_gold_clickstream(silver_click_dir, silver_loandim_dir, gold_directory, spark): 

    # connect to silver layer
    df_loandim = spark.read.parquet(silver_loandim_dir)
    df_click = spark.read.parquet(silver_click_dir)

    # get customer id and loan_start_date from df_loandim
    df_loandim = df_loandim.withColumn("Customer_ID",
                                       F.split(col("loan_id"), r"_(?=\d{4}_\d{2}_\d{2})")[0]).drop("loan_id")
    
    # filter clickstream data for dates before loan start date for each customer
    df_click = df_click.join(df_loandim, on="Customer_ID", how="left")
    df_click = df_click.filter(col("snapshot_date") < col("loan_start_date"))
    
    # per customer aggregate clickstream data: find daily mean anonymised feature and row count
    fe_cols = [c for c in df_click.columns if c.startswith("fe_")]
    
    agg_exprs = [F.mean(col(c)).alias(f"{c}_mean") for c in fe_cols]
    agg_exprs.append(F.count("*").alias("clickstream_days"))
    
    df = df_click.groupBy("Customer_ID").agg(*agg_exprs)
    print(f'Aggregated clickstream per customer: {df.count()} unique customers')
    print(f'Aggregated feature columns: {[f"{c}_mean" for c in fe_cols] + ["clickstream_days"]}')

    # create directory path
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    # save to datamart
    filepath = gold_directory + 'gold_clickstream.parquet'
    df.write.mode('overwrite').parquet(filepath)
    print(f'Saved gold clickstream table to: {filepath}, {df.count()} row(s), {len(df.columns)} col(s)')

    # return df for joining with other dfs
    return df




# join attr, fin, click ========================================================================
def process_gold_join(attr_df, fin_df, click_df, gold_directory, spark):

    # join attributes and financials
    df = attr_df.join(fin_df, on=["Customer_ID", "snapshot_date"], how="outer")

    # add has_clickstream column to df_click
    click_df_agg = click_df.withColumn("has_clickstream", F.lit(1))

    # join click to df
    df_gold = df.join(click_df_agg, on="Customer_ID", how="left")
    
    # Customers with no clickstream has_clickstream = null. fill with 0
    df_gold = df_gold.fillna({"has_clickstream": 0,
                             "clickstream_days": 0})

    # create directory path
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)

    # save to datamart
    filepath = gold_directory + 'gold_joined.parquet'
    df_gold.write.mode('overwrite').parquet(filepath)
    print(f'Saved unified gold feature table to: {filepath}, {df_gold.count()} row(s), {len(df_gold.columns)} col(s)')
    print(f'Feature columns: {df_gold.columns}')

    # return df
    return df_gold
    

# # split into train, test, oot ===================================================================
# def process_gold_splitdata (df_x, y_dir, gold_directory, spark):
#     # 1. join x and y df
#     # connect to loan_dim silver
#     loan_dim = spark.read.parquet("datamart/silver/loan_dimensions/loan_dim.parquet")
    
#     # connect to y df
#     df_y = spark.read.parquet(y_dir)


    
#     # join loan_start_date back to df_y
#     df_y = df_y.join(loan_dim, on = 'loan_id', how = 'left')

#     # rename snapshot_date in df_y to indicate mob6
#     df_y = df_y.withColumnRenamed('snapshot_date', 'mob6_snapshot_date')
    
#     # join df_x and df_y using customer_id and loan start date / snapshot date
#     df_all = df_x.join(df_y, 
#                        on = [df_x['Customer_ID'] == df_y['Customer_ID'],
#                              df_x['snapshot_date'] == df_y['loan_start_date']],
#                        how = 'inner').drop(df_x['Customer_ID'], df_y['loan_start_date'])


    
#     # 2. split into train, test, oot
#     oot_cutoff = '2024-09-01' 
#     print(f'OOT cutoff date: {oot_cutoff}')
    
#     df_oot   = df_all.filter(col('snapshot_date') >= oot_cutoff)
#     df_model = df_all.filter(col('snapshot_date') <  oot_cutoff)
#     print(f'Model data (train+test): {df_model.count()} rows')
#     print(f'OOT data: {df_oot.count()} rows')

    
#     # convert to pandas for train/test split
#     df_model_pd = df_model.toPandas()
#     df_oot_pd   = df_oot.toPandas()

#     # convert snapshot_date to year and month columns
#     df_model_pd["snapshot_date"] = pd.to_datetime(df_model_pd["snapshot_date"])
#     df_model_pd["snapshot_year"]  = df_model_pd["snapshot_date"].dt.year
#     df_model_pd["snapshot_month"] = df_model_pd["snapshot_date"].dt.month
    
#     df_oot_pd["snapshot_date"] = pd.to_datetime(df_oot_pd["snapshot_date"])
#     df_oot_pd["snapshot_year"]  = df_oot_pd["snapshot_date"].dt.year
#     df_oot_pd["snapshot_month"] = df_oot_pd["snapshot_date"].dt.month

    
#     # define feature and label columns
#     exclude_cols = ['Customer_ID', 'mob6_snapshot_date', 
#                     'loan_id', 'label', 'label_def', 'snapshot_date']
#     feature_cols = [c for c in df_model_pd.columns if c not in exclude_cols]
#     print(f'\nFeature columns ({len(feature_cols)} total): {feature_cols}')
    
#     X = df_model_pd[feature_cols].copy()
#     y = df_model_pd['label'].copy()
    
#     X_oot = df_oot_pd[feature_cols].copy()
#     y_oot = df_oot_pd['label'].copy()
    
#     # train/test split — 80/20
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=0.2,
#         random_state=55,
#         stratify=y
#     )


    
#     # 3. post-split processing
#     print('\n--- Post-split processing ---')

#     # mean imputation — fit on train only
#     mean_cols = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 
#                  'Num_of_Delayed_Payment', 'Age']
#     for c in mean_cols:
#         mean_val = X_train[c].mean()                   
#         X_train[c] = X_train[c].fillna(mean_val)
#         X_test[c]  = X_test[c].fillna(mean_val)        
#         X_oot[c]   = X_oot[c].fillna(mean_val)
#     print(f'Mean imputation applied to: {mean_cols}')

#     # median imputation — fit on train only
#     median_cols = ['Delay_from_due_date', 'Num_Credit_Inquiries', 
#                    'Monthly_Balance', 'Interest_Rate']
#     for c in median_cols:
#         median_val = X_train[c].median()               
#         X_train[c] = X_train[c].fillna(median_val)
#         X_test[c]  = X_test[c].fillna(median_val)      
#         X_oot[c]   = X_oot[c].fillna(median_val)       
#     print(f'Median imputation applied to: {median_cols}')

    
#     # clickstream fe_ imputation — fit on train clickers only
#     fe_cols = [f'fe_{i}_mean' for i in range(1, 21)]
#     for c in fe_cols:
#         mean_val = X_train.loc[X_train['has_clickstream'] == 1, c].mean() 
#         X_train[c] = X_train[c].fillna(mean_val)
#         X_test[c]  = X_test[c].fillna(mean_val)
#         X_oot[c]   = X_oot[c].fillna(mean_val)
#     print(f'Clickstream mean imputation applied to: {fe_cols}')

#     # dummy encoding for object columns
#     cat_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 
#                 'Spending_Behaviour', 'Payments_Size']
#     X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
#     X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=True)
#     X_oot   = pd.get_dummies(X_oot,   columns=cat_cols, drop_first=True)
#     print(f'Dummy encoding applied to: {cat_cols}')
#     print(f'Feature columns after encoding ({len(X_train.columns)} total): {list(X_train.columns)}')
    
#     # scaling for 
#     scaler = StandardScaler()
#     scale_cols = [
#         # continuous numeric — large magnitude differences
#         'Age', 'Annual_Income', 'Monthly_Inhand_Salary',
#         'Outstanding_Debt', 'Total_EMI_per_month',
#         'Amount_invested_monthly', 'Monthly_Balance',
#         'Changed_Credit_Limit', 'Credit_Utilization_Ratio',
#         'Credit_History_Age_Years', 'snapshot_year', 'snapshot_month',
        
#         # count columns — different ranges
#         'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan',
#         'Num_Credit_Inquiries', 'Num_of_Delayed_Payment',
#         'Delay_from_due_date', 'Interest_Rate', 'clickstream_days',
        
#         # engineered features
#         'EMI_to_Salary_Ratio', 'Debt_to_Income', 'Disposable_Income',
#         'months12_to_annual_income_ratio',

#         # loan counts
#         'Loan_Auto_Loan', 'Loan_Credit_Builder_Loan',
#         'Loan_Debt_Consolidation_Loan', 'Loan_Home_Equity_Loan',
#         'Loan_Mortgage_Loan', 'Loan_Not_Specified', 'Loan_Payday_Loan',
#         'Loan_Personal_Loan', 'Loan_Student_Loan'
#     ] + [f'fe_{i}_mean' for i in range(1, 21)]

#     print(f'Standard scaling applied to {len(scale_cols)} columns')
    
#     X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
#     X_test[scale_cols]  = scaler.transform(X_test[scale_cols])
#     X_oot[scale_cols]   = scaler.transform(X_oot[scale_cols])

#     print('\n--- Split Summary ---')
#     print(f"Train: {len(X_train)} rows, bad rate: {y_train.mean():.1%}")
#     print(f"Test:  {len(X_test)}  rows, bad rate: {y_test.mean():.1%}")
#     print(f"OOT:   {len(X_oot)}   rows, bad rate: {y_oot.mean():.1%}")
#     print(f'Total feature columns: {len(X_train.columns)}')
#     print(f'Features:{X_train.columns.tolist()}')


    
#     # 4. save to datamart
#     if not os.path.exists(gold_directory):
#         os.makedirs(gold_directory)
    
#     spark.createDataFrame(X_train).write.mode('overwrite').parquet(gold_directory + 'X_train.parquet')
#     spark.createDataFrame(X_test).write.mode('overwrite').parquet(gold_directory  + 'X_test.parquet')
#     spark.createDataFrame(X_oot).write.mode('overwrite').parquet(gold_directory   + 'X_oot.parquet')
#     spark.createDataFrame(y_train.to_frame()).write.mode('overwrite').parquet(gold_directory + 'y_train.parquet')
#     spark.createDataFrame(y_test.to_frame()).write.mode('overwrite').parquet(gold_directory  + 'y_test.parquet')
#     spark.createDataFrame(y_oot.to_frame()).write.mode('overwrite').parquet(gold_directory   + 'y_oot.parquet')
    
#     print(f"Saved all splits to: {gold_directory}")
    
#     return X_train, X_test, X_oot, y_train, y_test, y_oot










