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



# lms ====================================================================
def process_silver_lms(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())
    
    # clean data: enforce schema / data type
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType()
    }
    
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    
    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    
    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
    
    # keep full df for loan_prod and loan_dim
    df_full = df
    
    # drop columns extracted to loan_prod and loan_dim
    df_dropped = df.drop("loan_start_date", "tenure", "loan_amt", "due_amt")
    
    # save silver loan daily table (dropped version)
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df_dropped.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    # return both — full for loan_prod/loan_dim, dropped for reference
    return df_full, df_dropped


def process_silver_loan_prod(dfs, silver_directory):
    # union all monthly dfs and get distinct
    df_all = dfs[0]
    for df in dfs[1:]:
        df_all = df_all.union(df)
    
    loan_prod = df_all.select("tenure", "loan_amt", "due_amt").distinct().filter(col("due_amt") > 0)
    
    filepath = silver_directory + "loan_prod.parquet"
    loan_prod.write.mode("overwrite").parquet(filepath)
    print(f"\nSaved loan_prod (tenure, principal, monthly repayment): {loan_prod.count()} row(s) to {filepath}")
    
    return loan_prod


def process_silver_loan_dim(dfs, silver_directory):
    # union all monthly dfs and get distinct loan_id + loan_start_date
    df_all = dfs[0]
    for df in dfs[1:]:
        df_all = df_all.union(df)
    
    loan_dim = df_all.select("loan_id", "loan_start_date").distinct()
    
    filepath = silver_directory + "loan_dim.parquet"
    loan_dim.write.mode("overwrite").parquet(filepath)
    print(f"\nSaved loan_dim (loan id and start date): {loan_dim.count()} row(s) to {filepath}")
    
    return loan_dim




# attributes ====================================================================
def process_silver_attr(bronze_directory, silver_directory, spark):
    # connect to bronze table
    df = spark.read.csv(bronze_directory, header = True, inferSchema = True)


    # remove trailing "_" in age
    df = df.withColumn("Age", F.regexp_replace(col("Age"), "_$", ""))

    # enforce schema
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(), 
        "Age": IntegerType(), 
        "SSN": StringType(), 
        "Occupation": StringType(),
        "snapshot_date": DateType()
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # replace invalid SSN with null
    df = df.withColumn(
        "SSN",
        F.when(col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), col("SSN")).otherwise(None)
    )

    # replace invalid age with null
    df = df.withColumn(
        "Age",
        F.when((col("Age") > 0) & (col("Age") <= 120), col("Age")).otherwise(None)
    ) 

    # replace invalid occupation with null
    df = df.withColumn(
        "Occupation",
        F.when(col("Occupation") == '_______', None).otherwise(col("Occupation"))
    )

    print("\nfeature_attributes: silver layer cleaning done")

    # create directory path
    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)
    
    # save to datamart
    filepath = silver_directory + 'silver_cust_attr' + '.parquet'
    df.write.mode('overwrite').parquet(filepath)
    print("saved to: ", filepath)

    return df


# financials ====================================================================
def count_loan_type(entry, loan_type):
    if entry is None:
        return 0
    cleaned = re.sub(r",\s+and\s+", ", ", entry)
    loans = [loan.strip() for loan in cleaned.split(", ")]
    return loans.count(loan_type)

def process_silver_fin(bronze_directory, silver_directory, spark):
    # connect to bronze table
    df = spark.read.csv(bronze_directory, header = True, inferSchema = True)
    
    
    # remove trailing "_" in specified columns
    cols_to_clean = ["Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", 
                    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
                    "Monthly_Balance"]
    
    df = df.select(
        [
            F.regexp_replace(col(c), r"^_+|_+$", "").alias(c)
            if c in cols_to_clean
            else col(c)
            for c in df.columns
        ]
    )
    
    
    # convert Credit_History_Age to integer (years)
    df = df.withColumn("years", F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Year", 1).cast("integer")) \
           .withColumn("months", F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Month", 1).cast("integer")) \
           .withColumn("Credit_History_Age_Years", F.round(col("years") + col("months") / 12, 2)) \
           .drop("years", "months", "Credit_History_Age")
    
    
    # enforce schema
    column_type_map = {
        'Customer_ID': StringType(),
        'Annual_Income': FloatType(), 
        'Monthly_Inhand_Salary': FloatType(), 
        'Num_Bank_Accounts': IntegerType(), 
        'Num_Credit_Card': IntegerType(),
        'Interest_Rate': FloatType(),
        'Num_of_Loan': IntegerType(), 
        'Type_of_Loan': StringType(), 
        'Delay_from_due_date': IntegerType(), 
        'Num_of_Delayed_Payment': IntegerType(), 
        'Changed_Credit_Limit': FloatType(), 
        'Num_Credit_Inquiries': IntegerType(), 
        'Credit_Mix': StringType(), 
        'Outstanding_Debt': FloatType(),
        'Credit_Utilization_Ratio': FloatType(),
        'Credit_History_Age_Years': FloatType(), 
        'Payment_of_Min_Amount': StringType(), 
        'Total_EMI_per_month': FloatType(),
        'Amount_invested_monthly': FloatType(),
        'Payment_Behaviour': StringType(),
        'Monthly_Balance': FloatType(),
        'snapshot_date': DateType()
    }
    
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    
    
    # handle anomalous values for specified quantitative columns
    cols_iqr = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Delay_from_due_date', 
                'Num_of_Delayed_Payment', 'Num_Credit_Inquiries']
    
    for c in cols_iqr:
        df = df.withColumn(c, F.when(col(c)<0, None).otherwise(col(c)))
        q1 = df.approxQuantile(c, [0.25], 0.0)[0]
        q3 = df.approxQuantile(c, [0.75], 0.0)[0]
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        df = df.withColumn(
            c,
            F.when((col(c) >= 0) & (col(c) <= upper), col(c)).otherwise(None)
        )

    # handle negative values for other quantitative columns
    cols_others = ['Annual_Income', 'Interest_Rate', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
                   'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

    for c in cols_others:
        df = df.withColumn(c, F.when(col(c)<0, None).otherwise(col(c)))

    # handle anomalous interest_rate
    df = df.withColumn('Interest_Rate',
                       F.when(col('Interest_Rate')>100, None).otherwise(col("Interest_Rate"))
                      )
    

    # replace Type_of_Loan with columns specifying type of loan and the count
    ## 1. get unique loan types 
    loan_series = df.select("Type_of_Loan").dropna().toPandas()["Type_of_Loan"]
    
    all_loans = []
    for entry in loan_series:
        cleaned = re.sub(r",\s+and\s+", ", ", entry)
        loans = [loan.strip() for loan in cleaned.split(", ")]
        all_loans.extend(loans)
    
    unique_loans = sorted(set(all_loans))
    
    ## 2. add a column for each loan type
    for loan_type in unique_loans:
        col_name = "Loan_" + loan_type.replace(" ", "_").replace("-", "_")
        
        count_udf = udf(lambda x: count_loan_type(x, loan_type), IntegerType())
        
        df = df.withColumn(col_name, count_udf(col("Type_of_Loan")))
    
    ## 3. drop original column
    df = df.drop("Type_of_Loan")
    
    
    # replace "_" in Credit_Mix 
    df = df.withColumn("Credit_Mix",
        F.when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix"))
    )
    
    # separate Payment_Behaviour into 2 columns
    valid_pattern = r"^[A-Za-z]+_spent_[A-Za-z]+_value_payments$"
    
    df = (
        df
        .withColumn("parts", F.split(col("Payment_Behaviour"), "_"))
        .withColumn("Spending_Behaviour",
                    F.when(col("Payment_Behaviour").rlike(valid_pattern), col("parts")[0]).otherwise(None)
                   )
        .withColumn("Payments_Size",
                    F.when(col("Payment_Behaviour").rlike(valid_pattern), col("parts")[2]).otherwise(None)
                   )
        .drop("Payment_Behaviour", "parts")
    )

    print("\nfeature_financials: silver layer cleaning done")
    
    # create directory path
    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)
    
    # save to datamart
    filepath = silver_directory + 'silver_cust_fin' + '.parquet'
    df.write.mode('overwrite').parquet(filepath)
    print("saved to: ", filepath)
    
    return df


# clickstream ====================================================================
def process_silver_clickstream(snapshot_date_str, bronze_directory, silver_directory, spark):
    # connect to bronze table
    partition_name = "bronze_clickstream_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    
    # enforce schema
    for c in df.columns:
        if c.startswith("fe_"):
            df = df.withColumn(c, col(c).cast(IntegerType()))
    
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    
    # check and drop duplicates
    df = df.dropDuplicates()


    # save silver table
    partition_name = "silver_clickstream_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


