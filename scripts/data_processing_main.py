import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import re
import argparse

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import scripts.utils.data_processing_bronze_table
import scripts.utils.data_processing_silver_table
import scripts.utils.data_processing_gold_table


def get_spark():
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    current = datetime(start_date.year, start_date.month, 1)
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current = current + relativedelta(months=1)
    return dates


# BRONZE =============================================================

def run_bronze_lms():
    print(f'\n{"="*60}\nBRONZE - LMS\n{"="*60}')
    spark = get_spark()
    bronze_lms_directory = "datamart/bronze/lms/"
    if not os.path.exists(bronze_lms_directory):
        os.makedirs(bronze_lms_directory)
    for date_str in generate_first_of_month_dates("2023-01-01", "2025-11-01"):
        scripts.utils.data_processing_bronze_table.process_bronze_lms_table(date_str, bronze_lms_directory, spark)


def run_bronze_click():
    print(f'\n{"="*60}\nBRONZE - Clickstream\n{"="*60}')
    spark = get_spark()
    bronze_clickstream_directory = "datamart/bronze/clickstream/"
    if not os.path.exists(bronze_clickstream_directory):
        os.makedirs(bronze_clickstream_directory)
    for date_str in generate_first_of_month_dates("2023-01-01", "2024-12-01"):
        scripts.utils.data_processing_bronze_table.process_bronze_clickstream_table(date_str, bronze_clickstream_directory, spark)


def run_bronze_attr():
    print(f'\n{"="*60}\nBRONZE - Attributes\n{"="*60}')
    spark = get_spark()
    if not os.path.exists("datamart/bronze/attributes/"):
        os.makedirs("datamart/bronze/attributes/")
    scripts.utils.data_processing_bronze_table.process_bronze_other_tables(
        "data/features_attributes.csv", "datamart/bronze/attributes/", "cust_attr", spark
    )


def run_bronze_fin():
    print(f'\n{"="*60}\nBRONZE - Financials\n{"="*60}')
    spark = get_spark()
    if not os.path.exists("datamart/bronze/financials/"):
        os.makedirs("datamart/bronze/financials/")
    scripts.utils.data_processing_bronze_table.process_bronze_other_tables(
        "data/features_financials.csv", "datamart/bronze/financials/", "cust_fin", spark
    )


# SILVER =============================================================

def run_silver_lms():
    print(f'\n{"="*60}\nSILVER - LMS\n{"="*60}')
    spark = get_spark()
    silver_loan_daily_directory = "datamart/silver/loan_daily/"
    if not os.path.exists(silver_loan_daily_directory):
        os.makedirs(silver_loan_daily_directory)
    dfs_full = []
    for date_str in generate_first_of_month_dates("2023-01-01", "2025-11-01"):
        df_full, df_dropped = scripts.utils.data_processing_silver_table.process_silver_lms(
            date_str, "datamart/bronze/lms/", silver_loan_daily_directory, spark
        )
        dfs_full.append(df_full)
    scripts.utils.data_processing_silver_table.process_silver_loan_prod(dfs_full, "datamart/silver/loan_product/")
    scripts.utils.data_processing_silver_table.process_silver_loan_dim(dfs_full, "datamart/silver/loan_dimensions/")


def run_silver_attr():
    print(f'\n{"="*60}\nSILVER - Attributes\n{"="*60}')
    spark = get_spark()
    scripts.utils.data_processing_silver_table.process_silver_attr(
        "datamart/bronze/attributes/bronze_cust_attr.csv", "datamart/silver/attributes/", spark
    )


def run_silver_fin():
    print(f'\n{"="*60}\nSILVER - Financials\n{"="*60}')
    spark = get_spark()
    scripts.utils.data_processing_silver_table.process_silver_fin(
        "datamart/bronze/financials/bronze_cust_fin.csv", "datamart/silver/financials/", spark
    )


def run_silver_click():
    print(f'\n{"="*60}\nSILVER - Clickstream\n{"="*60}')
    spark = get_spark()
    silver_clickstream_directory = "datamart/silver/clickstream/"
    if not os.path.exists(silver_clickstream_directory):
        os.makedirs(silver_clickstream_directory)
    for date_str in generate_first_of_month_dates("2023-01-01", "2024-12-01"):
        scripts.utils.data_processing_silver_table.process_silver_clickstream(
            date_str, "datamart/bronze/clickstream/", silver_clickstream_directory, spark
        )


# GOLD =============================================================

def run_gold_label():
    print(f'\n{"="*60}\nGOLD - Label Store\n{"="*60}')
    spark = get_spark()
    gold_label_store_directory = "datamart/gold/label_store/"
    silver_loan_daily_directory = "datamart/silver/loan_daily/"
    if not os.path.exists(gold_label_store_directory):
        os.makedirs(gold_label_store_directory)
    for date_str in generate_first_of_month_dates("2023-01-01", "2025-11-01"):
        scripts.utils.data_processing_gold_table.process_labels_gold_table(
            date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd=30, mob=6
        )


def run_gold_features():
    print(f'\n{"="*60}\nGOLD - Feature Store\n{"="*60}')
    spark = get_spark()
    gold_feat_store_dir = "datamart/gold/feature_store/"

    scripts.utils.data_processing_gold_table.process_gold_copy(
        "datamart/silver/loan_dimensions/loan_dim.parquet", gold_feat_store_dir, "gold_loan_dim.parquet", spark
    )
    scripts.utils.data_processing_gold_table.process_gold_copy(
        "datamart/silver/loan_product/loan_prod.parquet", gold_feat_store_dir, "gold_loan_prod.parquet", spark
    )

    attr_df = scripts.utils.data_processing_gold_table.process_gold_attr(
        "datamart/silver/attributes/silver_cust_attr.parquet", gold_feat_store_dir, spark
    )
    fin_df = scripts.utils.data_processing_gold_table.process_gold_fin(
        "datamart/silver/financials/silver_cust_fin.parquet", gold_feat_store_dir, spark
    )
    click_df = scripts.utils.data_processing_gold_table.process_gold_clickstream(
        "datamart/silver/clickstream/silver_clickstream_daily_*.parquet",
        "datamart/silver/loan_dimensions/loan_dim.parquet",
        gold_feat_store_dir,
        spark
    )

    scripts.utils.data_processing_gold_table.process_gold_join(
        attr_df, fin_df, click_df, gold_feat_store_dir, spark
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run data processing stage")
    parser.add_argument("--stage", type=str, required=True,
                        help="bronze_lms | bronze_click | bronze_attr | bronze_fin | "
                             "silver_lms | silver_attr | silver_fin | silver_click | "
                             "gold_label | gold_features")
    args = parser.parse_args()

    stages = {
        "bronze_lms":    run_bronze_lms,
        "bronze_click":  run_bronze_click,
        "bronze_attr":   run_bronze_attr,
        "bronze_fin":    run_bronze_fin,
        "silver_lms":    run_silver_lms,
        "silver_attr":   run_silver_attr,
        "silver_fin":    run_silver_fin,
        "silver_click":  run_silver_click,
        "gold_label":    run_gold_label,
        "gold_features": run_gold_features,
    }

    if args.stage not in stages:
        raise ValueError(f"Unknown stage: {args.stage}. Choose from: {list(stages.keys())}")

    stages[args.stage]()
