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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_lms_table(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/lms_loan_daily.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + ' row count:', df.count())
    
    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df



def process_bronze_clickstream_table(snapshot_date_str, bronze_clickstream_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/feature_clickstream.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + ' row count:', df.count())
    
    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_clickstream_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df



def process_bronze_other_tables(source_path, bronze_other_directory, filename, spark): # financials and attributes

    # load data
    df = spark.read.csv(source_path, header = True, inferSchema = True)
    print(f'{os.path.basename(source_path)} has {df.count()} rows.')

    # save to datamart
    filepath = bronze_other_directory + 'bronze_' + filename + '.csv'
    df.toPandas().to_csv(filepath, index = False)
    print('saved to: ', filepath)

    return df