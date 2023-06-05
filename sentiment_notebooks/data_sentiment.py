
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import time

from transformers import pipeline
from transformers.pipelines.text_classification import TextClassificationPipeline

from polygon import RESTClient
from polygon.rest.models import *

from termcolor import colored as cl
import requests
import datetime
  
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
def get_data_from_catalog(db_name, table, glue_context):
    dyf = glueContext.create_dynamic_frame.from_catalog(database=db_name, table_name=table)
    pandas_df = dyf.toDF().toPandas()
    
    return pandas_df
def write_aws_table(
    pandas_df, 
    s3_path, 
    glue_context, 
    spark_session, 
    db_name, 
    table_name
):
    s3output = glue_context.getSink(
      path=s3_path,
      connection_type="s3",
      updateBehavior="UPDATE_IN_DATABASE",
      partitionKeys=[],
      compression="snappy",
      enableUpdateCatalog=True,
      transformation_ctx="s3output",
    )
    s3output.setCatalogInfo(
      catalogDatabase=db_name, catalogTableName=table_name
    )
    s3output.setFormat("glueparquet")
    data_spark_df = spark.createDataFrame(pandas_df)
    data_dyf = DynamicFrame.fromDF(data_spark_df, glue_context, f"{table_name}_df")
    s3output.writeFrame(data_dyf)
wfc_tgt_news = get_data_from_catalog("project", "transformed_wfc_tgt_news", glueContext)
wfc_tgt_news
# !pip install -q transformers
distilroberta_pipeline = pipeline(model= "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
distilroberta_pipeline
def get_polarity(
    df: DataFrame,
    sentiment_model_pipe: TextClassificationPipeline
  ):

  #Set polarity with 0 as a placeholder
  df["polarity"] = 0

  #Get sentiment results for each title in the dataset
  df["sent_result"] = df["title"].apply(lambda title: sentiment_model_pipe(title)[0])

  #Extract both the predicted label and score from the model's results
  df["pred_label"] = df["sent_result"].apply(lambda x: x.get('label'))
  df["pred_score"] = df["sent_result"].apply(lambda x: x.get('score'))
  df = df.drop("sent_result", axis= 1)

  #Set the condition to be later passed to "np.select" in order to identify each case
  conditions = [
    (df['pred_label'] == 'positive'),
    (df['pred_label'] == 'negative'),
    (df['pred_label'] == 'neutral')
  ]

  #Set the choices to transform each prob into a polarity
  choices = [
      df['pred_score'],
      -1*df['pred_score'],
      0
  ]

  df["polarity"] = np.select(conditions, choices, default=0)

  return df

init_pred_time_sec = time.time()
analyzed_df = get_polarity(wfc_tgt_news, distilroberta_pipeline)
print(f"Total prediction time for {parquet_data.shape[0]} rows was {time.time() - init_pred_time_sec}")
ticket_daily_avg_sentiment = (analyzed_df[["ticker", "date", "polarity"]]
                              .groupby(["ticker", "date"])
                              .agg(
                                  polarity_mean = pd.NamedAgg(column="polarity", aggfunc="mean"),
                              )
)
ticket_daily_avg_sentiment
def get_missing_dates(values, init_date, end_date):
  values = set(values)
  missing_dates = []
  init_date_dt = datetime.datetime.strptime(init_date, '%Y/%m/%d')
  end_date_dt = datetime.datetime.strptime(end_date, '%Y/%m/%d')
  date_diff = (end_date_dt - init_date_dt).days

  for day_index in range(1, date_diff + 1):
    curr_date = init_date_dt + datetime.timedelta(days = day_index)
    curr_date_str = curr_date.strftime('%Y/%m/%d')
    #print(f"Current date in datetime: {curr_date}")

    if init_date_dt <= curr_date <= end_date_dt and curr_date_str in values:
      #print(f"{curr_date_str} date not missing")
      continue
    else:
      missing_dates.append(curr_date_str)
      #print(f"Missing date: {curr_date_str}")
  
  return missing_dates
INIT_DATE = "2019/01/01"
END_DATE = "2023/06/03"

def get_defaul_polarity_for_missing_dates(
      df, 
      date_col= "date", 
      init_date= "2019/01/01", 
      end_date= "2023/06/03", 
      default_polarity= 0.0
    ):
  target_tickers = pd.unique(df.index.get_level_values("ticker"))
  new_rows = []
  
  for ticker in target_tickers:
    ticker_data = df.loc[df.index.get_level_values("ticker") == ticker]
    ticker_dates = pd.unique(ticker_data.index.get_level_values("date"))
    missing_ticker_dates = get_missing_dates(values= ticker_dates, init_date= init_date, end_date= end_date)
    print(f"Ticker: {ticker}")
    print(f"Missing Dates between {init_date} and {end_date}: {len(missing_ticker_dates)}")
    ticker_new_rows = [[ticker, missing_date, default_polarity] for missing_date in missing_ticker_dates]
    new_rows.extend(ticker_new_rows)

  print(f"Total missing Dates between {init_date} and {end_date}: {len(new_rows)}")
  return new_rows

new_records = get_defaul_polarity_for_missing_dates(ticket_daily_avg_sentiment)
new_records_df = pd.DataFrame(new_records, columns= ["ticker", "date", "polarity_mean"])

full_period_sentiment_df = pd.concat([
    ticket_daily_avg_sentiment["polarity_mean"].reset_index(),
    new_records_df
    ])

full_period_sentiment_df
# full_period_sentiment_df.to_parquet("s3://project-2023-datalake/transformed/cleaned/wfc_tgt_full_sentiment_2019_to_2023.parquet")

write_aws_table(
    full_period_sentiment_df, 
    "s3://project-2023-datalake/transformed/cleaned/wfc_tgt_full_sentiment/", 
    glueContext, 
    spark, 
    "project, 
    "transformed_wfc_tgt_full_sentiment"
)
from torch.utils.data import Dataset, DataLoader
import os
import site

# reload() has been moved to importlib in Python 3.4
# https://docs.python.org/3/whatsnew/3.4.html#importlib
from importlib import reload

from setuptools.command import easy_install
install_path = os.environ['GLUE_INSTALLATION']
easy_install.main( ["--install-dir", install_path, "torch"] )
reload(site)


import torch
print(torch.__version__)
job.commit()