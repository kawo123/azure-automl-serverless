#%%
# Import the necessary packages. The Open Datasets package contains
# a class representing each data source (NycTlcGreen for example)
# to easily filter date parameters before downloading

from azureml.opendatasets import NycTlcGreen
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import pandas as pd

#%%
# Create a dataframe to hold the taxi data. When working in a non-
# Spark environment, Open Datasets only allows downloading one month
# of data at a time with certain classes to avoid MemoryError with
# large datasets

green_taxi_df = pd.DataFrame([])
start = datetime.strptime("1/1/2015","%m/%d/%Y")
end = datetime.strptime("1/31/2015","%m/%d/%Y")

for sample_month in range(12):
    temp_df_green = NycTlcGreen(start + relativedelta(months=sample_month),
      end + relativedelta(months=sample_month)) \
      .to_pandas_dataframe()
    green_taxi_df = green_taxi_df.append(temp_df_green.sample(2000))

green_taxi_df.to_csv(os.path.dirname(os.path.abspath(__file__))
  + 'nyc_green_taxi.csv')

green_taxi_df.head(10)

