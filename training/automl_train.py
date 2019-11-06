#%%
# Import the necessary packages. The Open Datasets package contains
# a class representing each data source (NycTlcGreen for example)
# to easily filter date parameters before downloading
import logging
import os
import pandas as pd
from sklearn.externals import joblib

#%%
# Import NycTlcGreen dataset
csv_path = os.path.dirname(os.path.abspath(__file__)) + '/../data/nyc_green_taxi_sample.csv'
green_taxi_df = pd.read_csv(csv_path, parse_dates=['lpepDropoffDatetime', 'lpepPickupDatetime'])

green_taxi_df.head(10)

#%%
# Define a function to create various time-based features from the
# pickup datetime field
def build_time_features(vector):
    pickup_datetime = vector[0]
    month_num = pickup_datetime.month
    day_of_month = pickup_datetime.day
    day_of_week = pickup_datetime.weekday()
    hour_of_day = pickup_datetime.hour

    return pd.Series((month_num, day_of_month, day_of_week, hour_of_day))

green_taxi_df[["month_num", "day_of_month","day_of_week", "hour_of_day"]] = \
  green_taxi_df[["lpepPickupDatetime"]].apply(build_time_features, axis=1)

green_taxi_df.head(10)

#%%
# Remove some of the columns that you won't need for training or
# additional feature building.
columns_to_remove = ["lpepPickupDatetime", "lpepDropoffDatetime", "puLocationId", "doLocationId", "extra", "mtaTax",
                     "improvementSurcharge", "tollsAmount", "ehailFee", "tripType", "rateCodeID",
                     "storeAndFwdFlag", "paymentType", "fareAmount", "tipAmount"
                    ]

for col in columns_to_remove:
    green_taxi_df.pop(col)

green_taxi_df.head(5)

#%%
# First filter the lat/long fields to be within the bounds of the
# Manhattan area. This will filter out longer taxi trips or trips
# that are outliers in respect to their relationship with other
# features.
#
# Additionally filter the tripDistance field to be greater than
# zero but less than 31 miles (the haversine distance between the
# two lat/long pairs). This eliminates long outlier trips that have
# inconsistent trip cost.
#
# Lastly, the totalAmount field has negative values for the taxi
# fares, which don't make sense in the context of our model, and
# the passengerCount field has bad data with the minimum values
# being zero.
#
# Filter out these anomalies using query functions, and then remove
# the last few columns unnecessary for training
final_df = green_taxi_df.query("pickupLatitude>=40.53 and pickupLatitude<=40.88")
final_df = final_df.query("pickupLongitude>=-74.09 and pickupLongitude<=-73.72")
final_df = final_df.query("tripDistance>=0.25 and tripDistance<31")
final_df = final_df.query("passengerCount>0 and totalAmount>0")

columns_to_remove_for_training = ["pickupLongitude", "pickupLatitude", "dropoffLongitude", "dropoffLatitude"]
for col in columns_to_remove_for_training:
    final_df.pop(col)

#%%
# Create a workspace object from the existing workspace.
# A Workspace is a class that accepts your Azure subscription and
# resource information. It also creates a cloud resource to monitor
# and track your model runs.
from azureml.core.workspace import Workspace

SUBSCRIPTION_ID = '<azure-subscription-id>' # TODO
RESOURCE_GROUP = 'azure-automl-srvless-rg'
WORKSPACE_NAME = 'azure-automl-srvless-aml-ws'
WORKSPACE_REGION = 'eastus'

try:
  ws = Workspace.from_config()
  print('Workspace configuration succeeded. Skipping the workspace creation steps.')
except:
  print('Workspace not accessible. Creating a new workspace.')
  ws = Workspace.create(name = WORKSPACE_NAME,
                        subscription_id = SUBSCRIPTION_ID,
                        resource_group = RESOURCE_GROUP,
                        location = WORKSPACE_REGION,
                        create_resource_group = True,
                        exist_ok = True)
  ws.write_config()

#%%
# Split the data into training and test sets by using the
# train_test_split function in the scikit-learn library.
from sklearn.model_selection import train_test_split

y_df = final_df.pop("totalAmount")
x_df = final_df

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=223)

#%%
# Define the experiment parameter,  model settings, and configuration
# for automl training.
from azureml.train.automl import AutoMLConfig

automl_settings = {
    "iteration_timeout_minutes": 2,
    "iterations": 20,
    "primary_metric": 'spearman_correlation',
    "preprocess": True,
    "verbosity": logging.INFO,
    "n_cross_validations": 5
}

automl_config = AutoMLConfig(task='regression',
                             debug_log='automated_ml_errors.log',
                             X=x_train.values,
                             y=y_train.values.flatten(),
                             **automl_settings)

#%%
# Create an experiment object in your workspace. An experiment acts
# as a container for your individual runs. Pass the defined
# automl_config object to the experiment, and set the output to
# True to view progress during the run.
from azureml.core.experiment import Experiment
experiment = Experiment(ws, "taxi-experiment")
local_run = experiment.submit(automl_config, show_output=True)

#%%
# Select the best model from your iterations.
best_run, fitted_model = local_run.get_output()

print('Best run:')
print(best_run)
print()
print('Fitted model:')
print(fitted_model)
print()

#%%
# Use the best model to run predictions on the test data set to
# predict taxi fares.
y_predict = fitted_model.predict(x_test.values)
print(y_predict[:10])

#%%
# Calculate the root mean squared error of the results.
from sklearn.metrics import mean_squared_error
from math import sqrt

y_actual = y_test.values.flatten().tolist()
rmse = sqrt(mean_squared_error(y_actual, y_predict))

print("Model RMSE:")
print(rmse)
print()

#%%
# Calculate mean absolute percent error (MAPE) by using the full
# y_actual and y_predict data sets.
sum_actuals = sum_errors = 0

for actual_val, predict_val in zip(y_actual, y_predict):
    abs_error = actual_val - predict_val
    if abs_error < 0:
        abs_error = abs_error * -1

    sum_errors = sum_errors + abs_error
    sum_actuals = sum_actuals + actual_val
mean_abs_percent_error = sum_errors / sum_actuals

print("Model MAPE:")
print(mean_abs_percent_error)
print()
print("Model Accuracy:")
print(1 - mean_abs_percent_error)

#%%
# Download best model to `./outputs`
joblib.dump(value=fitted_model, filename='./outputs/automl_model.pkl')

#%%
# Register best model to Azure Machine Learning model registry for
# tracking purposes. Registering model to AML model registry can
# also be used as a trigger to MLOps pipeline.
model = best_run.register_model(model_name='automl_model.pkl',
                                description = "Taxi cost regression model generated by automl")

print(f'Model Name: {model.name}')
print()
print(f'Model Id: {model.id}')
print()
print(f'Model version: {model.version}')
print()
