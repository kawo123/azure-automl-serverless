"""
Bootstrap saved model and predict cost
"""

import logging
import os
import numpy as np
import azureml.train.automl
from datetime import datetime
from sklearn.externals import joblib

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
model_path = os.path.join(script_dir, 'model.pkl')

model = None

def _initialize():
  global model
  model = joblib.load(model_path)
  _log_msg('Model is initialized.')

def _log_msg(msg):
  logging.info(f'{datetime.now()}: {msg}')

def predict_cost(features: np.ndarray):
  _initialize()

  try:
    result = model.predict(features)
    zipped = list(zip(features.tolist(), result))
    # Returns [{'feature': feature1, 'prediction': prediction1}, {'feature': feature2, 'prediction': prediction2}]
    return list(map(lambda x: { 'feature': x[0], 'prediction': x[1] } , zipped))

  except Exception as e:
    error = str(e)
    _log_msg(f'Error encountered during prediction. Error: {error}')
    return [error]

def main():
  _initialize()

  features_sample = np.array([[659211, 2, 1, 0.88, 10, 28, 2, 0],
                              [959797, 2, 1, 3.42, 11, 8, 6, 9]])

  results = predict_cost(features_sample)

  print('Results:')
  print(results)
  print()

if __name__ == "__main__":
  main()