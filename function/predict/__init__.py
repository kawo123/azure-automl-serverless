import logging

import json
import numpy as np
import azure.functions as func
from .predict import predict_cost

def main(req: func.HttpRequest) -> func.HttpResponse:
  logging.info('Python HTTP trigger function processed a request.')

  try:
    req_body = req.get_json()
  except ValueError:
    return func.HttpResponse('Please pass features in the request body', status_code=400)
  else:
    features = req_body.get('features')

  results = predict_cost(np.array(features))

  headers = {
    "Content-type": "application/json"
  }

  return func.HttpResponse(json.dumps(results), headers = headers)
