# Azure Automated Machine Learning and Serverless

Sample code for automated machine learning (automl) and serverless model deployment.

**Note**: At the time of writing, there is an outstanding gap in Azure Function Python where some of the system dependencies are missing. Until the issue is resolved, it is expected that this sample code would not work on Azure Function. For more details, see [Gotcha](#Gotcha) section

## Pre-requisite

- Python 3
- [Az CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)
- [Azure Functions Core Tools](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local#install-the-azure-functions-core-tools)

## Getting Started

- Clone the repository

- Provision the following Azure resources through Azure Portal:
  - All resources should be deployed to the same region
  - Resource group: `azure-automl-srvless-rg`
    - Azure Machine Learning Workspace: `azure-automl-srvless-aml-ws`
    - Azure Function: `azure-automl-srvless-function`
      - Publish: Code
      - Runtime stack: Python

- Install Python dependencies with `pip install -r requirements.txt`

- [Green NYC taxi trip](https://azure.microsoft.com/en-us/services/open-datasets/catalog/nyc-taxi-limousine-commission-green-taxi-trip-records/) sample data is included in `./data/nyc_green_taxi_sample.csv`
  - \[Optional\] To reproduce the training data used in this exercise, run `python ./data/data.py` (this will take some time to download the NYC taxi dataset using [azureml.opendatasets](https://docs.microsoft.com/en-us/python/api/azureml-opendatasets/azureml.opendatasets?view=azure-ml-py))

- Sample taxi cost regression model is included in `./training/outputs/model_sample.pkl`
  - \[Optional\] To reproduce the model used in this exercise, run `python ./training/automl_train.py`
    - Note: you will have to enter your Azure subscription on line 81 before running `automl_train.py`. You may have to change the Azure resources name on line 82 - 84 depending on your setup
    - **Warning**: if you followed the optional instruction above to reproduce the training data, you may need to change the filename on line 12 of `./training/automl_train.py`

- Use Az CLI to authenticate to Azure
  - Run `az login` to login
  - Run `az account set --subscription "{azure-subscription-id}"` to select Azure subscription context where Azure resources are deployed to

- The source code and configuration for Azure Function is included in `./function`:
  - Change directory to `./function` (`cd ./function`)
  - **Note**: if you have created custom model in `pkl` format, copy the custom model to current directory with `cp {path-to-custom-model} model.pkl`
  - To run local instance of Azure Function for development and testing purposes, run `func start`
    - Postman and cURL can be used for testing local instance of Azure Function
  - To deploy Azure Function for prediction, run `func azure functionapp publish {azure-function-name} --build remote`

- If you used the provided model when deploying to Azure Function, you should see similar results as below:
  - At the time of writing, there is an outstanding gap in Azure Function Python where some of the system dependencies are missing. Depending on when the issue is resolved, you may receive `500 Internal Server Error` from executing the below request. For more details, see [Gotcha](#Gotcha) section

```bash
curl -d '{"features": [[659211, 2, 1, 0.88, 10, 28, 2, 0], [959797, 2, 1, 3.42, 11, 8, 6, 9]]}' \
-H "Content-Type: application/json" \
-X POST \
http://localhost:7071/api/predict

< HTTP/1.1 200 OK
< Content-Type: application/jso
[{"feature": [659211.0, 2.0, 1.0, 0.88, 10.0, 28.0, 2.0, 0.0], "prediction": 7.2704150529543305}, {"feature": [959797.0, 2.0, 1.0, 3.42, 11.0, 8.0, 6.0, 9.0], "prediction": 16.984844216957228}]
```

- **Workaround**: deploy Docker container with automl model on Azure Function
  - Dockerfile is included in function project directory
  - For instructions on deploying custom container to Azure Function, please see [this](https://docs.microsoft.com/en-us/azure/azure-functions/functions-create-function-linux-custom-image?tabs=python)
  - Caveat: Deploying custom container to Azure Function requires Premium plan

## Next Steps

- Infrastructure as Code (ARM)
- Script initialization and setup
- Implement custom container workaround
- Document custom container workaround

## Gotcha

- At the time of writing (10/30/2019), Azure Machine Learning service is not available on Terraform Azure provider.
- At the time of writing (10/30/2019), one cannot name the automl Python script as `automl.py` as the name introduces circular import (discussed in this [GitHub issue](https://github.com/Azure/MachineLearningNotebooks/issues/327))
- At the time of writing (10/30/2019), there are dependency issues on Azure Function Python runtime that is being tracked by [GitHub issue](https://github.com/Azure/azure-functions-python-worker/issues/493). Current workaround is to use Docker container.
- At the time of writing (10/30/2019), deploying containers on Azure Function requires Premium Function plan. Also, the Portal does not support deployed containers on Azure Function. For example, there is no ways to obtain function/host keys for accessing container-deployed functions.

---

### PLEASE NOTE FOR THE ENTIRETY OF THIS REPOSITORY AND ALL ASSETS

1. No warranties or guarantees are made or implied.
2. All assets here are provided by me "as is". Use at your own risk. Validate before use.
3. I am not representing my employer with these assets, and my employer assumes no liability whatsoever, and will not provide support, for any use of these assets.
4. Use of the assets in this repo in your Azure environment may or will incur Azure usage and charges. You are completely responsible for monitoring and managing your Azure usage.

---

Unless otherwise noted, all assets here are authored by me. Feel free to examine, learn from, comment, and re-use (subject to the above) as needed and without intellectual property restrictions.

If anything here helps you, attribution and/or a quick note is much appreciated.