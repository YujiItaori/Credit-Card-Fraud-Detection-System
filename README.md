# Credit Card Fraud Detection System

An end-to-end credit-card fraud-detection system with supervised fraud
classification, Isolation Forest anomaly detection, a FastAPI inference
service, and an interactive Streamlit dashboard.

The system uses a saved operational decision threshold of **0.38**, loaded
from model metadata, to prioritize recall for fraud detection while
maintaining a practical precision level.

## Project overview

The application is organized into three main layers:

1. **Model and prediction layer**
   - Loads trained supervised and Isolation Forest model artifacts.
   - Prepares features consistently for inference.
   - Applies the saved supervised decision threshold.
   - Produces fraud scores, binary predictions, and review flags.

2. **FastAPI service layer**
   - Exposes prediction, anomaly-detection, health, model-information, and
     evaluation-report endpoints.
   - Validates incoming transaction data with Pydantic.
   - Keeps all model loading and inference logic server-side.
   - Returns structured JSON responses.

3. **Streamlit dashboard layer**
   - Provides manual single-transaction prediction.
   - Supports CSV batch prediction.
   - Supports transaction investigation from batch results.
   - Supports optional Isolation Forest anomaly scoring for batch requests.
   - Displays the saved model-evaluation report.
   - Communicates with FastAPI through a shared API client.

## Key features

- Supervised fraud classification using a trained pipeline.
- Isolation Forest anomaly detection.
- Operational decision threshold of 0.38, loaded from model metadata.
- Manual-review flag for transactions at or above the threshold.
- FastAPI-based inference service.
- Streamlit dashboard with Home, Single Prediction, Batch Prediction,
  Transaction Investigation, Anomaly Detection, and Model Evaluation pages.
- CSV upload and downloadable batch-prediction results.
- Optional anomaly scores and anomaly summaries for batch predictions.
- Saved model-evaluation report available through the API.
- Comprehensive validation for missing, non-numeric, infinite, negative, and
  unexpected features.
- Automated test suite covering preprocessing, prediction, API behavior,
  API-client behavior, batch ordering, and optional anomaly scoring.

## Project structure

```text
.
├── api/
│   ├── main.py
│   ├── routes.py
│   └── schemas.py
├── app/
│   ├── api_client.py
│   ├── streamlit_app.py
│   └── styles.css
├── src/
│   ├── data/
│   ├── models/
│   └── preprocessing/
├── tests/
│   ├── test_api.py
│   ├── test_api_client.py
│   ├── test_data_validation.py
│   ├── test_predictions.py
│   └── test_preprocessing.py
├── models/
├── reports/
├── .env.example
├── requirements.txt
└── README.md
```

## Installation

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Environment configuration

Copy the provided example environment file:

```powershell
Copy-Item .env.example .env
```

For local development, use:

```env
FRAUD_API_HOST=127.0.0.1
FRAUD_API_PORT=8000
FRAUD_API_RELOAD=true
FRAUD_API_BASE_URL=http://127.0.0.1:8000
FRAUD_API_TIMEOUT_SECONDS=30
MODEL_ARTIFACTS_DIR=models
```

Do not commit `.env` to version control.

## Start the FastAPI service

From the project root, with the virtual environment activated:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Start the Streamlit dashboard

In a second terminal, with the virtual environment activated:

```powershell
streamlit run app/streamlit_app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

The Streamlit app communicates with the API using the base URL configured by:

```env
FRAUD_API_BASE_URL=http://127.0.0.1:8000
```

## API endpoints

### `GET /`

Returns basic API information.

### `GET /health`

Returns API and model-health information.

Example response:

```json
{
  "status": "healthy",
  "supervised_model_loaded": true,
  "anomaly_model_loaded": true
}
```

### `GET /model-info`

Returns metadata for the loaded supervised and anomaly models, including the
saved operational decision threshold.

### `GET /evaluation-report`

Returns the saved model-evaluation report used by the Streamlit Model
Evaluation page.

The report is read from:

```text
reports/model_evaluation_report.md
```

The endpoint does not recalculate metrics or modify model artifacts.

Example response:

```json
{
  "report": "# Supervised vs Anomaly Detection Evaluation..."
}
```

If the saved report is unavailable, the endpoint returns HTTP 404 with a
clear error message.

### `POST /predict`

Generates a supervised fraud prediction for one transaction.

Example request:

```json
{
  "Time": 0.0,
  "V1": 0.0,
  "V2": 0.0,
  "V3": 0.0,
  "V4": 0.0,
  "V5": 0.0,
  "V6": 0.0,
  "V7": 0.0,
  "V8": 0.0,
  "V9": 0.0,
  "V10": 0.0,
  "V11": 0.0,
  "V12": 0.0,
  "V13": 0.0,
  "V14": 0.0,
  "V15": 0.0,
  "V16": 0.0,
  "V17": 0.0,
  "V18": 0.0,
  "V19": 0.0,
  "V20": 0.0,
  "V21": 0.0,
  "V22": 0.0,
  "V23": 0.0,
  "V24": 0.0,
  "V25": 0.0,
  "V26": 0.0,
  "V27": 0.0,
  "V28": 0.0,
  "Amount": 100.0
}
```

Example response:

```json
{
  "fraud_score": 0.0842,
  "fraud_prediction": 0,
  "decision_threshold": 0.38,
  "review_flag": 0,
  "anomaly_score": null,
  "anomaly_prediction": null
}
```

### `POST /predict/batch`

Generates supervised fraud predictions for multiple transactions.

The request accepts an optional field:

```json
{
  "include_anomaly_scores": false
}
```

When `include_anomaly_scores` is `true`, the response also includes:

- `anomaly_count`
- `anomaly_rate`
- `anomaly_score` for each transaction
- `anomaly_prediction` for each transaction

When it is `false` or omitted, the batch response remains supervised-only and
the anomaly fields are returned as `null`.

Batch predictions are returned in the same order as the submitted
transactions.

### `POST /detect-anomaly`

Generates both supervised fraud and Isolation Forest anomaly results for one
transaction.

Example response:

```json
{
  "fraud_score": 0.4218,
  "fraud_prediction": 1,
  "decision_threshold": 0.38,
  "review_flag": 1,
  "anomaly_score": -0.6521,
  "anomaly_prediction": -1
}
```

For Isolation Forest:

- `-1` means anomaly detected.
- `1` means no anomaly detected.
- Lower anomaly scores indicate more unusual transactions.
- An anomaly score is not a fraud probability.

## Streamlit dashboard pages

### Home

Displays the application overview and checks whether the API, supervised model,
and anomaly model are available.

### Single Prediction

Allows manual entry of all 30 transaction features and returns:

- Fraud score.
- Binary fraud prediction.
- Decision threshold.
- Manual-review recommendation.

### Batch Prediction

Accepts a CSV file containing the required transaction columns and returns:

- Total transactions.
- Number flagged for review.
- Flagged rate.
- Per-row predictions.
- Downloadable CSV results.
- Fraud-score and prediction distributions.
- Filters for prediction class, review status, and fraud-score range.
- Flagged-transactions view.
- Optional anomaly count, anomaly rate, anomaly scores, and anomaly predictions.
- Optional anomaly-score distribution, most-anomalous transactions, and
  supervised-versus-anomaly comparison.

### Transaction Investigation

Allows selection of a transaction from the latest batch prediction and displays:

- Original uploaded transaction features.
- API-returned fraud score.
- Binary fraud prediction.
- Decision threshold.
- Review recommendation.

Predictions are clearly identified as model outputs, not confirmed fraud.

### Anomaly Detection

Allows manual entry of one transaction and returns both:

- Supervised fraud results.
- Isolation Forest anomaly results.

### Model Evaluation

Displays the saved supervised and anomaly-detection evaluation report returned
by `GET /evaluation-report`.

The page shows historical validation results only. It does not recalculate
metrics, retrain models, or perform inference in Streamlit.

The saved evaluation report describes validation-set results and uses a
validation decision rule of `0.5`. This is separate from the API’s operational
supervised decision threshold of `0.38`, which is loaded from model metadata.

## Required CSV columns

Batch CSV files must contain exactly these columns:

```text
Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount
```

The `Amount` column must be non-negative. Missing values, infinite values,
non-numeric values, and unexpected columns are rejected.

## Decision threshold

The supervised model uses a saved operational decision threshold of:

```text
0.38
```

A transaction is predicted as fraud when:

```text
fraud_score >= 0.38
```

The threshold is loaded from model metadata by the prediction service. The
Streamlit dashboard does not hardcode or recalculate the threshold.

The saved model-evaluation report separately documents historical validation
results and a validation decision rule of `0.5`. These are distinct concepts:
the report describes historical model evaluation, while `0.38` is the
operational threshold used by the API for live predictions.

## Monitoring scope

Monitoring is intentionally excluded from the initial hosted demo.

The expected saved monitoring reports, including data-quality, drift, and
prediction-monitoring artifacts, are not available for the hosted application.
Therefore, the Streamlit dashboard does not include a Monitoring page in this
deployment scope.

This does not affect supervised prediction, Isolation Forest anomaly detection,
transaction investigation, batch prediction, or model-evaluation functionality.

## Running tests

Run the complete test suite:

```powershell
pytest tests -v
```

Expected result:

```text
69 passed, 1 warning
```

The test suite covers:

- Data validation.
- Preprocessing pipelines.
- Supervised prediction behavior.
- Anomaly prediction behavior.
- FastAPI endpoint validation and responses.
- Streamlit API-client configuration, requests, and error handling.
- Batch prediction row ordering.
- Optional anomaly-score fields in batch responses.
- Model-evaluation report endpoint and API-client behavior.

## Development notes

- All inference occurs in the FastAPI service.
- The Streamlit app contains no model-loading or prediction logic.
- The Streamlit app uses `app/api_client.py` for all HTTP communication.
- The API base URL and request timeout are configurable through environment
  variables.
- Existing supervised-only behavior remains unchanged.
- Anomaly scoring is optional for batch predictions.
- The Model Evaluation page displays the saved validation report through
  `GET /evaluation-report`.
- Monitoring is excluded from the initial hosted demo because the required
  saved monitoring reports do not exist.
- The API operational threshold is loaded from model metadata and is separate
  from the decision rule documented in the historical validation report.#
