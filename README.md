# Maple Ridge Loan-Risk API

This project provides a loan-default risk scoring service for Maple Ridge Credit Union. A loan officer can submit a new application and receive a probability of default, a high-risk or low-risk recommendation, the model version, the operating threshold, and a request id for tracing the decision. The service uses a logistic-regression model packaged in a scikit-learn pipeline and exposed through FastAPI.

## Requirements

For local development, this project was tested with Python 3.11.16. The Docker image uses Python 3.10 through `python:3.10-slim`.

Required tools:

- Python 3.11
- pip
- Docker
- Postman for running the included Postman collection

The project uses free local tools. No cloud account, paid service, or payment information is required.

## Setup on a clean machine

From the project root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create the local configuration file:

```bash
cp .env.example .env
```

The default configuration is:

```text
MODEL_PATH=models/loan_default_model_v1.joblib
METADATA_PATH=models/loan_default_model_v1.json
MAX_BATCH=500
```

The real `.env` file is local configuration and must not be included in the submission ZIP.

## Reproduce the model

Run the notebooks in this order:

1. `notebooks/01_data_audit.ipynb` audits every column, documents keep/exclude decisions, checks missing values and data quality, and defines the fixed cleaning rules.
2. `notebooks/02_train_evaluate.ipynb` creates the seeded stratified splits, trains the logistic-regression pipeline, selects the operating threshold on validation data, evaluates once on the final-test set, and writes:
   - `models/loan_default_model_v1.joblib`
   - `models/loan_default_model_v1.json`
3. `notebooks/03_explainability.ipynb` loads the deployed model and explains global coefficients and a final-test prediction close to the operating threshold. It does not retrain the model.

The client CSV in `data/loan_applications.csv` must remain unchanged.

## Run the service locally

From the project root:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive FastAPI documentation:

```text
http://localhost:8000/docs
```

Check service health:

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

When the model and metadata pair load correctly, `/health` returns HTTP 200 with `"status": "ok"` and model information.

If an artifact is missing, cannot be loaded, or the metadata features do not match the fitted pipeline, the service stays up but `/health` and prediction requests return HTTP 503 with the reason. The artifact-loading failure is also written to the ERROR log.

## Run the tests

Run:

```bash
pytest -q
```

Expected result:

```text
23 passed
```

The tests cover cleaning helpers, artifact verification, health, prediction, batch prediction, validation errors, batch limits, unavailable-model behavior, threshold handling, and prediction-log privacy.

## Run the container

Build the image:

```bash
docker build -t loan-risk-api:v1 .
```

Run it using the local configuration file:

```bash
docker run --rm --name loan-risk-api \
  --env-file .env \
  -p 8000:8000 \
  loan-risk-api:v1
```

Check the containerized service:

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

The Docker image is based on `python:3.10-slim`. Runtime configuration can be changed through environment variables without rebuilding the image. See `RUN-COMMANDS.md` for the `MAX_BATCH` configuration demonstration.

## Benchmark the running service

Start the API first, then run:

```bash
python benchmark.py http://localhost:8000
```

A measured local run produced:

```json
{
  "single_median_ms": 8.6,
  "single_p95_ms": 11.8,
  "hundred_singles_ms": 779,
  "one_batch_of_100_ms": 9,
  "requests_per_second_5_clients": 141
}
```

These are local benchmark measurements and may vary by machine. They are compared with the expected workload of approximately 400 applications per month when discussing the infrastructure choice.

## Run the Postman collection

Import this file into Postman:

```text
postman/Maple-Ridge-Loan-Risk-API.postman_collection.json
```

Start the API or Docker container and run these requests:

1. `1 - Health`
2. `2 - Predict`
3. `3 - Batch Prediction`
4. `4 - Rejected Request`

The collection checks a healthy service, a valid prediction with a non-empty `request_id`, a two-application batch, and rejection of an invalid credit score with HTTP 422.

The default base URL is:

```text
http://localhost:8000
```

## Input contract

`POST /predict` accepts these model inputs:

| Field | Rule |
|---|---|
| `employment_status` | `employed`, `self-employed`, `retired`, or `unemployed` |
| `annual_income` | Positive number or `null` |
| `credit_score` | 300 to 850 or `null` |
| `debt_to_income_pct` | Numeric percentage from 0 to 100 |
| `home_ownership` | `own`, `mortgage`, or `rent` |
| `previous_defaults` | Integer greater than or equal to 0 |

Only `annual_income` and `credit_score` accept `null`, because these fields contain missing values in the source export and are handled by the fitted preprocessing pipeline.

Extra fields are forbidden. Invalid categories, invalid ranges, and unexpected fields are rejected by request validation.

A successful single prediction returns:

- `request_id`
- `default_prediction`
- `default_probability`
- `model_version`
- `threshold_used`

The `request_id` can be used to find the corresponding prediction decision in the service logs. Applicant input values and applicant identifiers are not written to the prediction log.
