# Operations Runbook — Maple Ridge Loan-Risk API

Every command runs from the project root. Under each heading, paste the exact command and only the output lines that matter (the image id, the health answer, the scored response, the log line, the 413).

## 1. Build the image

```bash
docker build -t loan-risk-api:v1 .
docker image inspect loan-risk-api:v1 --format '{{.Id}}'
```

Output:

```text
sha256:1cd5e3e37f4636926a0e0a37f49e5178efd15b95b1ec36f0007c62d094210e99
```

## 2. Run with configuration from a file; health check; one scored application

Local `.env` used for the normal run:

```text
MODEL_PATH=models/loan_default_model_v1.joblib
METADATA_PATH=models/loan_default_model_v1.json
MAX_BATCH=500
```

Run:

```bash
docker run --rm --name loan-risk-api \
  --env-file .env \
  -p 8000:8000 \
  loan-risk-api:v1
```

Health check from another terminal:

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Output:

```json
{
    "status": "ok",
    "model_loaded": true,
    "model_version": "1",
    "operating_threshold": 0.24638885712564745,
    "not_ready_reason": null
}
```

Score one invented application:

```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "employment_status": "employed",
    "annual_income": 65000,
    "credit_score": 710,
    "debt_to_income_pct": 28.7,
    "home_ownership": "mortgage",
    "previous_defaults": 0
  }' | python -m json.tool
```

Output:

```json
{
    "request_id": "bc7091c5fb2a4e9c815cee98e6e142a3",
    "default_prediction": "low-risk",
    "default_probability": 0.031,
    "model_version": "1",
    "threshold_used": 0.24638885712564745
}
```

## 3. Read the logs from outside the container; find one decision by its request id

```bash
docker logs loan-risk-api 2>&1 | \
grep 'bc7091c5fb2a4e9c815cee98e6e142a3'
```

Output:

```text
2026-10-01 21:50:11,437 INFO loan-risk-api prediction request_id=bc7091c5fb2a4e9c815cee98e6e142a3 verdict=low-risk probability=0.031 threshold=0.246389 version=1
```

The log contains the request id, verdict, probability, threshold and model version. Applicant input values are not logged.

## 4. Change MAX_BATCH by configuration alone (no rebuild) and prove it

Image id before the configuration change:

```bash
docker image inspect loan-risk-api:v1 \
  --format 'IMAGE ID BEFORE = {{.Id}}'
```

Output:

```text
IMAGE ID BEFORE = sha256:1cd5e3e37f4636926a0e0a37f49e5178efd15b95b1ec36f0007c62d094210e99
```

Change only the local `.env` from `MAX_BATCH=500` to `MAX_BATCH=1`. Do not rebuild the image.

Then run the same image:

```bash
docker run --rm --name loan-risk-api \
  --env-file .env \
  -p 8000:8000 \
  loan-risk-api:v1
```

Send a batch containing two invented applications:

```bash
curl -i -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "employment_status": "employed",
      "annual_income": 65000,
      "credit_score": 710,
      "debt_to_income_pct": 28.7,
      "home_ownership": "mortgage",
      "previous_defaults": 0
    },
    {
      "employment_status": "unemployed",
      "annual_income": 25000,
      "credit_score": 520,
      "debt_to_income_pct": 60,
      "home_ownership": "rent",
      "previous_defaults": 2
    }
  ]'
```

Output:

```text
HTTP/1.1 413 Request Entity Too Large
```

Verify the image id again:

```bash
docker image inspect loan-risk-api:v1 \
  --format 'IMAGE ID AFTER = {{.Id}}'
```

Output:

```text
IMAGE ID AFTER = sha256:1cd5e3e37f4636926a0e0a37f49e5178efd15b95b1ec36f0007c62d094210e99
```

The image id is unchanged. `MAX_BATCH` changed through runtime configuration only; the image was not rebuilt.

## 5. Stop and clean up

```bash
docker stop loan-risk-api
rm -f .env
```

Because the container was started with `--rm`, Docker removes the container after it stops. The local `.env` is also removed and is not submitted.

## Common errors

| Message | Cause | Fix |
|---|---|---|
| `lookup registry-1.docker.io: no such host` | Temporary DNS resolution problem while contacting Docker Hub | Check network/DNS and retry the build |
| `No matching distribution found for pandas==3.0.6` | pandas 3.0.6 requires Python 3.11+, but the Docker image uses Python 3.10 | Use the pinned Python-3.10-compatible runtime version |
| `No matching distribution found for numpy==2.4.6` | NumPy 2.4.6 requires Python 3.11+, but the Docker image uses Python 3.10 | Use the pinned Python-3.10-compatible runtime version |
| `port is already allocated` | Another process is already using port 8000 | Stop the local Uvicorn process before starting the container |
