import logging

import pytest
from fastapi.testclient import TestClient

from app import service
from app.main import app

client = TestClient(app)


@pytest.fixture
def safe_payload():
    return {
        "employment_status": "employed",
        "annual_income": 65432.0,
        "credit_score": 712.0,
        "debt_to_income_pct": 27.3,
        "home_ownership": "mortgage",
        "previous_defaults": 0,
    }


@pytest.fixture
def risky_payload():
    return {
        "employment_status": "unemployed",
        "annual_income": 23456.0,
        "credit_score": 510.0,
        "debt_to_income_pct": 63.7,
        "home_ownership": "rent",
        "previous_defaults": 2,
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200


def test_predict_happy_path_has_request_id(safe_payload):
    response = client.post("/predict", json=safe_payload)

    assert response.status_code == 200

    body = response.json()

    assert body["request_id"]
    assert body["default_prediction"] in {"high-risk", "low-risk"}
    assert 0 <= body["default_probability"] <= 1
    assert body["model_version"] == service.METADATA["version"]
    assert body["threshold_used"] == pytest.approx(service.THRESHOLD)


def test_predict_batch_happy_path_has_request_id(
    safe_payload,
    risky_payload,
):
    response = client.post(
        "/predict/batch",
        json=[safe_payload, risky_payload],
    )

    assert response.status_code == 200

    body = response.json()

    assert body["request_id"]
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_contract_break_returns_422(safe_payload):
    bad_payload = dict(safe_payload)
    bad_payload["credit_score"] = 900

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_extra_field_returns_422(safe_payload):
    bad_payload = dict(safe_payload)
    bad_payload["unexpected_field"] = "not allowed"

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_empty_batch_returns_422():
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 422


def test_batch_over_max_returns_413(monkeypatch, safe_payload):
    monkeypatch.setattr(service, "MAX_BATCH", 1)

    response = client.post(
        "/predict/batch",
        json=[safe_payload, safe_payload],
    )

    assert response.status_code == 413


def test_missing_model_returns_503(monkeypatch, safe_payload):
    monkeypatch.setattr(service, "model", None)
    monkeypatch.setattr(
        service,
        "NOT_READY_REASON",
        "model unavailable for test",
    )

    response = client.post("/predict", json=safe_payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "model unavailable for test"


def test_invalid_threshold_returns_422(safe_payload):
    response = client.post(
        "/predict?threshold=1",
        json=safe_payload,
    )

    assert response.status_code == 422


def test_prediction_log_privacy(caplog, safe_payload):
    with caplog.at_level(logging.INFO, logger="loan-risk-api"):
        response = client.post("/predict", json=safe_payload)

    assert response.status_code == 200

    request_id = response.json()["request_id"]

    prediction_logs = [
        record.getMessage()
        for record in caplog.records
        if record.name == "loan-risk-api"
        and "prediction request_id=" in record.getMessage()
    ]

    assert len(prediction_logs) == 1

    log_line = prediction_logs[0]

    # Required traceability.
    assert request_id in log_line

    # Privacy: applicant input values must not appear in the log.
    assert "65432" not in log_line
    assert "712" not in log_line
    assert "27.3" not in log_line
    assert "mortgage" not in log_line
    assert "employed" not in log_line
