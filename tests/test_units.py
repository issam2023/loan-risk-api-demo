import json

import joblib
import pytest

from app import service


# Invented applicants only — never copied from the client file.

@pytest.fixture
def safe_applicant():
    return {
        "employment_status": "employed",
        "annual_income": 65432.0,
        "credit_score": 712.0,
        "debt_to_income_pct": 27.3,
        "home_ownership": "mortgage",
        "previous_defaults": 0,
    }


@pytest.fixture
def risky_applicant():
    return {
        "employment_status": "unemployed",
        "annual_income": 23456.0,
        "credit_score": 510.0,
        "debt_to_income_pct": 63.7,
        "home_ownership": "rent",
        "previous_defaults": 2,
    }


@pytest.fixture
def borderline_applicant():
    return {
        "employment_status": "self-employed",
        "annual_income": 48765.0,
        "credit_score": 640.0,
        "debt_to_income_pct": 39.4,
        "home_ownership": "rent",
        "previous_defaults": 0,
    }


def test_normalize_employment_every_export_spelling():
    cases = {
        "employed": "employed",
        "Employed": "employed",
        "EMPLOYED": "employed",
        " employed ": "employed",
        "self-employed": "self-employed",
        "Self-Employed": "self-employed",
        "self employed": "self-employed",
        "Self employed": "self-employed",
        "retired": "retired",
        "Retired": "retired",
        "RETIRED": "retired",
        "unemployed": "unemployed",
        "Unemployed": "unemployed",
        " unemployed": "unemployed",
    }

    for raw, expected in cases.items():
        assert service.normalize_employment(raw) == expected


def test_normalize_employment_unknown_returns_none():
    assert service.normalize_employment("student") is None


def test_parse_percent_number():
    assert service.parse_percent(28.7) == pytest.approx(28.7)


def test_parse_percent_string_without_percent():
    assert service.parse_percent("28.7") == pytest.approx(28.7)


def test_parse_percent_string_with_percent():
    assert service.parse_percent("28.7%") == pytest.approx(28.7)


def test_parse_percent_empty_and_missing():
    assert service.parse_percent("") is None
    assert service.parse_percent(None) is None


def test_predict_one_response_shape(safe_applicant):
    threshold_used, result = service.predict_one(safe_applicant)

    assert threshold_used == pytest.approx(service.THRESHOLD)
    assert set(result) == {"default_prediction", "default_probability"}
    assert result["default_prediction"] in {"high-risk", "low-risk"}
    assert isinstance(result["default_probability"], float)
    assert 0 <= result["default_probability"] <= 1


def test_predict_one_verdict_at_operating_threshold(safe_applicant):
    threshold_used, result = service.predict_one(safe_applicant)

    expected = (
        "high-risk"
        if result["default_probability"] >= round(threshold_used, 3)
        else "low-risk"
    )

    assert result["default_prediction"] == expected


def test_predict_one_supplied_threshold(safe_applicant):
    threshold_used, result = service.predict_one(
        safe_applicant,
        threshold=0.01,
    )

    assert threshold_used == pytest.approx(0.01)
    assert result["default_prediction"] == "high-risk"


def test_verify_artifacts_accepts_shipped_pair():
    candidate = joblib.load("models/loan_default_model_v1.joblib")

    with open("models/loan_default_model_v1.json", encoding="utf-8") as f:
        metadata = json.load(f)

    assert service.verify_artifacts(candidate, metadata) is None


def test_verify_artifacts_rejects_feature_mismatch():
    candidate = joblib.load("models/loan_default_model_v1.joblib")

    with open("models/loan_default_model_v1.json", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["features"] = ["wrong_feature"]

    problem = service.verify_artifacts(candidate, metadata)

    assert problem is not None
    assert "features" in problem


def test_verify_artifacts_rejects_missing_required_key():
    candidate = joblib.load("models/loan_default_model_v1.joblib")

    with open("models/loan_default_model_v1.json", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata.pop("operating_threshold")

    problem = service.verify_artifacts(candidate, metadata)

    assert problem is not None
    assert "operating_threshold" in problem
