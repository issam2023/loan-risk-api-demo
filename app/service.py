import json
import logging
import os
import uuid
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("loan-risk-api")

REQUIRED_METADATA = ("version", "features", "operating_threshold")


def normalize_employment(value):
    if pd.isna(value):
        return None

    cleaned = str(value).strip().lower().replace("-", " ")
    cleaned = " ".join(cleaned.split())

    mapping = {
        "employed": "employed",
        "self employed": "self-employed",
        "retired": "retired",
        "unemployed": "unemployed",
    }

    return mapping.get(cleaned)


def parse_percent(value):
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return None

        if value.endswith("%"):
            value = value[:-1].strip()

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_env(path=".env"):
    if not Path(path).exists():
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


load_env()

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/loan_default_model_v1.joblib"))
METADATA_PATH = Path(os.environ.get("METADATA_PATH", "models/loan_default_model_v1.json"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "500"))

model = None
METADATA = {}
THRESHOLD = None
FEATURES = []
NOT_READY_REASON = "artifacts not loaded yet"


def verify_artifacts(candidate, metadata):
    #   a required metadata key is missing, or metadata["features"] differs from candidate.feature_names_in_

    for key in REQUIRED_METADATA:
        if key not in metadata:
            return f"required metadata key missing: {key}"

    if not hasattr(candidate, "feature_names_in_"):
        return "model does not contain feature_names_in_"

    model_features = list(candidate.feature_names_in_)

    if metadata["features"] != model_features:
        return "metadata features differ from model feature_names_in_"

    return None


def load_artifacts():
    #   log an ERROR, set NOT_READY_REASON, leave model as None, and RETURN (never raise);
    #   on success set model, METADATA, THRESHOLD, FEATURES, NOT_READY_REASON = None and log one INFO line

    global model, METADATA, THRESHOLD, FEATURES, NOT_READY_REASON

    model = None
    METADATA = {}
    THRESHOLD = None
    FEATURES = []

    try:
        if not MODEL_PATH.exists():
            NOT_READY_REASON = f"model artifact missing: {MODEL_PATH}"
            logger.error(NOT_READY_REASON)
            return

        if not METADATA_PATH.exists():
            NOT_READY_REASON = f"metadata artifact missing: {METADATA_PATH}"
            logger.error(NOT_READY_REASON)
            return

        candidate = joblib.load(MODEL_PATH)

        with open(METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)

        problem = verify_artifacts(candidate, metadata)

        if problem is not None:
            NOT_READY_REASON = problem
            logger.error(NOT_READY_REASON)
            return

        model = candidate
        METADATA = metadata
        THRESHOLD = float(metadata["operating_threshold"])
        FEATURES = list(metadata["features"])
        NOT_READY_REASON = None

        logger.info(
            "loaded model version=%s threshold=%.6f",
            METADATA["version"],
            THRESHOLD,
        )

    except Exception as exc:
        NOT_READY_REASON = f"artifact loading failed: {exc}"
        model = None
        logger.error(NOT_READY_REASON)
        return


def is_ready():
    return model is not None


def status():
    return {
        "model_loaded": is_ready(),
        "model_version": METADATA.get("version"),
        "operating_threshold": THRESHOLD,
        "not_ready_reason": NOT_READY_REASON,
    }


def new_request_id():
    return uuid.uuid4().hex


def _frame(records):
    frame = pd.DataFrame(records)

    if "employment_status" in frame.columns:
        frame["employment_status"] = frame["employment_status"].apply(normalize_employment)

    if "debt_to_income_pct" in frame.columns:
        frame["debt_to_income_pct"] = frame["debt_to_income_pct"].apply(parse_percent)

    return frame.reindex(columns=FEATURES)


def predict_one(application_dict, threshold=None):
    threshold_used = THRESHOLD if threshold is None else float(threshold)

    frame = _frame([application_dict])
    probability = float(model.predict_proba(frame)[:, 1][0])

    verdict = "high-risk" if probability >= threshold_used else "low-risk"

    return threshold_used, {
        "default_prediction": verdict,
        "default_probability": round(probability, 3),
    }


def predict_many(application_dicts, threshold=None):
    threshold_used = THRESHOLD if threshold is None else float(threshold)

    frame = _frame(application_dicts)
    probabilities = model.predict_proba(frame)[:, 1]

    predictions = []

    for probability in probabilities:
        probability = float(probability)
        verdict = "high-risk" if probability >= threshold_used else "low-risk"

        predictions.append({
            "default_prediction": verdict,
            "default_probability": round(probability, 3),
        })

    return threshold_used, predictions
