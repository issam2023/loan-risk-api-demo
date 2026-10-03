from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from app import service
from app.schemas import BatchResponse, LoanApplication, RiskResponse

API_VERSION = "1.0"

app = FastAPI(
    title="Maple Ridge Credit Union Loan Risk API",
    version=API_VERSION,
    description=(
        "Prediction API for estimating loan default risk. "
        "The service returns a default probability and a high-risk or low-risk recommendation."
    ),
)

service.load_artifacts()


@app.get("/", include_in_schema=False)
def web_interface():
    return FileResponse("app/static/index.html")


@app.get(
    "/health",
    tags=["Service"],
    summary="Check service health",
)
def health():
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail=service.NOT_READY_REASON,
        )

    return {
        "status": "ok",
        **service.status(),
    }


@app.get(
    "/version",
    tags=["Service"],
    summary="Show API and model version information",
)
def version():
    return {
        "api_version": API_VERSION,
        **service.status(),
    }


#   one INFO log line with request_id, verdict, probability, threshold, model version and NO input value;
#   response_model=RiskResponse
@app.post(
    "/predict",
    response_model=RiskResponse,
    tags=["Prediction"],
    summary="Predict default risk for one application",
)
def predict(
    application: LoanApplication,
    threshold: float | None = Query(default=None, gt=0, lt=1),
):
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail=service.NOT_READY_REASON,
        )

    request_id = service.new_request_id()

    threshold_used, result = service.predict_one(
        application.model_dump(),
        threshold=threshold,
    )

    service.logger.info(
        "prediction request_id=%s verdict=%s probability=%.3f threshold=%.6f version=%s",
        request_id,
        result["default_prediction"],
        result["default_probability"],
        threshold_used,
        service.METADATA["version"],
    )

    return {
        "request_id": request_id,
        **result,
        "model_version": service.METADATA["version"],
        "threshold_used": threshold_used,
    }


#   one predict_proba call through service.predict_many; one request_id for the batch; response_model=BatchResponse
@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["Prediction"],
    summary="Predict default risk for a batch of applications",
)
def predict_batch(
    applications: list[LoanApplication],
    threshold: float | None = Query(default=None, gt=0, lt=1),
):
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail=service.NOT_READY_REASON,
        )

    if len(applications) == 0:
        raise HTTPException(
            status_code=422,
            detail="batch must contain at least one application",
        )

    if len(applications) > service.MAX_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"batch exceeds MAX_BATCH={service.MAX_BATCH}",
        )

    request_id = service.new_request_id()

    threshold_used, predictions = service.predict_many(
        [application.model_dump() for application in applications],
        threshold=threshold,
    )

    return {
        "request_id": request_id,
        "count": len(predictions),
        "model_version": service.METADATA["version"],
        "threshold_used": threshold_used,
        "predictions": predictions,
    }


