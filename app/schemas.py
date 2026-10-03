from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoanApplication(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "employment_status": "employed",
                "annual_income": 65000.0,
                "credit_score": 710.0,
                "debt_to_income_pct": 28.7,
                "home_ownership": "mortgage",
                "previous_defaults": 0
            }
        }
    )


    employment_status: Literal[
        "employed", "self-employed", "retired", "unemployed"
    ] = Field(
        description="Employment status at application time."
    )

    annual_income: float | None = Field(
        default=None,
        gt=0,
        description="Declared gross annual income in CAD."
    )

    credit_score: float | None = Field(
        default=None,
        ge=300,
        le=850,
        description="Credit bureau score at application time."
    )

    debt_to_income_pct: float = Field(
        ge=0,
        le=100,
        description="Debt-to-income ratio as a percentage."
    )

    home_ownership: Literal[
        "own", "mortgage", "rent"
    ] = Field(
        description="Applicant home ownership status."
    )

    previous_defaults: int = Field(
        ge=0,
        description="Number of previous defaults on the credit file."
    )


class RiskResponse(BaseModel):
    request_id: str
    default_prediction: Literal["high-risk", "low-risk"]
    default_probability: float
    model_version: int
    threshold_used: float


class BatchPrediction(BaseModel):
    default_prediction: Literal["high-risk", "low-risk"]
    default_probability: float


class BatchResponse(BaseModel):
    request_id: str
    count: int
    model_version: int
    threshold_used: float
    predictions: list[BatchPrediction]
