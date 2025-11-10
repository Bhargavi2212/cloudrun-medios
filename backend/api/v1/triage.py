from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ...services.error_response import StandardResponse
from ...services.triage_service import TriageService

router = APIRouter(prefix="/triage", tags=["triage"])
service = TriageService()


class TriagePredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of feature_name -> value matching the triage model training columns.",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    model: Optional[str] = Field(
        default=None,
        description="Optional model override. Supported values: lightgbm, xgboost, stacking.",
    )
    use_shap: bool = Field(
        default=False,
        description="Whether to use SHAP for explainability (requires shap library).",
    )


@router.post("/predict", response_model=StandardResponse)
async def predict_triage(payload: TriagePredictRequest) -> StandardResponse:
    if not payload.features:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="features payload must include at least one feature.",
        )

    try:
        prediction = service.predict(
            payload.features,
            top_k=payload.top_k,
            model=payload.model,
            use_shap=payload.use_shap,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return StandardResponse(
        success=True,
        data={
            "severity_index": prediction.severity_index,
            "severity_label": prediction.severity_label,
            "probabilities": prediction.probabilities,
            "explanation": prediction.explanation,
            "model_used": prediction.model_used,
            "latency_ms": prediction.latency_ms,
        },
        is_stub=False,
    )


@router.post("/explain", response_model=StandardResponse)
async def explain_triage_prediction(payload: TriagePredictRequest) -> StandardResponse:
    """Explain a triage prediction using SHAP values.

    This endpoint provides detailed explanations for why a particular
    severity level was predicted, including feature contributions.
    """
    if not payload.features:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="features payload must include at least one feature.",
        )

    try:
        explanation = service.explain_prediction(
            payload.features,
            model=payload.model,
            top_k=payload.top_k,
            use_shap=(
                payload.use_shap if payload.use_shap else True
            ),  # Default to SHAP for explain endpoint
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return StandardResponse(
        success=True,
        data=explanation,
        is_stub=False,
    )


@router.get("/metadata", response_model=StandardResponse)
async def triage_metadata() -> StandardResponse:
    return StandardResponse(
        success=True,
        data={
            "feature_names": service.feature_names,
            "severity_labels": service.severity_labels,
            "default_model": service.default_model,
            "supported_models": service.supported_models,
        },
    )
