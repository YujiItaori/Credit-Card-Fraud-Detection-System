"""FastAPI application for credit-card fraud detection."""

from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT

from api.routes import (
    API_DESCRIPTION,
    API_NAME,
    API_VERSION,
    router,
)
from src.models.predict import create_prediction_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPERVISED_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "supervised"
    / "final_supervised_model"
)

ANOMALY_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "anomaly"
    / "baseline_anomaly_models"
    / "isolation_forest_baseline"
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model artifacts once when the application starts."""
    logger.info("Loading fraud-detection model artifacts.")

    prediction_service = create_prediction_service(
        supervised_model_directory=SUPERVISED_MODEL_DIRECTORY,
        anomaly_model_directory=ANOMALY_MODEL_DIRECTORY,
    )

    application.state.prediction_service = prediction_service

    logger.info("Fraud-detection model artifacts loaded.")

    yield

    logger.info("Shutting down fraud-detection API.")


def sanitize_non_finite_values(value: Any) -> Any:
    """Replace non-finite float values with None for safe JSON serialization."""
    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, dict):
        return {
            key: sanitize_non_finite_values(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            sanitize_non_finite_values(item)
            for item in value
        ]

    return value


app = FastAPI(
    title=API_NAME,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request,
    exc: RequestValidationError,
):
    """Return validation errors with JSON-safe non-finite values."""
    sanitized_detail = sanitize_non_finite_values(
        exc.errors(),
    )

    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": sanitized_detail,
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)