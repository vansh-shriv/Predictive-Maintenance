"""Request / response models for the RUL API."""
from pydantic import BaseModel, Field, create_model

from pmaint.data.loader import SENSOR_COLS, SETTING_COLS

MAX_HISTORY = 1000

# One row of raw C-MAPSS telemetry: 3 operating settings + 21 sensors (same names as the dataset).
Reading = create_model(
    "Reading", **{c: (float, Field(..., allow_inf_nan=False)) for c in SETTING_COLS + SENSOR_COLS}
)


class PredictRequest(BaseModel):
    engine_id: str | None = Field(None, description="Optional id, echoed back and logged.")
    readings: list[Reading] = Field(
        ..., min_length=1, max_length=MAX_HISTORY,
        description="Raw readings in chronological order (oldest first). The model uses the "
                    "last `window` cycles; shorter histories are front-padded.",
    )


class PredictResponse(BaseModel):
    engine_id: str | None
    predicted_rul: float = Field(..., description="Predicted remaining useful life in cycles.")
    cycles_received: int
    window: int
    padded: bool = Field(..., description="True if history was shorter than the model window.")
    model_version: str
    model_run_id: str


class ModelInfo(BaseModel):
    model_name: str
    version: str
    run_id: str
    model_type: str
    window: int
    features: list[str]
