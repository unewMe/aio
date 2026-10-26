from typing import List, Optional
from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    display_name: str
    role: str
    description: str
    optimal_threshold: float
    has_weights: bool
    weights_file: str
    parameters: int
    size_mb: float
    device: str


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


class BenchmarkRequest(BaseModel):
    model_name: str = "resnet18"
    iterations: int = 20


class BenchmarkResponse(BaseModel):
    model_name: str
    device: str
    iterations: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    target_met_le_100ms: bool


class ModelMemoryResponse(BaseModel):
    strategy: str
    max_loaded_models: int
    loaded_models: List[str]
    active_model: Optional[str] = None
    rss_memory_mb: float
    device: str

