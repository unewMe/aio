from typing import Optional, List
from pydantic import BaseModel, Field


class SimulationStepRequest(BaseModel):
    model_name: str = Field("resnet18", description="Nazwa modelu")
    threshold: float = Field(0.197, ge=0.01, le=0.99, description="Próg decyzyjny tau")
    item_id: Optional[int] = Field(None, description="Opcjonalne ID próbki ze zbioru testowego. Brak = losowa próbka")


class SimulationStepResponse(BaseModel):
    item_id: int
    filename: str
    filepath: str
    model_name: str
    threshold: float
    prob_defect: float
    prob_ok: float
    is_defective: bool
    verdict: str  # "DEFECT" | "OK"
    recommendation: str
    latency_ms: float
    ground_truth: str  # "def" | "ok"
    ground_truth_label: int  # 1 | 0
    ground_truth_name: str  # "WADLIWY" | "POPRAWNY"
    is_correct: bool  # True jeśli werdykt zgadza się ze stanem faktycznym
    classification_type: str  # "TP" | "TN" | "FP" | "FN"
    classification_badge: str
    image_url: str
    image_base64: Optional[str] = None
    raw_latency_ms: Optional[float] = None
    cpu_config: Optional[dict] = None


class SimulationItem(BaseModel):
    id: int
    filename: str
    ground_truth: str
    ground_truth_label: int
    ground_truth_name: str


class SimulationItemsResponse(BaseModel):
    total: int
    items: List[SimulationItem]
