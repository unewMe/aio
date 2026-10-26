from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class InspectionBase(BaseModel):
    model_name: str = Field("resnet18", description="Model used for inspection")
    threshold: float = Field(0.197, ge=0.01, le=0.99, description="Decision threshold tau")


class InspectionCreate(InspectionBase):
    filename: str
    sample_category: Optional[str] = None
    notes: Optional[str] = None


class InspectionResponse(BaseModel):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    filename: str
    model_name: str
    threshold: float
    prob_defect: float
    prob_ok: float
    is_defective: bool
    verdict: str  # "DEFECT" | "OK"
    recommendation: str
    latency_ms: float
    target_met_le_100ms: bool
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class InspectionHistoryItem(BaseModel):
    id: int
    created_at: datetime
    filename: str
    model_name: str
    threshold: float
    prob_defect: float
    prob_ok: float
    is_defective: bool
    verdict: str
    recommendation: str
    latency_ms: float

    class Config:
        from_attributes = True


class InspectionStats(BaseModel):
    total_inspections: int
    total_defects: int
    total_approved: int
    defect_rate_percent: float
    average_latency_ms: float
