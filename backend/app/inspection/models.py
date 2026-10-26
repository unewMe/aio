from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from app.database import Base


class InspectionRecord(Base):
    """
    SQLAlchemy model for persisting quality control inspection audits.
    Stores telemetry, AI verdict, applied decision thresholds, and recommendations.
    """
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=True)
    
    # Model parameters
    model_name = Column(String(100), nullable=False, index=True)
    threshold = Column(Float, nullable=False)
    
    # Predictions & Telemetry
    prob_defect = Column(Float, nullable=False)
    prob_ok = Column(Float, nullable=False)
    is_defective = Column(Boolean, nullable=False, index=True)
    verdict = Column(String(50), nullable=False)  # "DEFECT" | "OK"
    recommendation = Column(String(255), nullable=False)
    latency_ms = Column(Float, nullable=False)
    
    # Audit notes
    notes = Column(Text, nullable=True)
