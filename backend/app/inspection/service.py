import httpx
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi import HTTPException, UploadFile

from app.config import settings
from app.inspection.models import InspectionRecord
from app.inspection.schemas import InspectionResponse, InspectionHistoryItem, InspectionStats


async def run_inspection_from_upload(
    db: Session,
    file: UploadFile,
    model_name: str,
    threshold: float,
    notes: Optional[str] = None
) -> InspectionResponse:
    """
    Coordinates inspection workflow:
    1. Sends image bytes to independent ML service (/predict).
    2. Applies industrial decision rule tau.
    3. Persists audit record in relational database.
    """
    file_bytes = await file.read()
    filename = file.filename or "unknown.jpg"

    # Call ML Service
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.ML_SERVICE_URL}/predict",
                data={"model_name": model_name},
                files={"file": (filename, file_bytes, file.content_type or "image/jpeg")}
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Błąd serwisu ML: {response.text}"
                )
            ml_data = response.json()
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Nie można połączyć się z mikroserwisem ML ({settings.ML_SERVICE_URL}): {str(exc)}"
        )

    prob_defect = float(ml_data["prob_defect"])
    prob_ok = float(ml_data["prob_ok"])
    latency_ms = float(ml_data["latency_ms"])

    # Industrial decision rule
    is_defective = prob_defect >= threshold
    if is_defective:
        verdict = "DEFECT"
        recommendation = "ODRZUCONY • KIERUNEK: RECYKLING / PRZETOP"
    else:
        verdict = "OK"
        recommendation = "ZATWIERDZONY • KIERUNEK: MONTAŻ POMPY"

    # Persist in DB
    record = InspectionRecord(
        filename=filename,
        model_name=model_name,
        threshold=threshold,
        prob_defect=prob_defect,
        prob_ok=prob_ok,
        is_defective=is_defective,
        verdict=verdict,
        recommendation=recommendation,
        latency_ms=latency_ms,
        notes=notes
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return InspectionResponse(
        id=record.id,
        created_at=record.created_at,
        filename=filename,
        model_name=model_name,
        threshold=threshold,
        prob_defect=prob_defect,
        prob_ok=prob_ok,
        is_defective=is_defective,
        verdict=verdict,
        recommendation=recommendation,
        latency_ms=latency_ms,
        target_met_le_100ms=latency_ms <= 100.0,
        notes=notes
    )


async def run_inspection_from_sample(
    db: Session,
    category: str,
    filename: str,
    model_name: str,
    threshold: float,
    notes: Optional[str] = None
) -> InspectionResponse:
    """Runs inspection by fetching existing sample image from ML service."""
    # Download sample image from ML service
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            sample_res = await client.get(f"{settings.ML_SERVICE_URL}/samples/{category}/{filename}")
            if sample_res.status_code != 200:
                raise HTTPException(status_code=404, detail=f"Próbka {filename} nie została znaleziona w serwisie ML.")
            file_bytes = sample_res.content

            # Send to /predict
            predict_res = await client.post(
                f"{settings.ML_SERVICE_URL}/predict",
                data={"model_name": model_name},
                files={"file": (filename, file_bytes, "image/jpeg")}
            )
            if predict_res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd predykcji serwisu ML: {predict_res.text}")
            ml_data = predict_res.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")

    prob_defect = float(ml_data["prob_defect"])
    prob_ok = float(ml_data["prob_ok"])
    latency_ms = float(ml_data["latency_ms"])

    is_defective = prob_defect >= threshold
    if is_defective:
        verdict = "DEFECT"
        recommendation = "ODRZUCONY • KIERUNEK: RECYKLING / PRZETOP"
    else:
        verdict = "OK"
        recommendation = "ZATWIERDZONY • KIERUNEK: MONTAŻ POMPY"

    record = InspectionRecord(
        filename=filename,
        model_name=model_name,
        threshold=threshold,
        prob_defect=prob_defect,
        prob_ok=prob_ok,
        is_defective=is_defective,
        verdict=verdict,
        recommendation=recommendation,
        latency_ms=latency_ms,
        notes=notes or f"Próbka wzorcowa ({category})"
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return InspectionResponse(
        id=record.id,
        created_at=record.created_at,
        filename=filename,
        model_name=model_name,
        threshold=threshold,
        prob_defect=prob_defect,
        prob_ok=prob_ok,
        is_defective=is_defective,
        verdict=verdict,
        recommendation=recommendation,
        latency_ms=latency_ms,
        target_met_le_100ms=latency_ms <= 100.0,
        notes=record.notes
    )


def get_inspection_history(db: Session, skip: int = 0, limit: int = 50) -> List[InspectionHistoryItem]:
    """Retrieves paginated inspection audit log."""
    return db.query(InspectionRecord).order_by(InspectionRecord.created_at.desc()).offset(skip).limit(limit).all()


def get_inspection_stats(db: Session) -> InspectionStats:
    """Computes aggregated KPIs from database."""
    total = db.query(InspectionRecord).count()
    if total == 0:
        return InspectionStats(
            total_inspections=0,
            total_defects=0,
            total_approved=0,
            defect_rate_percent=0.0,
            average_latency_ms=0.0
        )

    defects = db.query(InspectionRecord).filter(InspectionRecord.is_defective == True).count()
    approved = total - defects
    avg_latency = db.query(func.avg(InspectionRecord.latency_ms)).scalar() or 0.0

    return InspectionStats(
        total_inspections=total,
        total_defects=defects,
        total_approved=approved,
        defect_rate_percent=round((defects / total) * 100.0, 1),
        average_latency_ms=round(float(avg_latency), 2)
    )


def clear_inspection_history(db: Session) -> int:
    """Clears history audit records."""
    deleted = db.query(InspectionRecord).delete()
    db.commit()
    return deleted
