from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.inspection import service
from app.inspection.schemas import InspectionResponse, InspectionHistoryItem, InspectionStats

router = APIRouter(prefix="/inspections", tags=["Inspections"])


@router.post("/upload", response_model=InspectionResponse, summary="Uruchom inspekcję z przesłanego pliku")
async def inspect_uploaded_image(
    file: UploadFile = File(...),
    model_name: str = Form("resnet18"),
    threshold: float = Form(0.197),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Przyjmuje plik obrazu wirnika, odpytuje zewnętrzny mikroserwis ML i rejestruje werdykt w bazie danych.
    """
    return await service.run_inspection_from_upload(
        db=db,
        file=file,
        model_name=model_name,
        threshold=threshold,
        notes=notes
    )


@router.post("/sample", response_model=InspectionResponse, summary="Uruchom inspekcję próbki wzorcowej")
async def inspect_sample_image(
    category: str = Form(..., description="Kategoria próbki: 'def' lub 'ok'"),
    filename: str = Form(..., description="Nazwa pliku próbki"),
    model_name: str = Form("resnet18"),
    threshold: float = Form(0.197),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Pobiera próbkę ze zbioru referencyjnego serwisu ML i wykonuje inspekcję.
    """
    return await service.run_inspection_from_sample(
        db=db,
        category=category,
        filename=filename,
        model_name=model_name,
        threshold=threshold,
        notes=notes
    )


@router.get("/history", response_model=List[InspectionHistoryItem], summary="Pobierz historię inspekcji")
def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Zwraca stronicowaną historię wykonanych inspekcji jakościowych."""
    return service.get_inspection_history(db=db, skip=skip, limit=limit)


@router.get("/stats", response_model=InspectionStats, summary="Pobierz zagregowane statystyki KPI")
def get_stats(db: Session = Depends(get_db)):
    """Oblicza wskaźnik wadliwości, całkowitą liczbę odlewów oraz średnie opóźnienie."""
    return service.get_inspection_stats(db=db)


@router.delete("/clear", summary="Wyczyść historię inspekcji")
def clear_history(db: Session = Depends(get_db)):
    """Czyści dziennik audytu inspekcji."""
    count = service.clear_inspection_history(db=db)
    return {"status": "success", "deleted_records": count}
