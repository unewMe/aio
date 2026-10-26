import httpx
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException
from fastapi.responses import Response

from app.config import settings
from app.inspection.models import InspectionRecord
from app.simulation.schemas import SimulationStepResponse, SimulationItemsResponse


async def fetch_simulation_items() -> SimulationItemsResponse:
    """Fetches list of all test set items with ground truth labels from ML service."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(f"{settings.ML_SERVICE_URL}/simulation/items")
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd pobierania listy próbek testowych: {res.text}")
            return SimulationItemsResponse(**res.json())
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z mikroserwisem ML: {str(exc)}")


async def run_simulation_step(
    db: Session,
    model_name: str,
    threshold: float,
    item_id: Optional[int] = None,
    category_filter: str = "all"
) -> SimulationStepResponse:
    """
    Executes a single production conveyor simulation step:
    1. Requests inference on a random (or specified) test set sample from ML service.
    2. Applies industrial decision rule tau.
    3. Compares model verdict with Ground Truth (labels from test set).
    4. Categorizes into TP (True Positive), TN (True Negative), FN (False Negative), FP (False Positive).
    5. Saves inspection record into database audit trail.
    """
    form_data = {
        "model_name": model_name,
        "category_filter": category_filter
    }
    if item_id is not None and item_id >= 0:
        form_data["item_id"] = str(item_id)
    else:
        form_data["item_id"] = "-1"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(f"{settings.ML_SERVICE_URL}/simulation/inspect", data=form_data)
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd inferencji w serwisie ML: {res.text}")
            ml_data = res.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z mikroserwisem ML: {str(exc)}")

    sim_id = int(ml_data["item_id"])
    filename = ml_data["filename"]
    filepath = ml_data["filepath"]
    prob_defect = float(ml_data["prob_defect"])
    prob_ok = float(ml_data["prob_ok"])
    latency_ms = float(ml_data["latency_ms"])
    ground_truth = ml_data["ground_truth"]  # "def" | "ok"
    ground_truth_label = int(ml_data["ground_truth_label"])
    ground_truth_name = ml_data["ground_truth_name"]

    # Model Decision Rule
    is_defective = prob_defect >= threshold
    if is_defective:
        verdict = "DEFECT"
        recommendation = "ODRZUCONY • KIERUNEK: RECYKLING / PRZETOP"
    else:
        verdict = "OK"
        recommendation = "ZATWIERDZONY • KIERUNEK: MONTAŻ POMPY"

    # Ground Truth vs Prediction Evaluation
    is_actual_defect = ground_truth_label == 1

    if is_defective and is_actual_defect:
        classification_type = "TP"
        is_correct = True
        classification_badge = "Trafne wykrycie wady (True Positive)"
    elif not is_defective and not is_actual_defect:
        classification_type = "TN"
        is_correct = True
        classification_badge = "Trafne zatwierdzenie (True Negative)"
    elif not is_defective and is_actual_defect:
        classification_type = "FN"
        is_correct = False
        classification_badge = "BŁĄD KRYTYCZNY: Przeoczenie wady (False Negative)"
    else:  # is_defective and not is_actual_defect
        classification_type = "FP"
        is_correct = False
        classification_badge = "Fałszywy alarm: Odrzucenie sprawnego (False Positive)"

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
        notes=f"Symulacja taśmy produkcyjnej | Ground Truth: {ground_truth_name} | Typ: {classification_type}"
    )
    db.add(record)
    db.commit()

    image_url = f"/api/v1/simulation/image/{sim_id}"
    image_base64 = ml_data.get("image_base64")

    return SimulationStepResponse(
        item_id=sim_id,
        filename=filename,
        filepath=filepath,
        model_name=model_name,
        threshold=threshold,
        prob_defect=prob_defect,
        prob_ok=prob_ok,
        is_defective=is_defective,
        verdict=verdict,
        recommendation=recommendation,
        latency_ms=latency_ms,
        ground_truth=ground_truth,
        ground_truth_label=ground_truth_label,
        ground_truth_name=ground_truth_name,
        is_correct=is_correct,
        classification_type=classification_type,
        classification_badge=classification_badge,
        image_url=image_url,
        image_base64=image_base64,
        raw_latency_ms=ml_data.get("raw_latency_ms"),
        cpu_config=ml_data.get("cpu_config")
    )


async def proxy_simulation_image(item_id: int) -> Response:
    """Streams test set image from ML service."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(f"{settings.ML_SERVICE_URL}/simulation/image/{item_id}")
            if res.status_code != 200:
                raise HTTPException(status_code=404, detail="Próbka testowa nie została odnaleziona.")
            return Response(content=res.content, media_type="image/jpeg")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z mikroserwisem ML: {str(exc)}")
