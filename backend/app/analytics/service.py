import httpx
from typing import Dict, Any
from fastapi import HTTPException
from fastapi.responses import Response
from app.config import settings


async def get_evaluation_data() -> Dict[str, Any]:
    """Fetches final evaluation results (test set metrics, confusion matrices, ROC/PR stats)."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(f"{settings.ML_SERVICE_URL}/reports/evaluation")
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd raportu serwisu ML: {res.text}")
            return res.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")


async def get_training_history() -> Dict[str, Any]:
    """Fetches epoch-by-epoch loss and validation metrics."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(f"{settings.ML_SERVICE_URL}/reports/training-history")
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd historii serwisu ML: {res.text}")
            return res.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")


async def list_available_figures() -> Dict[str, Any]:
    """Lists plot image files available in reports."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(f"{settings.ML_SERVICE_URL}/reports/figures")
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd listy wykresów: {res.text}")
            return res.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")


async def proxy_figure_image(figure_name: str) -> Response:
    """Streams figure PNG from ML service."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(f"{settings.ML_SERVICE_URL}/reports/figures/{figure_name}")
            if res.status_code != 200:
                raise HTTPException(status_code=404, detail="Wykres nie został odnaleziony w serwisie ML.")
            return Response(content=res.content, media_type="image/png")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")
