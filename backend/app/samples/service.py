import httpx
from typing import Dict, Any
from fastapi import HTTPException
from fastapi.responses import Response
from app.config import settings


async def get_sample_list() -> Dict[str, Any]:
    """Retrieves list of reference test samples from ML microservice."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(f"{settings.ML_SERVICE_URL}/samples")
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd pobierania próbek: {res.text}")
            return res.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")


async def stream_sample_image(category: str, filename: str) -> Response:
    """Streams sample image JPEG from ML microservice."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(f"{settings.ML_SERVICE_URL}/samples/{category}/{filename}")
            if res.status_code != 200:
                raise HTTPException(status_code=404, detail="Próbka nie została odnaleziona w serwisie ML.")
            return Response(content=res.content, media_type="image/jpeg")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")
