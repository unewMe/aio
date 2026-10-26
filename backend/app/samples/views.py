from fastapi import APIRouter
from app.samples import service

router = APIRouter(prefix="/samples", tags=["Sample Images"])


@router.get("", summary="Pobierz listę próbek wzorcowych")
async def list_samples():
    """Zwraca dostępne obrazy odlewów poprawnych (ok) oraz wadliwych (def)."""
    return await service.get_sample_list()


@router.get("/{category}/{filename}", summary="Pobierz plik obrazu próbki")
async def get_sample_image(category: str, filename: str):
    """Zwraca binarny plik graficzny próbki (JPEG)."""
    return await service.stream_sample_image(category=category, filename=filename)
