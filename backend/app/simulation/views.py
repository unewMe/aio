from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.simulation import service
from app.simulation.schemas import SimulationStepResponse, SimulationItemsResponse

router = APIRouter(prefix="/simulation", tags=["Conveyor Simulation"])


@router.get("/items", response_model=SimulationItemsResponse, summary="Pobierz listę próbek ze zbioru testowego")
async def get_test_items():
    """Zwraca listę 1200 odlewów ze zbioru testowego wraz z ich etykietami Ground Truth."""
    return await service.fetch_simulation_items()


@router.post("/step", response_model=SimulationStepResponse, summary="Wykonaj pojedynczy krok symulacji taśmy produkcyjnej")
async def run_simulation_step(
    model_name: str = Query("resnet18", description="Nazwa modelu"),
    threshold: float = Query(0.197, description="Próg decyzyjny tau"),
    item_id: Optional[int] = Query(None, description="ID próbki testowej (None = losowa)"),
    category_filter: str = Query("all", description="Filtr próbek: 'all', 'def', 'ok', 'realistic'"),
    db: Session = Depends(get_db)
):
    """
    Pobiera odlew z taśmy (losowo ze zbioru testowego lub wskazany ID),
    wykonuje inferencję, porównuje decyzję ze stanem faktycznym (Ground Truth)
    oraz kwalifikuje wynik jako TP, TN, FP lub FN.
    """
    return await service.run_simulation_step(
        db=db,
        model_name=model_name,
        threshold=threshold,
        item_id=item_id,
        category_filter=category_filter
    )


@router.get("/image/{item_id}", summary="Pobierz obraz próbki testowej")
async def get_simulation_image(item_id: int):
    """Strumieniuje plik graficzny próbki testowej."""
    return await service.proxy_simulation_image(item_id=item_id)


from app.model_registry import service as mr_service


@router.get("/cpu-config", summary="Pobierz konfigurację mocy procesora CPU symulacji")
async def get_simulation_cpu_config():
    """Zwraca stan dławienia procesora CPU i profile sprzętowe (np. Raspberry Pi, IPC)."""
    return await mr_service.fetch_cpu_config()


@router.post("/cpu-config", summary="Zmień moc obliczeniową CPU symulacji")
async def update_simulation_cpu_config(payload: dict):
    """Pozwala ograniczyć moc CPU w celu przetestowania taśmy produkcyjnej na słabszym sprzęcie."""
    return await mr_service.update_cpu_config(payload)
