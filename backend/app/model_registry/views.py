from fastapi import APIRouter, Query
from app.model_registry import service
from app.model_registry.schemas import ModelListResponse, BenchmarkResponse

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get("", response_model=ModelListResponse, summary="Pobierz listę zarejestrowanych modeli")
async def list_models():
    """Zwraca metadane modeli AI, status załadowanych wag oraz optymalne progi decyzyjne."""
    return await service.fetch_available_models()


@router.post("/benchmark", response_model=BenchmarkResponse, summary="Uruchom pomiar opóźnienia inferencji CPU")
async def benchmark_model(
    model_name: str = Query("resnet18", description="Identyfikator modelu: resnet18, mobilenet_v2, cnn"),
    iterations: int = Query(20, ge=5, le=100, description="Liczba powtórzeń testu")
):
    """Mierzy czas inferencji (średnia, mediana, p95) w mikroserwisie ML."""
    return await service.run_model_benchmark(model_name=model_name, iterations=iterations)


@router.get("/memory", summary="Pobierz stan pamięci RAM i załadowanych modeli")
async def get_model_memory():
    """Zwraca bieżące zużycie RAM procesu ML oraz listę załadowanych wag on-demand."""
    return await service.fetch_model_memory_telemetry()


@router.post("/unload", summary="Zwolnij wszystkie modele z pamięci RAM")
async def unload_models():
    """Wymusza wyczyszczenie pamięci podręcznej modeli i uruchamia garbage collector w mikroserwisie ML."""
    return await service.request_unload_models()


@router.get("/cpu-config", summary="Pobierz konfigurację i profile sprzętowe procesora CPU")
async def get_cpu_configuration():
    """Zwraca bieżący profil CPU, liczbę wątków, procent dławienia oraz listę profili urządzeń."""
    return await service.fetch_cpu_config()


@router.post("/cpu-config", summary="Zmień moc obliczeniową CPU i profil sprzętowy")
async def update_cpu_configuration(payload: dict):
    """Pozwala ograniczyć moc procesora, wybrać profil sprzętowy (np. Raspberry Pi) lub ustawić liczbę wątków."""
    return await service.update_cpu_config(payload)


