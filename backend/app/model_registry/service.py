import httpx
from fastapi import HTTPException
from app.config import settings
from app.model_registry.schemas import ModelListResponse, BenchmarkResponse


async def fetch_available_models() -> ModelListResponse:
    """Queries ML microservice for registered model architectures and status."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.get(f"{settings.ML_SERVICE_URL}/models")
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd serwisu ML: {res.text}")
            return ModelListResponse(**res.json())
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")


async def run_model_benchmark(model_name: str, iterations: int = 20) -> BenchmarkResponse:
    """Executes latency benchmarking via ML microservice."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(
                f"{settings.ML_SERVICE_URL}/benchmark",
                data={"model_name": model_name, "iterations": iterations}
            )
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd benchmarku serwisu ML: {res.text}")
            return BenchmarkResponse(**res.json())
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")


async def fetch_model_memory_telemetry() -> dict:
    """Queries ML service for RAM usage and on-demand model status."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{settings.ML_SERVICE_URL}/models/memory")
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd odpytania serwisu ML: {res.text}")
            return res.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")


async def request_unload_models() -> dict:
    """Requests ML service to unload all PyTorch models from RAM."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(f"{settings.ML_SERVICE_URL}/models/unload")
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd odpytania serwisu ML: {res.text}")
            return res.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")


async def fetch_cpu_config() -> dict:
    """Queries ML service for CPU threads, hardware profiles, and throttling status."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{settings.ML_SERVICE_URL}/cpu/config")
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd odpytania serwisu ML o CPU: {res.text}")
            return res.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")


async def update_cpu_config(payload: dict) -> dict:
    """Updates CPU configuration (threads, throttling percentage, hardware profile) in ML service."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(f"{settings.ML_SERVICE_URL}/cpu/config", json=payload)
            if res.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Błąd aktualizacji konfiguracji CPU w ML: {res.text}")
            return res.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Brak połączenia z serwisem ML: {str(exc)}")


