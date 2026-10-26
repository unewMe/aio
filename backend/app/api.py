from fastapi import APIRouter
from app.inspection.views import router as inspection_router
from app.model_registry.views import router as model_router
from app.analytics.views import router as analytics_router
from app.samples.views import router as samples_router
from app.simulation.views import router as simulation_router

api_router = APIRouter()

# Register bounded domain routers (Netflix Dispatch DDD pattern)
api_router.include_router(inspection_router)
api_router.include_router(model_router)
api_router.include_router(analytics_router)
api_router.include_router(samples_router)
api_router.include_router(simulation_router)
