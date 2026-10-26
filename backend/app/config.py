import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "System Automatycznej Kontroli Jakości Odlewów (AOI)"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # ML Inference Service URL (separate microservice)
    ML_SERVICE_URL: str = os.getenv("ML_SERVICE_URL", "http://localhost:8001")
    
    # Database (SQLite by default for audit & inspection history)
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/inspection_history.db")
    
    # Upload storage for inspected images
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "*"
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
