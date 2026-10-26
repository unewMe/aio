from fastapi import APIRouter
from app.analytics import service

router = APIRouter(prefix="/analytics", tags=["Analytics & Reports"])


@router.get("/evaluation", summary="Pobierz wyniki ewaluacji modeli na zbiorze testowym")
async def get_evaluation():
    """Zwraca metryki modeli (Recall, Precision, F1, macierze pomyłek, krzywe ROC/PR)."""
    return await service.get_evaluation_data()


@router.get("/training-history", summary="Pobierz historię uczenia modeli (epoki)")
async def get_training_history():
    """Zwraca przebieg funkcji straty BCE oraz metryk walidacyjnych w kolejnych epokach."""
    return await service.get_training_history()


@router.get("/figures", summary="Lista wygenerowanych wykresów ewaluacyjnych")
async def list_figures():
    """Zwraca listę dostępnych wykresów w formacie PNG."""
    return await service.list_available_figures()


@router.get("/figures/{figure_name}", summary="Pobierz obraz wykresu PNG")
async def get_figure(figure_name: str):
    """Przesyła plik PNG z wykresem (np. test_roc_pr_curves.png, test_confusion_matrices.png)."""
    return await service.proxy_figure_image(figure_name)
