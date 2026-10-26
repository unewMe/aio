"""
Dedicated Machine Learning Inference Microservice for Casting Inspection.
Serves PyTorch models: ResNet18, MobileNetV2, and Custom CNN.
"""

import os
import glob
import time
import json
import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import torch

from src.models import build_model, count_parameters, get_model_size_mb
from src.dataset import get_eval_transforms

app = FastAPI(
    title="Casting Quality Inspection ML Service",
    description="Dedicated PyTorch inference service for industrial casting quality control",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "models_weights"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

import gc
import resource

# Model metadata and mapping with pre-calculated parameters and sizes
MODEL_CONFIGS = {
    "resnet18": {
        "display_name": "ResNet18",
        "key": "resnet18",
        "weights_file": "resnet18.pth",
        "optimal_threshold": 0.197,
        "role": "Najwyższa Dokładność • Transfer Learning",
        "description": "Sieć rezydualna z wagami ImageNet dostrojona do kontroli jakości wirników.",
        "parameters": 11177537,
        "size_mb": 42.71
    },
    "mobilenet_v2": {
        "display_name": "MobileNetV2",
        "key": "mobilenet_v2",
        "weights_file": "mobilenet_v2.pth",
        "optimal_threshold": 0.099,
        "role": "Optimum Brzegowe • Edge AI",
        "description": "Lekka architektura zoptymalizowana pod minimalne opóźnienia na procesorach CPU.",
        "parameters": 2225153,
        "size_mb": 8.56
    },
    "cnn": {
        "display_name": "Custom CNN",
        "key": "cnn",
        "weights_file": "custom_cnn.pth",
        "optimal_threshold": 0.158,
        "role": "Autorska Sieć Bazowa • Baseline",
        "description": "Autorska 4-blokowa sieć konwolucyjna stworzona specjalnie dla odlewów.",
        "parameters": 994369,
        "size_mb": 3.82
    }
}

# Cache for loaded models - On-Demand Single Active Model Policy
_loaded_models: Dict[str, torch.nn.Module] = {}
MAX_LOADED_MODELS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_eval_transforms()


def get_process_memory_mb() -> float:
    """Returns RSS memory usage of the current process in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(float(line.split()[1]) / 1024.0, 2)
    except Exception:
        pass
    try:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    except Exception:
        return 0.0


def unload_inactive_models(keep_key: Optional[str] = None):
    """Releases inactive models from memory and runs garbage collection."""
    global _loaded_models
    keys_to_remove = [k for k in list(_loaded_models.keys()) if k != keep_key]
    for k in keys_to_remove:
        print(f"[RAM Optimization] Unloading inactive model '{k}' from RAM...")
        del _loaded_models[k]
    if keys_to_remove:
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


TOTAL_SYSTEM_CORES = os.cpu_count() or 4
DEFAULT_CPU_THREADS = min(TOTAL_SYSTEM_CORES, torch.get_num_threads())

HARDWARE_PROFILES = {
    "unthrottled": {
        "id": "unthrottled",
        "name": "Mocna Stacja Robocza / Serwer (Pełna moc)",
        "threads": TOTAL_SYSTEM_CORES,
        "throttle_percent": 100,
        "multiplier": 1.0,
        "description": f"Maksymalna wydajność hosta ({TOTAL_SYSTEM_CORES} wątków CPU). Zero ograniczeń."
    },
    "industrial_pc": {
        "id": "industrial_pc",
        "name": "Przemysłowy IPC (Intel Core i3 / Celeron)",
        "threads": min(4, max(1, TOTAL_SYSTEM_CORES // 2)),
        "throttle_percent": 60,
        "multiplier": 1.8,
        "description": "Standardowy komputer panelowy w szafie automatyki przemysłowej."
    },
    "jetson_nano": {
        "id": "jetson_nano",
        "name": "Kontroler Krawędziowy (Jetson Nano / Edge AI)",
        "threads": min(2, TOTAL_SYSTEM_CORES),
        "throttle_percent": 35,
        "multiplier": 3.2,
        "description": "Moduł brzegowy zoptymalizowany pod montaż przy samej taśmie."
    },
    "raspberry_pi_4": {
        "id": "raspberry_pi_4",
        "name": "Raspberry Pi 4 / 5 (Tani Komputer IoT - ARM)",
        "threads": min(2, TOTAL_SYSTEM_CORES),
        "throttle_percent": 18,
        "multiplier": 5.5,
        "description": "Popularny budżetowy mikrokomputer jednopłytkowy (ARM Cortex-A72 @ 1.5 GHz)."
    },
    "ultra_low_power": {
        "id": "ultra_low_power",
        "name": "Skrajnie Słaby Mikrokontroler (1-Core Ultra Low-Power)",
        "threads": 1,
        "throttle_percent": 8,
        "multiplier": 12.0,
        "description": "Symulacja skrajnego dławienia i wąskiego gardła taśmy produkcyjnej."
    },
    "custom": {
        "id": "custom",
        "name": "Własna konfiguracja (Custom)",
        "threads": 1,
        "throttle_percent": 50,
        "multiplier": 2.0,
        "description": "Ręcznie zdefiniowane parametry mocy i liczby wątków."
    }
}

_current_cpu_config = {
    "profile": "unthrottled",
    "profile_name": HARDWARE_PROFILES["unthrottled"]["name"],
    "threads": TOTAL_SYSTEM_CORES,
    "throttle_percent": 100,
    "total_system_cores": TOTAL_SYSTEM_CORES,
    "is_throttled": False
}


def apply_cpu_configuration(
    profile: Optional[str] = None,
    threads: Optional[int] = None,
    throttle_percent: Optional[int] = None
) -> Dict[str, Any]:
    global _current_cpu_config
    if profile and profile in HARDWARE_PROFILES and profile != "custom":
        p = HARDWARE_PROFILES[profile]
        t = threads if (threads is not None and threads > 0) else p["threads"]
        pct = throttle_percent if (throttle_percent is not None and throttle_percent > 0) else p["throttle_percent"]
        prof_name = p["name"]
        prof_id = profile
    else:
        prof_id = "custom" if (profile == "custom" or threads is not None or throttle_percent is not None) else _current_cpu_config["profile"]
        prof_name = HARDWARE_PROFILES.get(prof_id, {}).get("name", "Własna konfiguracja (Custom)")
        t = threads if (threads is not None and threads > 0) else _current_cpu_config["threads"]
        pct = throttle_percent if (throttle_percent is not None and throttle_percent > 0) else _current_cpu_config["throttle_percent"]

    t = max(1, min(TOTAL_SYSTEM_CORES, int(t)))
    pct = max(5, min(100, int(pct)))
    is_throttled = (pct < 100 or t < TOTAL_SYSTEM_CORES)

    try:
        torch.set_num_threads(t)
    except Exception as e:
        print(f"[CPU Config] Warning setting torch threads to {t}: {e}")

    _current_cpu_config = {
        "profile": prof_id,
        "profile_name": prof_name,
        "threads": t,
        "throttle_percent": pct,
        "total_system_cores": TOTAL_SYSTEM_CORES,
        "is_throttled": is_throttled
    }
    print(f"[CPU Config] Updated: threads={t}/{TOTAL_SYSTEM_CORES}, throttle={pct}%, profile='{prof_id}'")
    return _current_cpu_config


def execute_throttled_inference(model: torch.nn.Module, tensor: torch.Tensor):
    """
    Executes PyTorch model forward pass respecting configured CPU threads
    and simulates weaker hardware clock rate/IPC if throttling is active.
    Returns (prob_defect, prob_ok, total_latency_ms, raw_latency_ms).
    """
    try:
        torch.set_num_threads(_current_cpu_config["threads"])
    except Exception:
        pass

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        logits = model(tensor)
        prob_defect = torch.sigmoid(logits).item()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    raw_latency_ms = (t1 - t0) * 1000.0

    # Simulate CPU throttling / weak hardware slowdown
    pct = _current_cpu_config["throttle_percent"]
    if pct < 100:
        multiplier = 100.0 / max(5.0, float(pct))
        target_total_sec = (raw_latency_ms * multiplier) / 1000.0
        spent_sec = t1 - t0
        extra_sec = target_total_sec - spent_sec
        if extra_sec > 0:
            time.sleep(extra_sec)

    t_end = time.perf_counter()
    total_latency_ms = (t_end - t0) * 1000.0
    prob_ok = 1.0 - prob_defect

    return prob_defect, prob_ok, round(float(total_latency_ms), 2), round(float(raw_latency_ms), 2)


def get_or_load_model(model_name: str) -> torch.nn.Module:
    """
    On-Demand Model Loader:
    Only loads the model into RAM when requested. If another model is active,
    it is automatically offloaded to keep the memory footprint minimal.
    """
    model_key = model_name.lower().replace("-", "_").replace(" ", "_")
    if model_key in ["custom_cnn", "customcnn"]:
        model_key = "cnn"
    elif model_key in ["resnet", "resnet18"]:
        model_key = "resnet18"
    elif model_key in ["mobilenet", "mobilenetv2"]:
        model_key = "mobilenet_v2"

    if model_key not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Nieznany model: {model_name}. Dozwolone: resnet18, mobilenet_v2, cnn")

    if model_key in _loaded_models:
        return _loaded_models[model_key]

    # Free memory from previous model before loading new one (Single Active Slot)
    unload_inactive_models(keep_key=None)

    config = MODEL_CONFIGS[model_key]
    weight_path = WEIGHTS_DIR / config["weights_file"]

    print(f"[RAM Optimization] Loading model '{model_key}' on-demand into RAM...")
    model = build_model(config["key"], pretrained=False, device=str(DEVICE))
    if weight_path.exists():
        state_dict = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Weights file {weight_path} not found. Running with uninitialized weights.")

    model.eval()
    _loaded_models[model_key] = model
    print(f"[RAM Optimization] Model '{model_key}' ready. RSS Memory: {get_process_memory_mb()} MB")
    return model


@app.get("/health")
def health_check():
    """Health check endpoint with memory telemetry."""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "loaded_models": list(_loaded_models.keys()),
        "active_model": list(_loaded_models.keys())[0] if _loaded_models else None,
        "rss_memory_mb": get_process_memory_mb(),
        "weights_dir_exists": WEIGHTS_DIR.exists(),
    }


@app.get("/models")
def list_models():
    """Returns available models without forcing all weights into RAM."""
    models_info = []
    for key, config in MODEL_CONFIGS.items():
        weight_path = WEIGHTS_DIR / config["weights_file"]
        has_weights = weight_path.exists()
        models_info.append({
            "id": key,
            "display_name": config["display_name"],
            "role": config["role"],
            "description": config["description"],
            "optimal_threshold": config["optimal_threshold"],
            "has_weights": has_weights,
            "weights_file": config["weights_file"],
            "parameters": config["parameters"],
            "size_mb": config["size_mb"],
            "device": str(DEVICE),
            "is_loaded": key in _loaded_models
        })
    return {
        "models": models_info,
        "strategy": "on-demand (single active model)",
        "loaded_models": list(_loaded_models.keys()),
        "rss_memory_mb": get_process_memory_mb()
    }


@app.get("/models/memory")
def get_memory_telemetry():
    """Returns RAM telemetry and on-demand model status."""
    return {
        "strategy": "on-demand (single active model)",
        "max_loaded_models": MAX_LOADED_MODELS,
        "loaded_models": list(_loaded_models.keys()),
        "active_model": list(_loaded_models.keys())[0] if _loaded_models else None,
        "rss_memory_mb": get_process_memory_mb(),
        "device": str(DEVICE)
    }


@app.post("/models/unload")
def unload_all_models():
    """Explicitly releases all loaded PyTorch models from RAM."""
    unload_inactive_models(keep_key=None)
    gc.collect()
    return {
        "status": "unloaded",
        "loaded_models": list(_loaded_models.keys()),
        "rss_memory_mb": get_process_memory_mb()
    }


@app.get("/cpu/config")
def get_cpu_configuration():
    """Returns current CPU threads, throttling factor, and hardware profiles."""
    return {
        **_current_cpu_config,
        "profiles": HARDWARE_PROFILES
    }


from pydantic import BaseModel


class CpuConfigUpdatePayload(BaseModel):
    profile: Optional[str] = None
    threads: Optional[int] = None
    throttle_percent: Optional[int] = None


@app.post("/cpu/config")
def update_cpu_configuration(
    payload: Optional[CpuConfigUpdatePayload] = None,
    profile: Optional[str] = None,
    threads: Optional[int] = None,
    throttle_percent: Optional[int] = None
):
    """Updates CPU hardware profile, thread allocation, and throttling percentage."""
    req_prof = payload.profile if payload and payload.profile else profile
    req_threads = payload.threads if payload and payload.threads is not None else threads
    req_pct = payload.throttle_percent if payload and payload.throttle_percent is not None else throttle_percent

    cfg = apply_cpu_configuration(profile=req_prof, threads=req_threads, throttle_percent=req_pct)
    return {
        **cfg,
        "profiles": HARDWARE_PROFILES
    }


@app.post("/predict")
async def predict_casting(
    file: UploadFile = File(...),
    model_name: str = Form("resnet18")
):
    """
    Runs PyTorch inference on uploaded image file.
    Respects current CPU threads and hardware throttling settings.
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Niepoprawny plik graficzny: {str(e)}")

    model = get_or_load_model(model_name)
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    prob_defect, prob_ok, latency_ms, raw_latency_ms = execute_throttled_inference(model, tensor)

    return {
        "model_name": model_name,
        "prob_defect": round(float(prob_defect), 6),
        "prob_ok": round(float(prob_ok), 6),
        "latency_ms": latency_ms,
        "raw_latency_ms": raw_latency_ms,
        "cpu_config": _current_cpu_config,
        "device": str(DEVICE),
        "image_size": [image.width, image.height]
    }


@app.post("/benchmark")
def benchmark_model(
    model_name: str = Form("resnet18"),
    iterations: int = Form(20)
):
    """Measures inference latency over N runs taking into account configured CPU throttling."""
    model = get_or_load_model(model_name)
    dummy_input = torch.randn(1, 3, 300, 300).to(DEVICE)

    try:
        torch.set_num_threads(_current_cpu_config["threads"])
    except Exception:
        pass

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)

    pct = _current_cpu_config["throttle_percent"]
    multiplier = 100.0 / max(5.0, float(pct))

    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            raw_lat = (t1 - t0) * 1000.0
            throttled_lat = raw_lat * multiplier
            latencies.append(throttled_lat)

    mean_lat = sum(latencies) / len(latencies)
    sorted_lat = sorted(latencies)
    median_lat = sorted_lat[len(sorted_lat) // 2]
    p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)]

    return {
        "model_name": model_name,
        "device": str(DEVICE),
        "iterations": iterations,
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "target_met_le_100ms": mean_lat <= 100.0,
        "cpu_config": _current_cpu_config
    }


@app.get("/reports/evaluation")
def get_evaluation_results():
    """Serves computed evaluation results from training/evaluation."""
    eval_file = REPORTS_DIR / "final_evaluation_results.json"
    if not eval_file.exists():
        raise HTTPException(status_code=404, detail="Raport ewaluacyjny nie został jeszcze wygenerowany.")
    with open(eval_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/reports/training-history")
def get_training_history():
    """Serves training loss and metric curves history."""
    hist_file = REPORTS_DIR / "final_train_history.json"
    if not hist_file.exists():
        raise HTTPException(status_code=404, detail="Historia trenowania nie została znaleziona.")
    with open(hist_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/reports/figures")
def list_figures():
    """Lists available report figures (plots)."""
    fig_dir = REPORTS_DIR / "figures"
    if not fig_dir.exists():
        return {"figures": []}
    files = [f.name for f in fig_dir.glob("*.png")]
    return {"figures": files}


@app.get("/reports/figures/{figure_name}")
def get_figure_image(figure_name: str):
    """Serves evaluation plot PNG file."""
    fig_path = REPORTS_DIR / "figures" / figure_name
    if not fig_path.exists():
        err_path = REPORTS_DIR / "error_analysis" / figure_name
        if err_path.exists():
            return FileResponse(err_path, media_type="image/png")
        raise HTTPException(status_code=404, detail="Wykres nie został odnaleziony.")
    return FileResponse(fig_path, media_type="image/png")


@app.get("/samples")
def list_sample_images():
    """Lists available sample images for quick testing."""
    sample_dir = DATA_DIR / "sample_images"
    samples = {"def": [], "ok": []}

    def_files = sorted(glob.glob(str(sample_dir / "def" / "*.jpeg")))
    if not def_files:
        def_files = sorted(glob.glob(str(BASE_DIR / "casting_data" / "def_front" / "*.jpeg")))[:20]
    for p in def_files[:15]:
        samples["def"].append({
            "category": "def",
            "filename": os.path.basename(p),
            "label": "Wadliwy (WADA - pęknięcie/niedolew)"
        })

    ok_files = sorted(glob.glob(str(sample_dir / "ok" / "*.jpeg")))
    if not ok_files:
        ok_files = sorted(glob.glob(str(BASE_DIR / "casting_data" / "ok_front" / "*.jpeg")))[:20]
    for p in ok_files[:15]:
        samples["ok"].append({
            "category": "ok",
            "filename": os.path.basename(p),
            "label": "Poprawny (OK - bez wad)"
        })

    return samples


@app.get("/samples/{category}/{filename}")
def get_sample_image(category: str, filename: str):
    """Serves a sample image file."""
    # Check sample_images first
    path = DATA_DIR / "sample_images" / category / filename
    if not path.exists():
        # Fallback to casting_data
        cat_folder = "def_front" if category == "def" else "ok_front"
        path = BASE_DIR / "casting_data" / cat_folder / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Próbka {filename} nie została odnaleziona.")
    return FileResponse(path, media_type="image/jpeg")

# 5. PRODUCTION STREAM SIMULATION (TEST SET WITH GROUND TRUTH)
import random

_test_items = []


def get_test_items():
    global _test_items
    if not _test_items:
        test_split_path = REPORTS_DIR / "test_split.json"
        if test_split_path.exists():
            with open(test_split_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            paths = data.get("test_filepaths", [])
            labels = data.get("test_labels", [])
            for i, (p, lbl) in enumerate(zip(paths, labels)):
                _test_items.append({
                    "id": i,
                    "filepath": p,
                    "filename": os.path.basename(p),
                    "ground_truth": "def" if lbl == 1 else "ok",
                    "ground_truth_label": int(lbl),
                    "ground_truth_name": "WADLIWY" if lbl == 1 else "POPRAWNY"
                })
    return _test_items


@app.get("/simulation/items")
def list_simulation_items():
    """Returns list of test set items with ground truth labels."""
    items = get_test_items()
    # Return light metadata for all 1200 test samples
    return {
        "total": len(items),
        "items": [
            {
                "id": it["id"],
                "filename": it["filename"],
                "ground_truth": it["ground_truth"],
                "ground_truth_label": it["ground_truth_label"],
                "ground_truth_name": it["ground_truth_name"]
            }
            for it in items
        ]
    }


@app.get("/simulation/image/{item_id}")
def get_simulation_image(item_id: int):
    """Streams the test set image file for given ID."""
    items = get_test_items()
    if item_id < 0 or item_id >= len(items):
        raise HTTPException(status_code=404, detail="Próbka testowa nie została znaleziona.")
    path = BASE_DIR / items[item_id]["filepath"]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Plik {path} nie istnieje.")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/simulation/inspect")
def inspect_simulation_item(
    item_id: int = Form(-1),
    model_name: str = Form("resnet18"),
    category_filter: str = Form("all")
):
    """
    Executes inspection on a test item (specified or random) and returns
    probabilities alongside Ground Truth for real-time accuracy evaluation.
    Supports filtering by category: 'all', 'def' (only defects), 'ok' (only valid), 'realistic' (~10% defects).
    """
    items = get_test_items()
    if not items:
        raise HTTPException(status_code=404, detail="Brak próbek testowych.")

    if item_id < 0 or item_id >= len(items):
        cat = category_filter.lower().strip()
        if cat == "def":
            candidates = [it for it in items if it["ground_truth"] == "def"]
        elif cat == "ok":
            candidates = [it for it in items if it["ground_truth"] == "ok"]
        elif cat.startswith("realistic") or cat.startswith("ratio_"):
            try:
                if "_" in cat:
                    val = float(cat.split("_")[1])
                    defect_rate = val / 100.0 if val > 1.0 else val
                else:
                    defect_rate = 0.10
            except Exception:
                defect_rate = 0.10
            if random.random() < defect_rate:
                candidates = [it for it in items if it["ground_truth"] == "def"]
            else:
                candidates = [it for it in items if it["ground_truth"] == "ok"]
        else:
            candidates = items

        item = random.choice(candidates) if candidates else random.choice(items)
    else:
        item = items[item_id]

    image_path = BASE_DIR / item["filepath"]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Nie odnaleziono pliku {image_path}")

    image = Image.open(image_path).convert("RGB")
    model = get_or_load_model(model_name)
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    prob_defect, prob_ok, latency_ms, raw_latency_ms = execute_throttled_inference(model, tensor)

    # Base64 encode for instantaneous zero-latency client rendering (prevents visual hiccup)
    try:
        with open(image_path, "rb") as f_img:
            img_bytes = f_img.read()
        image_base64 = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        image_base64 = None

    return {
        "item_id": item["id"],
        "filepath": item["filepath"],
        "filename": item["filename"],
        "ground_truth": item["ground_truth"],
        "ground_truth_label": item["ground_truth_label"],
        "ground_truth_name": item["ground_truth_name"],
        "prob_defect": round(float(prob_defect), 6),
        "prob_ok": round(float(prob_ok), 6),
        "latency_ms": latency_ms,
        "raw_latency_ms": raw_latency_ms,
        "cpu_config": _current_cpu_config,
        "model_name": model_name,
        "device": str(DEVICE),
        "image_base64": image_base64
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)

