"""
Benchmark czasu inferencji (CPU i GPU) oraz rozmiaru modeli.
Weryfikacja kryterium inżynierskiego:
Czas inferencji / obraz <= 100 ms (CPU)
"""

import time
import os
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from src.models import build_model, count_parameters, get_model_size_mb


def benchmark_inference_latency(
    model: nn.Module,
    device_str: str = "cpu",
    input_shape: tuple = (1, 3, 300, 300),
    warmup_runs: int = 15,
    test_runs: int = 100,
) -> Dict[str, Any]:
    """
    Mierzy opóźnienie pojedynczej inferencji (batch_size=1) na CPU lub GPU.
    Zwraca średni czas, odchylenie standardowe, medianę i 95. percentyl w milisekundach (ms).
    """
    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    # Warm-up (rozgrzewka)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    times_ms = []

    with torch.no_grad():
        for _ in range(test_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms)
    mean_lat = float(np.mean(times_ms))
    std_lat = float(np.std(times_ms))
    median_lat = float(np.median(times_ms))
    p95_lat = float(np.percentile(times_ms, 95))

    target_met = mean_lat <= 100.0 if device_str == "cpu" else True

    return {
        "device": device_str,
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "target_met_le_100ms": target_met,
    }


def benchmark_all_models(
    models_dict: Optional[Dict[str, nn.Module]] = None,
    include_gpu: bool = True
) -> Dict[str, Any]:
    """
    Przeprowadza kompleksowy benchmark dla Custom CNN, ResNet18 i MobileNetV2.
    """
    results = {}

    if models_dict is None:
        models_dict = {
            "Custom CNN": build_model("cnn", pretrained=False),
            "ResNet18": build_model("resnet18", pretrained=False),
            "MobileNetV2": build_model("mobilenet_v2", pretrained=False),
        }

    for name, model in models_dict.items():
        total_p, train_p = count_parameters(model)
        size_mb = get_model_size_mb(model)

        cpu_bench = benchmark_inference_latency(model, device_str="cpu")
        gpu_bench = None
        if include_gpu and torch.cuda.is_available():
            gpu_bench = benchmark_inference_latency(model, device_str="cuda")

        results[name] = {
            "total_params": total_p,
            "trainable_params": train_p,
            "size_mb": size_mb,
            "cpu_latency": cpu_bench,
            "gpu_latency": gpu_bench,
        }

    return results
