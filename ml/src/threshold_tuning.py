"""
Strojenie progu decyzyjnego klasyfikatora (Decision Threshold Tuning)
oraz analiza krzywych Precision-Recall w kontekście kryteriów przemysłowych:
- Cel: Recall (klasa wadliwa) >= 0.98
- Ograniczenie liczby fałszywych alarmów: Precision >= 0.90
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from src.train_eval import compute_metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.98,
    min_precision: float = 0.90,
    num_thresholds: int = 200,
) -> Dict[str, Any]:
    """
    Znajduje optymalny próg decyzyjny spełniający kryteria przemysłowe:
    1. Czułość (Recall) >= min_recall (np. 0.98)
    2. Maksymalizacja Precyzji (Precision) lub osiągnięcie >= min_precision
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    candidates = []

    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=t)
        # F2-score faworyzuje czułość dwukrotnie bardziej niż precyzję
        p, r = m["precision"], m["recall"]
        f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0
        m["f2_score"] = float(f2)
        candidates.append(m)

    # 1. Filtrujemy te, które spełniają oba kryteria
    both_satisfied = [c for c in candidates if c["recall"] >= min_recall and c["precision"] >= min_precision]

    if both_satisfied:
        # Wybieramy próg o najwyższym F1 / F2 score
        best = max(both_satisfied, key=lambda x: (x["recall"], x["precision"]))
        status = "Spełnione oba kryteria (Recall >= 0.98 oraz Precision >= 0.90)"
    else:
        # 2. Jeśli precyzja nie osiąga 0.90, priorytetyzujemy bezwzględnie Recall >= min_recall
        recall_satisfied = [c for c in candidates if c["recall"] >= min_recall]
        if recall_satisfied:
            best = max(recall_satisfied, key=lambda x: x["precision"])
            status = f"Spełnione kryterium Recall >= {min_recall:.2f} (maksymalizacja precyzji: {best['precision']:.4f})"
        else:
            # 3. Jeśli nie osiągnięto 0.98, wybieramy najwyższy Recall
            best = max(candidates, key=lambda x: (x["recall"], x["f1_score"]))
            status = f"Najwyższy możliwy Recall: {best['recall']:.4f}"

    return {
        "status": status,
        "optimal_threshold": round(best["threshold"], 4),
        "metrics_at_optimal": best,
        "metrics_at_default_0_5": compute_metrics(y_true, y_prob, threshold=0.5),
        "all_candidates": candidates,
    }


def plot_threshold_tuning_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_path: Optional[str] = None,
    min_recall: float = 0.98,
    min_precision: float = 0.90,
) -> Dict[str, Any]:
    """
    Rysuje i zapisuje wykres krzywych Precision, Recall, F1 oraz F2 w funkcji progu decyzyjnego.
    Zaznacza optymalny próg oraz domyślny próg 0.5.
    """
    res = find_optimal_threshold(y_true, y_prob, min_recall=min_recall, min_precision=min_precision)
    candidates = res["all_candidates"]
    opt_t = res["optimal_threshold"]

    thresh_vals = [c["threshold"] for c in candidates]
    rec_vals = [c["recall"] for c in candidates]
    prec_vals = [c["precision"] for c in candidates]
    f1_vals = [c["f1_score"] for c in candidates]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresh_vals, rec_vals, label="Recall (Czułość - wykrywanie wad)", color="#e74c3c", linewidth=2.2)
    ax.plot(thresh_vals, prec_vals, label="Precision (Precyzja - fałszywe alarmy)", color="#3498db", linewidth=2.2)
    ax.plot(thresh_vals, f1_vals, label="F1-Score", color="#2ecc71", linestyle="--", linewidth=1.8)

    # Poziome linie progów docelowych
    ax.axhline(min_recall, color="#e74c3c", linestyle=":", alpha=0.7, label=f"Cel Recall ({min_recall})")
    ax.axhline(min_precision, color="#3498db", linestyle=":", alpha=0.7, label=f"Cel Precision ({min_precision})")

    # Pionowa linia optymalnego progu
    ax.axvline(opt_t, color="#9b59b6", linestyle="-", linewidth=2, label=f"Optymalny próg $\\tau^* = {opt_t:.3f}$")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6, label="Domyślny próg $\\tau = 0.50$")

    ax.set_title(f"Strojenie progu decyzyjnego: {model_name}\n({res['status']})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Próg decyzyjny $\\tau$", fontsize=11)
    ax.set_ylabel("Wartość metryki", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return res
