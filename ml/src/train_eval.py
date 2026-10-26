"""
Procedury treningowe, walidacyjne i obliczanie metryk inżynierskich:
- Recall (klasa wadliwa)
- Precision (klasa wadliwa)
- F1-Score, Accuracy, Specificity
- ROC-AUC, PR-AUC (Average Precision)
- Macierz pomyłek (TP, FP, TN, FN)
"""

import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Oblicza kompletny zestaw metryk inżynierskich dla binarnej klasyfikacji odlewów.
    Klasa 1 = def_front (wadliwy), Klasa 0 = ok_front (poprawny).
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Obliczenie macierzy pomyłek
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Recall (Czułość) dla klasy wadliwej: TP / (TP + FN)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    # Precision (Precyzja) dla klasy wadliwej: TP / (TP + FP)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    # F1-Score
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    # Ogólna dokładność
    acc = accuracy_score(y_true, y_pred)
    # Specyficzność (Recall dla klasy OK): TN / (TN + FP)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Metryki zależne od prawdopodobieństw ciągłych (ROC-AUC i PR-AUC)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0

    return {
        "threshold": float(threshold),
        "recall": float(rec),
        "precision": float(prec),
        "f1_score": float(f1),
        "accuracy": float(acc),
        "specificity": float(spec),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Trenuje model przez 1 epokę z opcjonalnym automatycznym mieszaniem precyzji (AMP)."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.float().to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(1, total_samples)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Ewaluuje model na zadanym DataLoaderze.
    Zwraca: (średni_loss, słownik_metryk, y_true, y_prob).
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device, non_blocking=True)
            targets_float = targets.float().to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets_float)

            probs = torch.sigmoid(outputs)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / max(1, total_samples)
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    return avg_loss, metrics, y_true, y_prob


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Pełna pętla uczenia z optymalizatorem AdamW, CosineAnnealingLR oraz checkpointem najlepszego modelu.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_score = -1.0
    best_weights = None
    best_metrics = None
    best_epoch = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_recall": [],
        "val_precision": [],
        "val_f1": [],
        "val_accuracy": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_recall"].append(val_metrics["recall"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_f1"].append(val_metrics["f1_score"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        # Jako wskaźnik najlepszego modelu łączymy F1-score i Recall
        score = val_metrics["f1_score"] + val_metrics["recall"]
        if score > best_score:
            best_score = score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics
            best_epoch = epoch

        elapsed = time.time() - t0
        if verbose:
            print(
                f"Epoka [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Recall: {val_metrics['recall']:.4f} | Val Prec: {val_metrics['precision']:.4f} | "
                f"Val F1: {val_metrics['f1_score']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
            )

    total_time = time.time() - start_time

    # Wczytanie najlepszych wag
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
        "total_training_time_sec": total_time,
    }
