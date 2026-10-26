#!/usr/bin/env python3
"""
Trening końcowy trzech modeli (Custom CNN, ResNet18, MobileNetV2)
na podziale 80% train / 20% test z zapisem wag do katalogu 'models_weights/'.
Użycie:
    python3 train_final_models.py --epochs 10 --batch_size 64
"""

import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import scan_dataset, CastingDataset, get_train_transforms, get_eval_transforms
from src.models import build_model
from src.train_eval import train_model, evaluate, compute_metrics


def train_final_models(
    data_dir: str = "casting_data",
    models_dir: str = "models_weights",
    reports_dir: str = "reports",
    figures_dir: str = "reports/figures",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    test_size: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device(device_str)
    print(f"Rozpoczynanie treningu modeli końcowych na urządzeniu: {device}")

    filepaths, labels = scan_dataset(data_dir)
    filepaths_arr = np.array(filepaths)
    labels_arr = np.array(labels)

    # Podział Stratyfikowany 80/20 (Train / Test)
    train_idx, test_idx = train_test_split(
        np.arange(len(labels_arr)),
        test_size=test_size,
        random_state=seed,
        stratify=labels_arr
    )

    print(f"Podział danych: Trening={len(train_idx)} obrazów, Zbiór testowy={len(test_idx)} obrazów.")

    # Zapis indeksów i ścieżek zbioru testowego do zachowania powtarzalności ewaluacji
    test_split_info = {
        "test_filepaths": filepaths_arr[test_idx].tolist(),
        "test_labels": labels_arr[test_idx].tolist(),
        "seed": seed,
        "test_size": test_size,
    }
    with open(os.path.join(reports_dir, "test_split.json"), "w", encoding="utf-8") as f:
        json.dump(test_split_info, f, indent=2)

    train_trans = get_train_transforms()
    eval_trans = get_eval_transforms()

    train_ds = CastingDataset(filepaths_arr[train_idx], labels_arr[train_idx], transform=train_trans)
    test_ds = CastingDataset(filepaths_arr[test_idx], labels_arr[test_idx], transform=eval_trans)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    models_config = [
        ("cnn", "custom_cnn.pth", "Custom CNN (Autorska sieć)"),
        ("resnet18", "resnet18.pth", "ResNet18 (Transfer Learning)"),
        ("mobilenet_v2", "mobilenet_v2.pth", "MobileNetV2 (Transfer Learning)"),
    ]

    all_histories = {}
    test_results = {}

    for model_name, save_filename, display_name in models_config:
        print(f"\nTrening modelu: {display_name}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = build_model(model_name, pretrained=True, device=device)

        train_res = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            verbose=True,
        )

        # Zapis wag
        save_path = os.path.join(models_dir, save_filename)
        torch.save(model.state_dict(), save_path)
        print(f"Zapisano wagi do: {save_path} (Rozmiar: {os.path.getsize(save_path) / (1024**2):.2f} MB)")

        # Ewaluacja końcowa na zbiorze testowym
        criterion = torch.nn.BCEWithLogitsLoss()
        _, final_metrics, y_true, y_prob = evaluate(model, test_loader, criterion, device)

        test_results[model_name] = final_metrics
        all_histories[model_name] = train_res["history"]

        print(
            f"Wyniki testowe {display_name}: "
            f"Recall={final_metrics['recall']:.4f}, "
            f"Precision={final_metrics['precision']:.4f}, "
            f"F1={final_metrics['f1_score']:.4f}, "
            f"Accuracy={final_metrics['accuracy']:.4f}"
        )

    # Zapis historii treningu do JSON
    with open(os.path.join(reports_dir, "final_train_history.json"), "w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2)

    # Wykresy przebiegu treningu
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Krzywe uczenia modeli końcowych na zbiorze odlewów", fontsize=14, fontweight="bold")

    colors = {"cnn": "#e74c3c", "resnet18": "#2980b9", "mobilenet_v2": "#27ae60"}
    labels_map = {"cnn": "Custom CNN", "resnet18": "ResNet18", "mobilenet_v2": "MobileNetV2"}

    # Train Loss
    for m_key, hist in all_histories.items():
        axes[0, 0].plot(range(1, epochs + 1), hist["train_loss"], label=labels_map[m_key], color=colors[m_key], linewidth=2)
    axes[0, 0].set_title("Funkcja straty (Train Loss)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoka")
    axes[0, 0].set_ylabel("BCE Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Val Loss
    for m_key, hist in all_histories.items():
        axes[0, 1].plot(range(1, epochs + 1), hist["val_loss"], label=labels_map[m_key], color=colors[m_key], linewidth=2)
    axes[0, 1].set_title("Funkcja straty walidacyjnej (Val Loss)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoka")
    axes[0, 1].set_ylabel("BCE Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Val Recall
    for m_key, hist in all_histories.items():
        axes[1, 0].plot(range(1, epochs + 1), hist["val_recall"], label=labels_map[m_key], color=colors[m_key], linewidth=2)
    axes[1, 0].axhline(0.98, color="red", linestyle=":", label="Wymagany cel Recall (0.98)")
    axes[1, 0].set_title("Czułość (Recall - wykrywanie wad)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoka")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Val Precision
    for m_key, hist in all_histories.items():
        axes[1, 1].plot(range(1, epochs + 1), hist["val_precision"], label=labels_map[m_key], color=colors[m_key], linewidth=2)
    axes[1, 1].axhline(0.90, color="blue", linestyle=":", label="Docelowy cel Precision (0.90)")
    axes[1, 1].set_title("Precyzja (Precision - fałszywe alarmy)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoka")
    axes[1, 1].set_ylabel("Precision")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    curve_fig_path = os.path.join(figures_dir, "training_curves.png")
    fig.savefig(curve_fig_path, dpi=200)
    plt.close(fig)

    print(f"\nTrening zakończony pomyślnie. Zapisano wykresy do: {curve_fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Trening ostatecznych modeli do wdrożenia.")
    parser.add_argument("--data_dir", type=str, default="casting_data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train_final_models(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
