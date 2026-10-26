"""
Moduł procedury 5x2-fold cross-validation:
- RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
- Identyczne warunki treningu dla Custom CNN, ResNet18 i MobileNetV2
- Zbieranie metryk w 10 powtórzeniach i analiza statystyczna Diettericha
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold

from src.dataset import scan_dataset, CastingDataset, get_train_transforms, get_eval_transforms
from src.models import build_model
from src.train_eval import train_model, evaluate, compute_metrics
from src.statistical_tests import compare_all_model_pairs


def run_5x2_cross_validation(
    data_dir: str = "casting_data",
    models_to_test: List[str] = ("cnn", "resnet18", "mobilenet_v2"),
    epochs_per_fold: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 4,
    random_state: int = 42,
    output_dir: str = "reports",
    figures_dir: str = "reports/figures",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Wykonuje pełną procedurę 5x2-fold cross-validation dla zdefiniowanych modeli.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device(device_str)
    print(f"Uruchamianie 5x2 CV na urządzeniu: {device}")

    filepaths, labels = scan_dataset(data_dir)
    filepaths_arr = np.array(filepaths)
    labels_arr = np.array(labels)

    # 5 powtórzeń 2-krotnej walidacji krzyżowej
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random_state)

    train_trans = get_train_transforms()
    eval_trans = get_eval_transforms()

    # Struktura na zebrane wyniki
    # cv_data[model][metric] -> lista 5 krotek (fold1, fold2)
    all_fold_logs = []
    cv_records_recall = {m: [] for m in models_to_test}
    cv_records_precision = {m: [] for m in models_to_test}
    cv_records_f1 = {m: [] for m in models_to_test}
    cv_records_acc = {m: [] for m in models_to_test}

    current_repeat = 0
    repeat_fold_scores = {m: {"rec": [], "prec": [], "f1": [], "acc": []} for m in models_to_test}

    start_total_time = time.time()

    for fold_idx, (train_indices, test_indices) in enumerate(rskf.split(filepaths_arr, labels_arr)):
        repeat_num = (fold_idx // 2) + 1
        fold_in_repeat = (fold_idx % 2) + 1

        print(f"\nREPEAT {repeat_num}/5 | FOLD {fold_in_repeat}/2 (Global Fold #{fold_idx + 1}/10)")
        print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")

        # Tworzenie DataLoaderów dla danego folda
        train_ds = CastingDataset(filepaths_arr[train_indices], labels_arr[train_indices], transform=train_trans)
        test_ds = CastingDataset(filepaths_arr[test_indices], labels_arr[test_indices], transform=eval_trans)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        for model_name in models_to_test:
            print(f"\nTrenowanie: {model_name.upper()}")
            torch.manual_seed(random_state + fold_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state + fold_idx)

            # Inicjalizacja modelu
            model = build_model(model_name, pretrained=True, device=device)

            train_res = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                epochs=epochs_per_fold,
                lr=lr,
                device=device,
                verbose=True,
            )

            # Końcowa ewaluacja na test foldzie
            criterion = torch.nn.BCEWithLogitsLoss()
            _, eval_metrics, _, _ = evaluate(model, test_loader, criterion, device)

            fold_record = {
                "repeat": repeat_num,
                "fold": fold_in_repeat,
                "global_fold": fold_idx + 1,
                "model": model_name,
                "recall": eval_metrics["recall"],
                "precision": eval_metrics["precision"],
                "f1_score": eval_metrics["f1_score"],
                "accuracy": eval_metrics["accuracy"],
                "roc_auc": eval_metrics["roc_auc"],
                "pr_auc": eval_metrics["pr_auc"],
                "tp": eval_metrics["tp"],
                "fp": eval_metrics["fp"],
                "tn": eval_metrics["tn"],
                "fn": eval_metrics["fn"],
            }
            all_fold_logs.append(fold_record)

            repeat_fold_scores[model_name]["rec"].append(eval_metrics["recall"])
            repeat_fold_scores[model_name]["prec"].append(eval_metrics["precision"])
            repeat_fold_scores[model_name]["f1"].append(eval_metrics["f1_score"])
            repeat_fold_scores[model_name]["acc"].append(eval_metrics["accuracy"])

            # Jeśli zakończyliśmy parę foldów w powtórzeniu (fold 2)
            if fold_in_repeat == 2:
                pair_rec = (repeat_fold_scores[model_name]["rec"][-2], repeat_fold_scores[model_name]["rec"][-1])
                pair_prec = (repeat_fold_scores[model_name]["prec"][-2], repeat_fold_scores[model_name]["prec"][-1])
                pair_f1 = (repeat_fold_scores[model_name]["f1"][-2], repeat_fold_scores[model_name]["f1"][-1])
                pair_acc = (repeat_fold_scores[model_name]["acc"][-2], repeat_fold_scores[model_name]["acc"][-1])

                cv_records_recall[model_name].append(pair_rec)
                cv_records_precision[model_name].append(pair_prec)
                cv_records_f1[model_name].append(pair_f1)
                cv_records_acc[model_name].append(pair_acc)

    total_cv_time = time.time() - start_total_time
    print(f"\nProcedura 5x2 CV zakończona w czasie: {total_cv_time:.1f} s")

    # Obliczenie statystyk opisowych dla każdego modelu (średnia +/- std)
    df_folds = pd.DataFrame(all_fold_logs)
    summary_stats = {}
    for m in models_to_test:
        sub = df_folds[df_folds["model"] == m]
        summary_stats[m] = {
            "recall_mean": float(sub["recall"].mean()),
            "recall_std": float(sub["recall"].std()),
            "precision_mean": float(sub["precision"].mean()),
            "precision_std": float(sub["precision"].std()),
            "f1_mean": float(sub["f1_score"].mean()),
            "f1_std": float(sub["f1_score"].std()),
            "accuracy_mean": float(sub["accuracy"].mean()),
            "accuracy_std": float(sub["accuracy"].std()),
        }

    # Wykonanie testów statystycznych Diettericha
    recall_stats = compare_all_model_pairs(cv_records_recall, metric_name="Recall (klasa wadliwa)")
    precision_stats = compare_all_model_pairs(cv_records_precision, metric_name="Precision (klasa wadliwa)")

    # Zapis wyników do JSON
    json_path = os.path.join(output_dir, "cv_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "all_folds": all_fold_logs,
            "summary_stats": summary_stats,
            "statistical_tests_recall": recall_stats,
            "statistical_tests_precision": precision_stats,
            "total_cv_time_sec": total_cv_time,
        }, f, indent=2, ensure_ascii=False)

    # Generowanie raportu markdown
    md_report_path = os.path.join(output_dir, "cv_statistical_report.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Wyniki Walidacji Krzyżowej 5x2-fold i Testów Statystycznych Diettericha\n\n")
        f.write("System automatycznej kontroli jakości odlewów\n")
        f.write("Dawid Chudzicki 272715, Piotr Szczerbiak 294722\n\n")
        f.write("## 1. Zestawienie średnich wyników (10 iteracji 5x2 CV)\n\n")
        f.write("| Model | Recall (Czułość) | Precision (Precyzja) | F1-Score | Accuracy |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for m, stats_dict in summary_stats.items():
            f.write(
                f"| **{m.upper()}** | "
                f"{stats_dict['recall_mean']:.4f} ± {stats_dict['recall_std']:.4f} | "
                f"{stats_dict['precision_mean']:.4f} ± {stats_dict['precision_std']:.4f} | "
                f"{stats_dict['f1_mean']:.4f} ± {stats_dict['f1_std']:.4f} | "
                f"{stats_dict['accuracy_mean']:.4f} ± {stats_dict['accuracy_std']:.4f} |\n"
            )
        f.write("\n")

        f.write("## 2. Testy istotności statystycznej (Sparowany test t-Studenta Diettericha, df=5, alpha=0.05)\n\n")
        f.write("### Czułość (Recall - metryka główna detekcji wad):\n\n")
        f.write("| Porównanie par modeli | Średnia różnica | Statystyka t | p-value | Wniosek (H0) |\n")
        f.write("| :--- | :---: | :---: | :---: | :--- |\n")
        for pair_name, t_res in recall_stats.items():
            dt = t_res["dietterich_t_test"]
            rej = "**Odrzucona (istotna różnica)**" if dt["reject_null_hypothesis"] else "Brak podstaw do odrzucenia (b.d.)"
            f.write(f"| **{pair_name}** | {dt['mean_difference']:+.5f} | {dt['t_statistic']:.4f} | {dt['p_value']:.5f} | {rej} |\n")
        f.write("\n")

        f.write("### Precyzja (Precision - fałszywe odrzucenia):\n\n")
        f.write("| Porównanie par modeli | Średnia różnica | Statystyka t | p-value | Wniosek (H0) |\n")
        f.write("| :--- | :---: | :---: | :---: | :--- |\n")
        for pair_name, t_res in precision_stats.items():
            dt = t_res["dietterich_t_test"]
            rej = "**Odrzucona (istotna różnica)**" if dt["reject_null_hypothesis"] else "Brak podstaw do odrzucenia (b.d.)"
            f.write(f"| **{pair_name}** | {dt['mean_difference']:+.5f} | {dt['t_statistic']:.4f} | {dt['p_value']:.5f} | {rej} |\n")
        f.write("\n")

    # Wykres porównawczy: Boxplot metryk w 10 iteracjach
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_to_plot = [("recall", "Recall (Czułość)"), ("precision", "Precision (Precyzja)"), ("f1_score", "F1-Score")]
    colors = ["#3498db", "#2ecc71", "#e67e22"]

    for ax, (met_key, met_label), col in zip(axes, metrics_to_plot, colors):
        sns.boxplot(x="model", y=met_key, data=df_folds, ax=ax, palette="Set2")
        sns.stripplot(x="model", y=met_key, data=df_folds, ax=ax, color="black", alpha=0.6, jitter=0.2)
        ax.set_title(met_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("Wartość")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    box_fig_path = os.path.join(figures_dir, "cv_5x2_metrics_boxplot.png")
    fig.savefig(box_fig_path, dpi=200)
    plt.close(fig)

    print(f"Zapisano raport walidacji do: {md_report_path}")
    print(f"Zapisano wykres boxplot do: {box_fig_path}")

    return {
        "summary_stats": summary_stats,
        "statistical_tests_recall": recall_stats,
        "statistical_tests_precision": precision_stats,
        "json_path": json_path,
        "report_path": md_report_path,
        "boxplot_figure": box_fig_path,
    }
