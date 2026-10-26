#!/usr/bin/env python3
"""
Skrypt uruchomieniowy procedury 5x2-fold Cross-Validation oraz testów statystycznych Diettericha.
Użycie:
    python3 run_cross_validation.py --epochs 5 --batch_size 64
"""

import argparse
from src.cross_validation_5x2 import run_5x2_cross_validation

def main():
    parser = argparse.ArgumentParser(description="Procedura 5x2 CV dla modeli inspekcji odlewów.")
    parser.add_argument("--data_dir", type=str, default="casting_data", help="Katalog ze zbiorem danych.")
    parser.add_argument("--epochs", type=int, default=5, help="Liczba epok na każdy fold (domyślnie 5).")
    parser.add_argument("--batch_size", type=int, default=64, help="Rozmiar batcha (domyślnie 64).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (domyślnie 1e-4).")
    parser.add_argument("--num_workers", type=int, default=4, help="Liczba wątków DataLoadera.")
    parser.add_argument("--seed", type=int, default=42, help="Podstawowe ziarno losowości.")
    args = parser.parse_args()

    print("Rozpoczynanie procedury 5x2-fold Cross-Validation")
    results = run_5x2_cross_validation(
        data_dir=args.data_dir,
        epochs_per_fold=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        random_state=args.seed
    )
    print("Procedura 5x2 CV i testy statystyczne zakończone!")
    print("Średnie wyniki:")
    for m, stat in results["summary_stats"].items():
        print(f"  {m.upper():12} -> Recall: {stat['recall_mean']:.4f} ± {stat['recall_std']:.4f} | Precision: {stat['precision_mean']:.4f} ± {stat['precision_std']:.4f}")

if __name__ == "__main__":
    main()
