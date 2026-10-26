#!/usr/bin/env python3
"""
Skrypt uruchomieniowy wstępnej analizy i weryfikacji zbioru danych (EDA).
Użycie:
    python3 run_eda.py --data_dir casting_data
"""

import argparse
from src.eda import run_full_eda

def main():
    parser = argparse.ArgumentParser(description="Weryfikacja jakości i EDA zbioru danych odlewów.")
    parser.add_argument("--data_dir", type=str, default="casting_data", help="Ścieżka do katalogu ze zbiorem danych.")
    parser.add_argument("--output_dir", type=str, default="reports", help="Katalog wyjściowy dla raportów.")
    parser.add_argument("--figures_dir", type=str, default="reports/figures", help="Katalog wyjściowy dla wykresów.")
    args = parser.parse_args()

    print("Rozpoczęcie analizy EDA")
    results = run_full_eda(data_dir=args.data_dir, output_dir=args.output_dir, figures_dir=args.figures_dir)
    print("Analiza EDA zakończona pomyślnie!")
    print(f"Wyniki: {results['total_images']} obrazów, uszkodzonych: {results['corrupted_count']}")

if __name__ == "__main__":
    main()
