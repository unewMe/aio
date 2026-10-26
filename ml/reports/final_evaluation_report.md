# Raport Końcowy Ewaluacji Systemu Kontroli Jakości Odlewów

Dawid Chudzicki 272715, Piotr Szczerbiak 294722

## 1. Zestawienie metryk inżynierskich i weryfikacja celów

| Metryka / Cecha | Proponowany cel | Custom CNN | ResNet18 | MobileNetV2 |
| :--- | :---: | :---: | :---: | :---: |
| **Recall (klasa wadliwa)** | **≥ 0,98** | 1.0000 | 1.0000 | 1.0000 |
| **Precision (klasa wadliwa)** | **≥ 0,90** | 0.9772 | 0.9950 | 0.9820 |
| **F1-Score** | raportowany | 0.9885 | 0.9975 | 0.9909 |
| **Accuracy** | raportowany | 0.9883 | 0.9975 | 0.9908 |
| **ROC-AUC** | raportowany | 0.9999 | 1.0000 | 1.0000 |
| **PR-AUC** | raportowany | 0.9999 | 1.0000 | 1.0000 |
| **Czas inferencji CPU / obr.** | **≤ 100 ms** | 34.45 ms | 62.52 ms | 59.22 ms |
| **Czas inferencji GPU / obr.** | informacyjny | N/A | N/A | N/A |
| **Rozmiar modelu** | raportowany | 1.62 MB | 42.67 MB | 8.62 MB |
| **Liczba parametrów** | raportowany | 422,177 | 11,177,025 | 2,225,153 |
| **Optymalny próg $\tau^*$** | strojenie PR | 0.158 | 0.197 | 0.099 |

## 2. Analiza macierzy pomyłek i błędów krytycznych (FN vs FP)

W kontroli jakości odlewów najważniejsza jest eliminacja błędów **False Negative (FN)**, czyli wadliwych odlewów dopuszczonych omyłkowo do dystrybucji jako dobre.

| Model | False Negatives (Wady przeoczone) | False Positives (Fałszywe alarmy) | True Positives | True Negatives |
| :--- | :---: | :---: | :---: | :---: |
| **Custom CNN** | **0** | 14 | 600 | 586 |
| **ResNet18** | **0** | 3 | 600 | 597 |
| **MobileNetV2** | **0** | 11 | 600 | 589 |

## 3. Wnioski wdrożeniowe

1. **Najlepsza czułość i ogólna jakość:** Architektury ResNet18 i MobileNetV2 (dzięki transfer learning z wag ImageNet) osiągają znakomitą zbieżność i spełniają wyśrubowany cel Recall >= 0.98.
2. **Efektywność brzegowa (Edge AI):** MobileNetV2 charakteryzuje się rozmiarem jedynie ~8.6 MB i bardzo niskim czasem inferencji na CPU, co czyni go optymalnym wyborem dla instalacji na liniach produkcyjnych.
3. **Strojenie progu decyzyjnego:** Obniżenie progu klasyfikacji z domyślnego 0.5 pozwala zagwarantować Recall >= 0.98 przy zachowaniu precyzji przekraczającej 0.90.
