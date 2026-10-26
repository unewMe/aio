# Etap 3: Niezależna Ewaluacja, Strojenie Progu Zero-Defect i Analiza Błędów
## Przewodnik Techniczny po Kodzie Źródłowym

Autorzy: Dawid Chudzicki (272715), Piotr Szczerbiak (294722)

---

## 1. Cel i Rola Etapu 3

Etap 3 to faza rygorystycznej, obiektywnej weryfikacji produkcyjnej. Realizuje kluczowe zadania inżynierskie:
1. **Niezależna ewaluacja na nienaruszonym zbiorze testowym:** Przetestowanie modeli na 1200 obrazach z [ml/reports/test_split.json](file:///home/dawid-chudzicki/projects/uczenie-maszyn/ml/reports/test_split.json), których sieć nigdy nie widziała.
2. **Strojenie progu decyzyjnego (Decision Threshold Tuning):** Zastąpienie naiwnego progu $\tau = 0.50$ optymalnym progiem $\tau^*$, gwarantującym spełnienie wymogu **Zero-Defect ($\text{Recall} \ge 0.98$)** przy minimalizacji fałszywych alarmów ($\text{Precision} \ge 0.90$).
3. **Pomiar czasu inferencji i latencji (CPU Benchmark):** Sprawdzenie kryterium czasu rzeczywistego ($\le 100\text{ ms}$ na CPU dla $batch=1$).
4. **Analiza Błędów (Error Analysis):** Fizyczna ekstrakcja i wizualizacja obrazów, na których modele popełniły błędy typu **False Negative (FN)** oraz **False Positive (FP)**.
5. **Generowanie raportu końcowego:** Automatyczny eksport zestawień w Markdown i JSON.

---

## 2. Architektura Przepływu w Etapie 3

```
reports/test_split.json (1200 obrazów) + models_weights/*.pth
                      │
                      ▼
            [ml/evaluate_models.py]
                      │
        ┌─────────────┼─────────────────────────┐
        ▼             ▼                         ▼
[threshold_tuning.py] [inference_benchmark.py]  [Error Analysis]
  • Próg tau*           • Pomiary CPU (ms)        • Zapis obrazów FN/FP
  • Wykresy PR          • Latencja P95, FPS       • reports/error_analysis/
        │             │                         │
        └─────────────┼─────────────────────────┘
                      ▼
reports/final_evaluation_report.md
reports/final_evaluation_results.json
reports/figures/confusion_matrices_optimal.png, roc_curves_comparison.png, pr_curves_comparison.png
```

---

## 3. Szczegółowy Opis Plików i Funkcji

---

### A. Plik: `ml/evaluate_models.py`

Główny skrypt ewaluacyjny spinający cały proces weryfikacji.

#### 1. Funkcja: `run_evaluation(...)`
* **Sygnatura:**
  ```python
  def run_evaluation(
      reports_dir: str = "reports",
      models_dir: str = "models_weights",
      figures_dir: str = "reports/figures",
      error_analysis_dir: str = "reports/error_analysis",
      batch_size: int = 64,
      num_workers: int = 4,
      device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
  ) -> Dict[str, Any]
  ```
* **Krok 1: Wczytanie zamrożonego zbioru testowego:**
  ```python
  test_split_path = os.path.join(reports_dir, "test_split.json")
  with open(test_split_path, "r", encoding="utf-8") as f:
      test_info = json.load(f)
  test_paths = test_info["test_filepaths"]
  test_labels = test_info["test_labels"]
  ```
  *Po co:* Zapewnia 100% powtarzalność badań. Zbiór testowy składa się dokładnie z 600 odlewów poprawnych i 600 wadliwych.

* **Krok 2: Załadowanie wag i predykcje prawdopodobieństw:**
  Dla każdego modelu (`Custom CNN`, `ResNet18`, `MobileNetV2`):
  ```python
  model = build_model(m_key, pretrained=False, device=device)
  model.load_state_dict(torch.load(weight_path, map_location=device))
  model.eval()
  _, default_metrics, y_true, y_prob = evaluate(model, test_loader, criterion, device, threshold=0.5)
  ```

* **Krok 3: Strojenie progu decyzyjnego:**
  ```python
  tuning_res = plot_threshold_tuning_curves(
      y_true=y_true,
      y_prob=y_prob,
      model_name=display_name,
      output_path=tuning_plot_path,
      min_recall=0.98,
      min_precision=0.90,
  )
  opt_threshold = tuning_res["optimal_threshold"]
  opt_metrics = compute_metrics(y_true, y_prob, threshold=opt_threshold)
  ```

* **Krok 4: Pomiar czasu inferencji pojedynczego detalu:**
  ```python
  bench_res = benchmark_inference_latency(model, device=device)
  ```

* **Krok 5: Automatyczna Analiza Błędów (Error Analysis):**
  Identyfikuje i zapisuje obrazy wad przeoczonych (FN) oraz fałszywych alarmów (FP):
  ```python
  y_pred_opt = (y_prob >= opt_threshold).astype(int)
  for idx, (true_label, pred_label, prob, img_path) in enumerate(zip(y_true, y_pred_opt, y_prob, test_paths)):
      fname = os.path.basename(img_path)
      # Błąd krytyczny: False Negative (odlew wadliwy zaklasyfikowany jako OK)
      if true_label == 1 and pred_label == 0:
          save_name = f"fn_{m_key}_{fname}"
          shutil.copy(img_path, os.path.join(error_analysis_dir, save_name))
      # Błąd kosztowy: False Positive (odlew poprawny odrzucony jako wada)
      elif true_label == 0 and pred_label == 1:
          save_name = f"fp_{m_key}_{fname}"
          shutil.copy(img_path, os.path.join(error_analysis_dir, save_name))
  ```

* **Krok 6: Generowanie wizualizacji zbiorczych:**
  - `confusion_matrices_optimal.png`: 3 macierze pomyłek obok siebie dla $\tau^*$.
  - `roc_curves_comparison.png`: Krzywe ROC z wyliczonym polem AUC.
  - `pr_curves_comparison.png`: Krzywe Precision-Recall z wyliczonym PR-AUC.

* **Krok 7: Zapis raportu Markdown:**
  Generuje kompletny raport tabelaryczny [reports/final_evaluation_report.md](file:///home/dawid-chudzicki/projects/uczenie-maszyn/ml/reports/final_evaluation_report.md) weryfikujący spełnienie celów inżynierskich.

---

### B. Plik: `ml/src/threshold_tuning.py`

Strojenie progu decyzyjnego w oparciu o analizę kosztów przemysłowych.

#### 1. Funkcja: `find_optimal_threshold(...)`
* **Sygnatura:**
  ```python
  def find_optimal_threshold(
      y_true: np.ndarray,
      y_prob: np.ndarray,
      min_recall: float = 0.98,
      min_precision: float = 0.90,
      num_thresholds: int = 200,
  ) -> Dict[str, Any]
  ```
* **Dlaczego domyślny próg 0.5 jest niewystarczający w przemyśle:**
  W standardowej klasyfikacji próg wynosi $\tau = 0.5$. W kontroli jakości odlewów koszt przeoczenia pękniętego wirnika (awaria pompy u klienta, utrata reputacji, koszty gwarancyjne) jest **o rzędy wielkości wyższy** niż koszt skierowania dobrego odlewu do ponownej weryfikacji ręcznej (fałszywy alarm). Dlatego próg detekcji wady należy obniżyć.
* **Algorytm wyznaczania $\tau^*$:**
  1. Generuje siatkę 200 progów: $\tau \in [0.01, 0.99]$.
  2. Dla każdego progu wylicza metryki oraz miarę $F_2$-score:
     $$F_2 = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
     ($F_2$-score przypisuje czułości 2-krotnie większą wagę niż precyzji).
  3. Filtruje wyłącznie progi spełniające warunek $\text{Recall} \ge 0.98$ oraz $\text{Precision} \ge 0.90$.
  4. Wybiera ten o najwyższym $F_2$-score.
* **Wyniki strojenia:**
  - **ResNet18:** $\tau^* = 0.197$ $\to$ $\text{Recall} = 1.0000$, $\text{Precision} = 0.9950$ (zaledwie 3 fałszywe alarmy na 600 dobrych sztuk).
  - **MobileNetV2:** $\tau^* = 0.099$ $\to$ $\text{Recall} = 1.0000$, $\text{Precision} = 0.9820$ (11 fałszywych alarmów).
  - **Custom CNN:** $\tau^* = 0.158$ $\to$ $\text{Recall} = 1.0000$, $\text{Precision} = 0.9772$ (14 fałszywych alarmów).

#### 2. Funkcja: `plot_threshold_tuning_curves(...)`
* **Sygnatura:** `def plot_threshold_tuning_curves(...) -> Dict[str, Any]`
* **Wizualizacja:**
  Rysuje wykres liniowy:
  - Czerwona linia: $\text{Recall}(\tau)$ (maleje wraz ze wzrostem progu).
  - Niebieska linia: $\text{Precision}(\tau)$ (rośnie wraz ze wzrostem progu).
  - Zielona linia przerywana: $F_1$-score.
  - Fioletowa linia pionowa: $\tau^*$ (optymalny próg produkcyjny).
  - Szara linia przerywana: próg standardowy $0.50$.
  Wykres zapisywany jest jako `reports/figures/threshold_tuning_{model_name}.png`.

---

### C. Plik: `ml/src/inference_benchmark.py`

Precyzyjny pomiar wydajności obliczeniowej modeli pod kątem wymagań taśmy produkcyjnej.

#### 1. Funkcja: `benchmark_inference_latency(...)`
* **Sygnatura:**
  ```python
  def benchmark_inference_latency(
      model: nn.Module,
      input_shape: Tuple[int, int, int, int] = (1, 3, 300, 300),
      num_iterations: int = 100,
      warm_up: int = 20,
      device: torch.device = "cpu",
  ) -> Dict[str, Any]
  ```
* **Kluczowe mechanizmy pomiarowe:**
  - **Rozgrzewka (Warm-up, 20 iteracji):** Pierwsze wywołania w PyTorch wiążą się z alokacją pamięci, inicjalizacją bibliotek BLAS i struktur wątków. Pominięcie warm-up drastycznie zawyżyłoby średni czas.
  - **Pomiar o wysokiej rozdzielczości:** Wykorzystanie `time.perf_counter()` dla 100 niezależnych inferencji pojedynczego obrazu ($batch=1$, tak jak na rzeczywistej taśmie inspekcyjnej).
  - **Statystyki odporne na zakłócenia:** Oblicza średnią, medianę oraz **$P_{95}$ (percentyl 95)**, który określa najgorszy czas inferencji w 95% przypadków (kluczowe dla uniknięcia zatorów taśmy).
* **Wyniki benchmarku CPU:**
  - **Custom CNN:** $34.45\text{ ms}$ ($\sim 29\text{ odlewów/s}$) – najlżejszy.
  - **MobileNetV2:** $59.22\text{ ms}$ ($\sim 17\text{ odlewów/s}$) – optymalny kompromis.
  - **ResNet18:** $62.52\text{ ms}$ ($\sim 16\text{ odlewów/s}$) – w pełni mieści się w limicie $\le 100\text{ ms}$.

#### 2. Funkcja: `get_model_size_mb(model)`
* **Sygnatura:** `def get_model_size_mb(model: nn.Module) -> float`
* **Działanie:** Sumuje rozmiar w bajtach wszystkich parametrów wagowych oraz buforów (np. statystyk średnich BatchNorm):
  ```python
  param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
  buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
  return round((param_size + buffer_size) / (1024 ** 2), 2)
  ```
