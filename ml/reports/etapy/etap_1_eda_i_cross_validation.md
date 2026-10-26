# Etap 1: Eksploracja Danych (EDA) oraz Walidacja Krzyżowa 5x2 z Testami Statystycznymi
## Przewodnik Techniczny po Kodzie Źródłowym

Autorzy: Dawid Chudzicki (272715), Piotr Szczerbiak (294722)

---

## 1. Cel i Rola Etapu 1

Etap 1 to faza analityczno-badawcza projektu. Odpowiada za:
1. **Weryfikację integralności i jakości danych** (wykrywanie uszkodzeń, badanie balansu klas, eliminacja duplikatów).
2. **Rygorystyczne porównanie naukowe 3 architektur** (Custom CNN, ResNet18, MobileNetV2) przy pomocy protokołu **5x2-fold Cross-Validation**.
3. **Statystyczną weryfikację hipotez** (sparowany test t-Studenta Diettericha oraz test ze skorygowaną wariancją Nadeau-Bengio), aby dowieść, czy przewaga modeli pretrenowanych jest istotna statystycznie ($p < 0.05$), a nie jest efektem losowego podziału danych.

---

## 2. Architektura Przepływu w Etapie 1

```
casting_data/
  │
  ├──► [ml/run_eda.py] ────────► [ml/src/eda.py] ────────────────► reports/eda_report.md
  │                                                                reports/figures/eda_*.png
  │
  └──► [ml/run_cross_validation.py]
            │
            ├──► [ml/src/cross_validation_5x2.py] ───────────────► 10 foldów treningu/ewaluacji
            │         ▲                │
            │         │                ▼
            │   [ml/src/models.py]  [ml/src/statistical_tests.py]─► Paired t-test Diettericha (df=5)
            │   [ml/src/dataset.py]                                 Nadeau-Bengio corrected test
```

---

## 3. Szczegółowy Opis Plików i Funkcji

---

### A. Plik: `ml/run_eda.py`

Główny skrypt uruchomieniowy CLI dla eksploracyjnej analizy danych.

#### Funkcja: `main()`
* **Sygnatura:** `def main() -> None`
* **Skąd wywoływana:** Bezpośrednio z wiersza poleceń: `python3 run_eda.py --data_dir casting_data`.
* **Co wywołuje pod spodem:** `src.eda.run_eda(data_dir=args.data_dir)`.
* **Kod funkcji:**
  ```python
  def main():
      parser = argparse.ArgumentParser(description="EDA dla zbioru inspekcji odlewów.")
      parser.add_argument("--data_dir", type=str, default="casting_data", help="Katalog ze zbiorem danych.")
      args = parser.parse_args()

      print("Rozpoczynanie eksploracyjnej analizy danych (EDA)...")
      results = run_eda(data_dir=args.data_dir)
      print("EDA zakończone pomyślnie!")
  ```
* **Po co istnieje:** Zapewnia wygodny interfejs CLI z możliwością parametryzacji ścieżki do danych.

---

### B. Plik: `ml/src/eda.py`

Biblioteka implementująca algorytmy sprawdzania poprawności, wykrywania duplikatów i generowania wykresów statystycznych.

#### 1. Funkcja: `check_image_integrity(filepaths)`
* **Sygnatura:** `def check_image_integrity(filepaths: List[str]) -> Dict[str, Any]`
* **Cel inżynierski:** Zabezpieczenie potoku treningowego przed błędami I/O (uszkodzone nagłówki JPEG, pliki zerowej długości, niezgodne wymiary lub nieprawidłowe formaty kanałów).
* **Parametry wejściowe:** `filepaths` (lista ścieżek bezwzględnych do wszystkich plików w `casting_data/`).
* **Zwraca:** Słownik zawierający listę uszkodzonych plików, unikalne wymiary obrazów, tryby kolorów (np. RGB) oraz statystyki pikseli (min, max, mean).
* **Kluczowy fragment kodu:**
  ```python
  for p in filepaths:
      try:
          with Image.open(p) as img:
              img.verify()  # Weryfikacja integralności struktury pliku
          with Image.open(p) as img:
              sizes.add(img.size)
              modes.add(img.mode)
              arr = np.array(img, dtype=np.float32)
              pixel_means.append(arr.mean())
      except Exception as e:
          corrupted.append((p, str(e)))
  ```
* **Dlaczego to robimy:** W systemach wizyjnych uszkodzony plik może przerwać wielogodzinny proces uczenia w połowie epoki.

#### 2. Funkcja: `detect_duplicates(filepaths)`
* **Sygnatura:** `def detect_duplicates(filepaths: List[str]) -> Tuple[Dict[str, List[str]], Dict[int, List[str]]]`
* **Cel inżynierski:** Wykrycie wycieku danych (*data leakage*) pomiędzy foldami.
* **Metodyka:**
  1. **Dokładne duplikaty (MD5):** Haszowanie binarne zawartości pliku przy użyciu `hashlib.md5()`.
  2. **Percepcyjne duplikaty (dHash):** Skalowanie obrazu do $9\times 8$ w skali szarości i porównywanie różnic gradientu sąsiednich pikseli.
* **Kluczowy fragment kodu:**
  ```python
  # dHash: 9x8 w skali szarości, porównanie poziome
  resized = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
  pixels = list(resized.getdata())
  diff = []
  for row in range(8):
      for col in range(8):
          pixel_left = pixels[row * 9 + col]
          pixel_right = pixels[row * 9 + col + 1]
          diff.append(pixel_left > pixel_right)
  # Obliczenie 64-bitowego hasza binarnego
  ```
* **Wnioski z EDA:** Zbiór zawiera 0 dokładnych duplikatów MD5, co gwarantuje pełną niezależność próbek.

#### 3. Funkcja: `generate_eda_visualizations(filepaths, labels, figures_dir)`
* **Sygnatura:** `def generate_eda_visualizations(filepaths: List[str], labels: List[int], figures_dir: str) -> None`
* **Cel:** Generowanie 4 kluczowych wykresów diagnostycznych:
  1. `eda_class_balance.png`: Wykres słupkowy proporcji klas (weryfikacja balansu 50/50).
  2. `eda_brightness_distribution.png`: Histogram średniej jasności odlewów poprawnych vs wadliwych.
  3. `eda_pixel_intensity.png`: Rozkład gęstości prawdopodobieństwa wartości pikseli.
  4. `eda_mean_std_images.png`: Średni obraz wirnika oraz mapa wariancji (odchylenia standardowego pikseli) w przestrzeni 2D dla klasy OK i DEF, ujawniająca obszary najczęstszego występowania skaz.

#### 4. Funkcja: `run_eda(data_dir, output_dir, figures_dir)`
* **Sygnatura:** `def run_eda(data_dir="casting_data", ...) -> Dict[str, Any]`
* **Rola:** Główny orkiestrator. Skanuje zbiór przy pomocy `scan_dataset()`, uruchamia `check_image_integrity()`, `detect_duplicates()`, `generate_eda_visualizations()` oraz zapisuje raport Markdown do `reports/eda_report.md`.

---

### C. Plik: `ml/run_cross_validation.py`

Punkt wejścia CLI do procedury 5x2-fold Cross-Validation i testów statystycznych.

#### Funkcja: `main()`
* **Sygnatura:** `def main() -> None`
* **Argumenty CLI:** `--data_dir`, `--epochs` (domyślnie 5 na fold), `--batch_size` (64), `--lr` (1e-4), `--num_workers` (4), `--seed` (42).
* **Co wywołuje:** `src.cross_validation_5x2.run_5x2_cross_validation(...)`.
* **Wyświetla:** Zestawienie wyników $\text{Recall}$ oraz $\text{Precision}$ dla każdego modelu w formacie: `średnia ± odchylenie standardowe`.

---

### D. Plik: `ml/src/cross_validation_5x2.py`

Implementacja procedury walidacyjnej zaprojektowanej specjalnie do porównywania algorytmów uczenia maszynowego.

#### Funkcja: `run_5x2_cross_validation(...)`
* **Sygnatura:**
  ```python
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
  ) -> Dict[str, Any]
  ```
* **Co wywołuje skąd:**
  - `scan_dataset` z [ml/src/dataset.py](file:///home/dawid-chudzicki/projects/uczenie-maszyn/ml/src/dataset.py): Pobiera ścieżki i etykiety.
  - `RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)` ze `sklearn.model_selection`: Dzieli 6000 próbek na 5 powtórzeń po 2 foldy (po 3000 train / 3000 test w każdym foldzie z zachowaniem proporcji klas 50/50).
  - `build_model` z [ml/src/models.py](file:///home/dawid-chudzicki/projects/uczenie-maszyn/ml/src/models.py): Tworzy świeżą instancję modelu na każdym foldzie.
  - `train_model` oraz `evaluate` z [ml/src/train_eval.py](file:///home/dawid-chudzicki/projects/uczenie-maszyn/ml/src/train_eval.py): Przeprowadza trening i zbiera metryki.
  - `compare_all_model_pairs` z [ml/src/statistical_tests.py](file:///home/dawid-chudzicki/projects/uczenie-maszyn/ml/src/statistical_tests.py): Oblicza testy istotności statystycznej.
* **Struktura pętli 10 foldów:**
  ```python
  rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random_state)
  for fold_idx, (train_indices, test_indices) in enumerate(rskf.split(filepaths_arr, labels_arr)):
      # Dla każdego z 3 modeli:
      # 1. Zainicjalizuj model od nowa (świeże wagi)
      # 2. Wytrenuj na foldzie train
      # 3. Zmierz Recall, Precision, F1 na foldzie test
      # 4. Zapisz wynik krotki (score_fold1, score_fold2)
  ```
* **Dlaczego n_splits=2?** Podział 50/50 gwarantuje, że zbiory testowe wewnątrz jednego powtórzenia są całkowicie rozłączne, co jest matematycznym wymogiem testu t-Studenta Diettericha.

---

### E. Plik: `ml/src/statistical_tests.py`

Implementacja rygorystycznych testów statystycznych dla uczenia maszynowego.

#### 1. Funkcja: `dietterich_5x2cv_paired_t_test(scores_a, scores_b, alpha=0.05)`
* **Sygnatura:** `def dietterich_5x2cv_paired_t_test(scores_a: List[Tuple[float, float]], scores_b: List[Tuple[float, float]], alpha: float = 0.05) -> Dict[str, Any]`
* **Matematyczna podstawa testu:**
  Standardowy test t-Studenta przy 10-fold CV ma zawyżony błąd I rodzaju (twierdzi, że modele różnią się istotnie, gdy w rzeczywistości tak nie jest), ponieważ zbiory treningowe w 90% na siebie nachodzą. Thomas Dietterich (1998) zaproponował statystykę:
  $$t = \frac{p_1^{(1)}}{\sqrt{\frac{1}{5} \sum_{i=1}^5 s_i^2}}$$
  gdzie:
  - $p_1^{(1)} = a_1^{(1)} - b_1^{(1)}$ to różnica wyników modeli $A$ i $B$ w pierwszym foldzie pierwszego powtórzenia.
  - $s_i^2 = (p_i^{(1)} - \bar{p}_i)^2 + (p_i^{(2)} - \bar{p}_i)^2$ to oszacowanie wariancji różnic wewnątrz $i$-tego powtórzenia.
  - Zmienna $t$ ma rozkład $t$-Studenta z **5 stopniami swobody** ($df=5$).
* **Kluczowy fragment kodu:**
  ```python
  p1_1 = diff_1  # różnica z pierwszego folda
  mean_diff = (diff_1 + diff_2) / 2.0
  s_i_sq = (diff_1 - mean_diff) ** 2 + (diff_2 - mean_diff) ** 2
  s_squared_sum += s_i_sq

  denominator = np.sqrt((1.0 / 5.0) * s_squared_sum)
  t_stat = p1_1 / denominator
  p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=5))
  reject_h0 = bool(p_val < alpha)
  ```
* **Interpretacja:** Jeśli $p < 0.05$, odrzucamy hipotezę zerową $H_0$ o równości modeli – różnica jakościowa jest statystycznie udowodniona.

#### 2. Funkcja: `nadeau_bengio_corrected_t_test(diffs, n_train_ratio=0.5, alpha=0.05)`
* **Sygnatura:** `def nadeau_bengio_corrected_t_test(diffs: List[float], n_train_ratio: float = 0.5, alpha: float = 0.05) -> Dict[str, Any]`
* **Cel:** Test uzupełniający. Claude Nadeau i Yoshua Bengio wprowadzili korektę wariancji uwzględniającą zależność prób:
  $$\sigma_{\text{corrected}}^2 = \left(\frac{1}{n} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) s^2$$
  W 2-fold CV stosunek $n_{\text{test}} / n_{\text{train}} = 1.0$.

#### 3. Funkcja: `compare_all_model_pairs(cv_metric_records, metric_name="recall", alpha=0.05)`
* **Sygnatura:** `def compare_all_model_pairs(...) -> Dict[str, Any]`
* **Rola:** Automatycznie generuje macierz porównań dla wszystkich par:
  - ResNet18 vs Custom CNN
  - MobileNetV2 vs Custom CNN
  - ResNet18 vs MobileNetV2
  Zwraca kompletne statystyki $t$, wartości $p\text{-value}$ oraz jednoznaczne wnioski tekstowe.
