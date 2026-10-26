# Etap 2: Trening Końcowy Modeli Produkcyjnych i Potok Danych
## Przewodnik Techniczny po Kodzie Źródłowym

Autorzy: Dawid Chudzicki (272715), Piotr Szczerbiak (294722)

---

## 1. Cel i Rola Etapu 2

Gdy na etapie 1 (walidacja krzyżowa 5x2) potwierdziliśmy statystyczną wyższość i stabilność badanych architektur, przechodzimy do **fazy produkcyjnej**:
1. **Definitywny podział danych:** Wydzielenie i zamrożenie nienaruszalnego zbioru testowego 1200 obrazów (20%) zapisanego w pliku [ml/reports/test_split.json](file:///home/dawid-chudzicki/projects/uczenie-maszyn/ml/reports/test_split.json).
2. **Trening pełnowymiarowy:** Wytrenowanie modeli na **100% zbioru treningowego (4800 obrazów)** z zastosowaniem augmentacji obrazu.
3. **Eksport wag produkcyjnych:** Wygenerowanie plików wag `.pth` do katalogu [ml/models_weights/](file:///home/dawid-chudzicki/projects/uczenie-maszyn/ml/models_weights/) dla serwisów inferencyjnych.
4. **Rejestracja historii zbieżności:** Eksport krzywych funkcji straty, czułości i precyzji do [ml/reports/figures/training_curves.png](file:///home/dawid-chudzicki/projects/uczenie-maszyn/ml/reports/figures/training_curves.png).

---

## 2. Architektura Przepływu w Etapie 2

```
casting_data/ (6000 obrazów)
      │
      ▼
train_test_split (stratify, 80/20)
      │
      ├──► Zbiór Testowy (1200 szt.) ─────► reports/test_split.json (ZAMROŻONY DO ETAPU 3)
      │
      └──► Zbiór Treningowy (4800 szt.)
                │
                ├──► [ml/src/dataset.py] (Augmentacje: Flip, Rotation, ColorJitter)
                │         │
                │         ▼
                │    DataLoader (batch=64, shuffle=True, workers=4)
                │         │
                ├──► [ml/src/models.py] (CustomCNN / ResNet18 / MobileNetV2)
                │         │
                └──► [ml/src/train_eval.py] (Pętla trenująca, BCEWithLogitsLoss, Adam)
                          │
                          ├──► models_weights/ (*.pth: custom_cnn, resnet18, mobilenet_v2)
                          └──► reports/final_train_history.json & training_curves.png
```

---

## 3. Szczegółowy Opis Plików i Funkcji

---

### A. Plik: `ml/train_final_models.py`

Główny orkiestrator procesu treningowego generujący ostateczne wagi wdrożeniowe.

#### 1. Funkcja: `train_final_models(...)`
* **Sygnatura:**
  ```python
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
  ) -> None
  ```
* **Krok 1: Stratyfikowany podział 80/20 i zamrożenie test splitu:**
  ```python
  train_idx, test_idx = train_test_split(
      np.arange(len(labels_arr)),
      test_size=test_size,
      random_state=seed,
      stratify=labels_arr  # Gwarantuje idealne 50/50 klas w obu częściach
  )
  test_split_info = {
      "test_filepaths": filepaths_arr[test_idx].tolist(),
      "test_labels": labels_arr[test_idx].tolist(),
      "seed": seed,
      "test_size": test_size,
  }
  with open(os.path.join(reports_dir, "test_split.json"), "w", encoding="utf-8") as f:
      json.dump(test_split_info, f, indent=2)
  ```
  *Po co:* Zbiór testowy zostaje trwale zapisany w formacie JSON. Dzięki temu ewaluacja w Etapie 3 operuje dokładnie na tych samych 1200 próbkach, bez ryzyka wycieku danych.

* **Krok 2: Sekwencyjny trening 3 architektur:**
  ```python
  for model_name, save_filename, display_name in models_config:
      model = build_model(model_name, pretrained=True, device=device)
      train_res = train_model(
          model=model,
          train_loader=train_loader,
          val_loader=test_loader,
          epochs=epochs,
          lr=lr,
          device=device,
      )
      save_path = os.path.join(models_dir, save_filename)
      torch.save(model.state_dict(), save_path)
  ```

* **Krok 3: Generowanie 4-panelowego wykresu zbieżności:**
  Rysuje w jednym oknie:
  - Panel [0, 0]: Funkcja straty treningowej (`train_loss`).
  - Panel [0, 1]: Funkcja straty walidacyjnej (`val_loss`).
  - Panel [1, 0]: Czułość walidacyjna (`val_recall`) z naniesioną linią krytyczną $\text{Recall} = 0.98$.
  - Panel [1, 1]: Precyzja walidacyjna (`val_precision`) z linią docelową $\text{Precision} = 0.90$.
  Wykres zapisywany jest jako `reports/figures/training_curves.png`.

#### 2. Funkcja: `main()`
* **Sygnatura:** `def main() -> None`
* **Rola:** Parser argumentów CLI (`--epochs`, `--batch_size`, `--lr`, `--num_workers`). Pozwala uruchomić trening komendą:
  `python3 train_final_models.py --epochs 10 --batch_size 64`

---

### B. Plik: `ml/src/dataset.py`

Zarządza transformacjami obrazów, normalizacją i formatem danych wejściowych PyTorch.

#### 1. Transformacje: `get_train_transforms()` i `get_eval_transforms()`
* **Sygnatury:**
  - `def get_train_transforms(image_size=(300, 300)) -> transforms.Compose`
  - `def get_eval_transforms(image_size=(300, 300)) -> transforms.Compose`
* **Kod potoku treningowego:**
  ```python
  transforms.Compose([
      transforms.Resize((300, 300)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomVerticalFlip(p=0.5),
      transforms.RandomRotation(degrees=15),
      transforms.ColorJitter(brightness=0.1, contrast=0.1),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  ```
* **Dlaczego takie augmentacje:**
  - Wirnik jest detalem osiowosymetrycznym – obroty o $\pm 15^\circ$ oraz odbicia lustrzane są w 100% fizycznie poprawne i symulują losowe ułożenie na taśmie przenośnika.
  - `ColorJitter` uodparnia sieć na wahania oświetlenia przemysłowego w hali fabrycznej.
  - Normalizacja stałymi ImageNet jest krytyczna dla warstw transfer learningu w ResNet18 i MobileNetV2.

#### 2. Funkcja: `scan_dataset(data_dir="casting_data")`
* **Sygnatura:** `def scan_dataset(data_dir: str) -> Tuple[List[str], List[int]]`
* **Działanie:** Przeszukuje podfoldery `ok_front` (mapowane na `0`) oraz `def_front` (mapowane na `1`), filtruje rozszerzenia `.jpeg`, `.jpg`, `.png` i zwraca posortowane listy ścieżek i etykiet.

#### 3. Klasa: `CastingDataset(Dataset)`
* **Sygnatura:** `class CastingDataset(torch.utils.data.Dataset)`
* **Metody:**
  - `__len__()`: Zwraca liczbę próbek w zbiorze.
  - `__getitem__(idx)`: Otwiera plik PIL Image, konwertuje do RGB (`img.convert("RGB")`), aplikuje zdefiniowane transformacje i zwraca krotkę: `(tensor_obrazu, etykieta, sciezka_do_pliku)`.
* **Dlaczego wymuszamy RGB:** Obrazy w `casting_data` są monochromatyczne lub szare, ale modele z ImageNet (ResNet, MobileNet) wymagają wejścia o kształcie $(3, H, W)$.

---

### C. Plik: `ml/src/models.py`

Definicje architektur sieci konwolucyjnych i wzorzec fabryki modeli.

#### 1. Klasa: `CustomCNN(nn.Module)`
* **Sygnatura:** `class CustomCNN(nn.Module)`
* **Architektura:**
  - 4 bloki Feature Extractor:
    - Blok 1: Conv2d(3, 32, k=3, p=1) $\to$ BatchNorm2d $\to$ ReLU $\to$ MaxPool2d(2, 2) [rozmiar mapy: $150\times 150$]
    - Blok 2: Conv2d(32, 64, k=3, p=1) $\to$ BatchNorm2d $\to$ ReLU $\to$ MaxPool2d(2, 2) [rozmiar mapy: $75\times 75$]
    - Blok 3: Conv2d(64, 128, k=3, p=1) $\to$ BatchNorm2d $\to$ ReLU $\to$ MaxPool2d(2, 2) [rozmiar mapy: $37\times 37$]
    - Blok 4: Conv2d(128, 256, k=3, p=1) $\to$ BatchNorm2d $\to$ ReLU $\to$ MaxPool2d(2, 2) [rozmiar mapy: $18\times 18$]
  - Redukcja przestrzenna: `AdaptiveAvgPool2d((1, 1))` (uśrednienie do wektora 256 cech niezależnie od wymiarów wejścia).
  - Klasyfikator: Linear(256, 128) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.3) $\to$ Linear(128, 1).
  - Wyjście: pojedynczy logit (kształt `(B,)`).
* **Liczba parametrów:** 422 177 (~1.6 MB na dysku).

#### 2. Funkcja: `build_resnet18(pretrained=True, dropout_rate=0.3)`
* **Sygnatura:** `def build_resnet18(...) -> nn.Module`
* **Implementacja Transfer Learningu:**
  ```python
  model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
  in_features = model.fc.in_features  # 512
  model.fc = nn.Sequential(
      nn.Dropout(p=dropout_rate),
      nn.Linear(in_features, 1)  # Binarne wyjście
  )
  ```
* **Liczba parametrów:** 11 177 025 (~42.7 MB). Wagi ImageNet dostarczają bogatych filtrów niskopoziomowych (krawędzie, tekstury, cienie skaz odlewniczych).

#### 3. Funkcja: `build_mobilenet_v2(pretrained=True, dropout_rate=0.3)`
* **Sygnatura:** `def build_mobilenet_v2(...) -> nn.Module`
* **Implementacja dla Edge AI:**
  ```python
  model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
  in_features = model.classifier[1].in_features  # 1280
  model.classifier = nn.Sequential(
      nn.Dropout(p=dropout_rate),
      nn.Linear(in_features, 1)
  )
  ```
* **Liczba parametrów:** 2 225 153 (~8.6 MB). Używa splotów głębokościowo rozdzielczych (Depthwise Separable Convolutions), co daje 5-krotnie mniejszy rozmiar niż ResNet18 przy zachowaniu bardzo wysokiej dokładności.

#### 4. Funkcja: `build_model(model_name, pretrained=True, device="cpu")`
* **Wzorzec fabryki:** Przyjmuje string `"cnn"`, `"resnet18"` lub `"mobilenet_v2"` i zwraca gotowy model przeniesiony na żądane urządzenie (`cuda` lub `cpu`).

---

### D. Plik: `ml/src/train_eval.py`

Procedury optymalizacyjne i obliczanie metryk inżynierskich.

#### 1. Funkcja: `train_one_epoch(...)`
* **Sygnatura:**
  ```python
  def train_one_epoch(
      model: nn.Module,
      dataloader: DataLoader,
      criterion: nn.Module,
      optimizer: torch.optim.Optimizer,
      device: torch.device,
      scaler: Optional[torch.amp.GradScaler] = None,
  ) -> float
  ```
* **Działanie:** Iteruje po batchach obrazów. Wykonuje:
  1. Wyzerowanie gradientów: `optimizer.zero_grad()`.
  2. Przepływ w przód (Forward pass): `outputs = model(images)`.
  3. Obliczenie straty: `loss = criterion(outputs, targets)`.
  4. Wsteczną propagację (Backward pass): `loss.backward()`.
  5. Aktualizację wag: `optimizer.step()`.
  Zwraca średni koszt epoki (`running_loss / total_samples`).

#### 2. Funkcja: `train_model(...)`
* **Sygnatura:**
  ```python
  def train_model(
      model: nn.Module,
      train_loader: DataLoader,
      val_loader: DataLoader,
      epochs: int = 10,
      lr: float = 1e-4,
      device: torch.device = "cpu",
      verbose: bool = True,
  ) -> Dict[str, Any]
  ```
* **Kluczowe mechanizmy:**
  - Optymalizator: `torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)`.
  - Funkcja straty: `nn.BCEWithLogitsLoss()`.
  - **Checkpointing najlepszych wag:** Po każdej epoce ewaluuje model na walidacji. Jeśli walidacyjny F1-score lub Recall jest wyższy od dotychczasowego maksimum, zapisuje `best_weights = copy.deepcopy(model.state_dict())`. Na koniec przywraca najlepsze wagi.
  - Zwraca słownik z historią uczenia (`train_loss`, `val_loss`, `val_recall`, `val_precision`, `val_f1`, `val_accuracy`).

#### 3. Funkcja: `evaluate(...)`
* **Sygnatura:**
  ```python
  def evaluate(
      model: nn.Module,
      dataloader: DataLoader,
      criterion: nn.Module,
      device: torch.device,
      threshold: float = 0.5,
  ) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]
  ```
* **Działanie:** Przełącza model w tryb `model.eval()`, wyłącza śledzenie gradientów (`with torch.no_grad():`), zamienia logity na ciągłe prawdopodobieństwa funkcją sigmoidalną `probs = torch.sigmoid(outputs)` i przekazuje wyniki do `compute_metrics()`.

#### 4. Funkcja: `compute_metrics(y_true, y_prob, threshold=0.5)`
* **Sygnatura:** `def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]`
* **Obliczane metryki:**
  - Macierz pomyłek: $TP, FP, TN, FN$.
  - Czułość (Recall): $\frac{TP}{TP + FN}$ (wykrywalność wad).
  - Precyzja (Precision): $\frac{TP}{TP + FP}$ (brak fałszywych alarmów).
  - $F_1$-score: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
  - Specyficzność (Specificity): $\frac{TN}{TN + FP}$.
  - ROC-AUC oraz PR-AUC (Average Precision).
