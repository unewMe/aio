# Etap 4: Mikroserwis Produkcyjny FastAPI, Zarządzanie RAM On-Demand i Symulator Taśmy
## Przewodnik Techniczny po Kodzie Źródłowym

Autorzy: Dawid Chudzicki (272715), Piotr Szczerbiak (294722)

---

## 1. Cel i Rola Etapu 4

Etap 4 to wdrożenie operacyjne wytrenowanych modeli na linii produkcyjnej. Realizuje kluczowe wyzwania inżynierii systemów przemysłowych:
1. **Mikroserwis inferencyjny niskich opóźnień:** Samodzielny kontener Docker (`casting_ml_service`) z FastAPI serwujący predykcje PyTorch na porcie 8001.
2. **Polityka pamięci RAM On-Demand (`Single Active Model Policy`):** Utrzymanie w pamięci operacyjnej tylko jednego aktywnego modelu naraz, zwalnianie nieużywanych wag i odśmiecanie pamięci (`gc.collect()`), dzięki czemu proces zużywa <450 MB RAM i może działać na tanich komputerach brzegowych (np. Raspberry Pi 4 lub komputerach panelowych IPC).
3. **Emulacja ograniczeń sprzętowych (CPU Throttling):** Możliwość testowania zachowania systemu pod obciążeniem i symulowania wąskich gardeł linii produkcyjnej na profilach sprzętowych (Stacja Robocza, IPC, Jetson Nano, Raspberry Pi, Mikrokontroler Low-Power).
4. **Strumieniowa symulacja taśmy (1200 próbek z Ground Truth):** Ciągły napływ obrazów wirników w formacie Zero-Latency Base64 z automatyczną oceną werdyktu sieci w czasie rzeczywistym ($TP, FP, TN, FN$).

---

## 2. Architektura Przepływu w Etapie 4

```
                  ┌──────────────────────────────────────────────┐
                  │          KONTENER DOCKER (ml:8001)           │
                  │                                              │
Kamera AOI /      │  POST /predict ────────► Preprocessing (RGB) │
Plik Wirnika ────►│                               │              │
                  │                               ▼              │
                  │                    Model PyTorch On-Demand   │
                  │                    (ResNet18 / MobileNetV2)  │
                  │                               │              │
                  │                               ▼              │
                  │                    Sigmoid + Próg tau*       │
                  │                               │              │
                  │                               ▼              │
                  │                    JSON: Werdykt (OK/DEFECT) │
                  │                          Prawdopodobieństwo  │
                  │                          Latencja (ms)       │
                  │                                              │
                  │  POST /system/cpu-throttle ──► Throttling    │
                  │  POST /simulation/step ────► Test Stream B64 │
                  │  GET  /samples/{cat}/{file} ──► Serwowanie   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │        BACKEND API PROXY (backend:8000)      │
                  │        • Historia inspekcji (SQLite)         │
                  │        • Telemetria taśmy produkcyjnej       │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         FRONTEND UI (Next.js:3000)           │
                  │         • Podgląd kamery AOI na żywo         │
                  │         • Symulacja taśmy ciągłej (1200 szt) │
                  │         • Statystyki Andon i telemetry grid  │
                  └──────────────────────────────────────────────┘
```

---

## 3. Szczegółowy Opis Pliku: `ml/server.py`

Główny mikroserwis wdrożeniowy łączący modele PyTorch z resztą ekosystemu aplikacji.

---

### A. Konfiguracja Modeli i Metadane Produkcyjne

Słownik `MODEL_CONFIGS` definiuje parametry operacyjne każdego modelu, w tym optymalne progi odcięcia $\tau^*$ wyznaczone na Etapie 3:
```python
MODEL_CONFIGS = {
    "resnet18": {
        "display_name": "ResNet18",
        "key": "resnet18",
        "weights_file": "resnet18.pth",
        "optimal_threshold": 0.197,
        "role": "Najwyższa Dokładność • Transfer Learning",
        "parameters": 11177537,
        "size_mb": 42.71
    },
    "mobilenet_v2": {
        "display_name": "MobileNetV2",
        "key": "mobilenet_v2",
        "weights_file": "mobilenet_v2.pth",
        "optimal_threshold": 0.099,
        "role": "Optimum Brzegowe • Edge AI",
        "parameters": 2225153,
        "size_mb": 8.56
    },
    "cnn": {
        "display_name": "Custom CNN",
        "key": "cnn",
        "weights_file": "custom_cnn.pth",
        "optimal_threshold": 0.158,
        "role": "Autorska Sieć Bazowa • Baseline",
        "parameters": 994369,
        "size_mb": 3.82
    }
}
```

---

### B. Zarządzanie Pamięcią RAM: Polityka On-Demand

#### 1. Funkcja: `get_process_memory_mb()`
* **Sygnatura:** `def get_process_memory_mb() -> float`
* **Zasada działania:** Odczytuje rzeczywistą fizyczną pamięć rezydentną procesu (`VmRSS` - Resident Set Size) bezpośrednio z jądra Linuksa z pliku `/proc/self/status`. Jeśli plik jest niedostępny, używa modułu systemowego `resource.getrusage()`.
* **Dlaczego to robimy:** Pozwala monitorować w czasie rzeczywistym zużycie RAM mikroserwisu i wyświetlać je na dashboardzie frontendowym.

#### 2. Funkcja: `unload_inactive_models(keep_key=None)`
* **Sygnatura:** `def unload_inactive_models(keep_key: Optional[str] = None) -> None`
* **Implementacja:**
  ```python
  def unload_inactive_models(keep_key: Optional[str] = None):
      global _loaded_models
      keys_to_remove = [k for k in list(_loaded_models.keys()) if k != keep_key]
      for k in keys_to_remove:
          del _loaded_models[k]
      if keys_to_remove:
          gc.collect()  # Wymuszenie czyszczenia pamięci w Pythonie
          if DEVICE.type == "cuda":
              torch.cuda.empty_cache()
  ```
* **Cel inżynierski:** Zwalnia z RAM wagi wszystkich modeli poza aktualnie używanym. Chroni system przed błędem `Out-Of-Memory (OOM)` na urządzeniach wbudowanych.

#### 3. Funkcja: `load_model_to_memory(model_key)`
* **Sygnatura:** `def load_model_to_memory(model_key: str) -> torch.nn.Module`
* **Działanie:** Leniwe ładowanie (*Lazy Loading*). Jeśli model jest już w pamięci, natychmiast go zwraca. Jeśli nie:
  1. Wywołuje `unload_inactive_models()` aby zwolnić pamięć po starym modelu.
  2. Tworzy instancję architektury: `model = build_model(model_key, pretrained=False, device=DEVICE)`.
  3. Ładuje wagi z pliku `.pth`: `model.load_state_dict(torch.load(weights_path, map_location=DEVICE))`.
  4. Przełącza model w tryb ewaluacji: `model.eval()`.
  5. Umieszcza w słowniku cache `_loaded_models[model_key]`.

---

### C. Emulacja Ograniczeń Sprzętowych (CPU Throttling)

#### 1. Słownik profili: `HARDWARE_PROFILES`
Definiuje realistyczne charakterystyki sprzętowe:
- `unthrottled`: Wszystkie rdzenie procesora, 100% mocy, brak opóźnień (mnożnik $\times 1.0$).
- `industrial_pc`: Przemysłowy komputer panelowy (Intel Core i3, limit wątków, 60% mocy, mnożnik $\times 1.8$).
- `jetson_nano`: Moduł krawędziowy Edge AI (2 wątki, 35% mocy, mnożnik $\times 3.2$).
- `raspberry_pi_4`: Komputer jednopłytkowy ARM (2 wątki, 18% mocy, mnożnik $\times 5.5$).
- `ultra_low_power`: Ekstremalne dławienie (1 rdzeń, 8% mocy, mnożnik $\times 12.0$) do wymuszenia zatoru i zatrzymania taśmy.

#### 2. Funkcja: `apply_cpu_configuration(profile, threads, throttle_percent)`
* **Sygnatura:** `def apply_cpu_configuration(...) -> Dict[str, Any]`
* **Działanie:** Wywołuje funkcję PyTorcha `torch.set_num_threads(t)`, co fizycznie ogranicza pulę wątków obliczeniowych bibliotek OpenMP/MKL na poziomie procesora.

---

### D. Punkty Końcowe API (Endpoints)

#### 1. `POST /predict`
* **Cel:** Główna inferencja inspekcyjna pojedynczego wirnika.
* **Parametry wejściowe (Form-Data):**
  - `file: UploadFile`: Plik graficzny odlewu (JPEG/PNG).
  - `model_name: str` (opcjonalny, domyślnie `"resnet18"`).
  - `threshold: Optional[float]` (opcjonalny; jeśli nie podano, używany jest optymalny $\tau^*$ z `MODEL_CONFIGS`).
* **Implementacja krok po kroku:**
  ```python
  # 1. Odczyt obrazu do PIL i transformacja
  image_bytes = await file.read()
  img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
  input_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # Kształt: (1, 3, 300, 300)

  # 2. Pobranie aktywnego modelu (On-Demand)
  model = load_model_to_memory(model_name)

  # 3. Pomiar inferencji w przód
  start_t = time.perf_counter()
  with torch.no_grad():
      logits = model(input_tensor)
      prob_defect = float(torch.sigmoid(logits).item())
  raw_inference_ms = (time.perf_counter() - start_t) * 1000.0

  # 4. Werdykt Zero-Defect w oparciu o próg
  active_threshold = threshold if threshold is not None else cfg["optimal_threshold"]
  is_defective = bool(prob_defect >= active_threshold)
  verdict = "DEFECT" if is_defective else "OK"
  ```
* **Zwracana odpowiedź JSON:**
  ```json
  {
    "verdict": "DEFECT",
    "is_defective": true,
    "confidence": 0.9984,
    "prob_defect": 0.9984,
    "prob_ok": 0.0016,
    "model_used": "resnet18",
    "threshold_used": 0.197,
    "inference_time_ms": 62.4,
    "hardware_profile": "unthrottled",
    "zero_defect_compliant": true,
    "memory_rss_mb": 423.5
  }
  ```

#### 2. `POST /simulation/step`
* **Cel:** Zasilanie symulatora taśmy produkcyjnej w czasie rzeczywistym.
* **Zasada działania:**
  - Wczytuje 1200 próbek ze zbioru [reports/test_split.json](file:///home/dawid-chudzicki/projects/uczenie-maszyn/ml/reports/test_split.json).
  - Pobiera odlew o wskazanym indeksie lub losowy według filtra (`all`, `def`, `ok`, `realistic` - 10% wad jak w prawdziwej fabryce).
  - Konwertuje plik JPEG do formatu **Base64** (`data:image/jpeg;base64,...`), co eliminuje konieczność ponownego pobierania pliku przez przeglądarkę (Zero-Latency Stream).
  - Dołącza etykietę wzorcową **Ground Truth** (`ground_truth_label`), umożliwiając automatyczną ocenę czy klasyfikacja była poprawna ($TP, TN$), czy wystąpił błąd ($FP, FN$).

#### 3. `GET /samples/{category}/{filename}`
* **Cel:** Serwowanie plików graficznych próbek wzorcowych dla miniatur w menu wyboru próbek.
* **Działanie:** Weryfikuje istnienie pliku w `data/sample_images/` lub `casting_data/` i zwraca strumień obrazu przez `FileResponse(path, media_type="image/jpeg")`.

#### 4. `POST /models/load` oraz `POST /models/unload`
* **Cel:** Dynamiczne zarządzanie pamięcią operacyjną. Endpoint `/models/unload` pozwala operatorowi linii natychmiast zrzucić model z RAM, redukując zużycie pamięci do minimum.

#### 5. `GET /metrics/final`
* **Cel:** Dostarcza frontendowi zweryfikowane metryki ze zbioru testowego z pliku `reports/final_evaluation_results.json` na potrzeby kart charakterystyk w menu modeli.
