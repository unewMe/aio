# System Automatycznej Wizyjnej Kontroli Jakości Odlewów (AOI)

**Autorzy:** Dawid Chudzicki (272715), Piotr Szczerbiak (294722)  
**Przedmiot:** Uczenie Maszynowe / Sztuczna Inteligencja  
**Architektura:** Trójwarstwowy system produkcyjny (Frontend Next.js + Backend FastAPI w Netflix Dispatch DDD + Dedykowany Mikroserwis ML PyTorch)

---

## 1. Architektura Systemu (3 Foldery & Docker Compose)
    
Projekt został podzielony na trzy niezależne, wyspecjalizowane warstwy:

```text
uczenie-maszyn/
├── docker-compose.yml               # Orkiestracja wszystkich 3 usług
├── README.md                        # Niniejsza dokumentacja
│
├── ml/                              # 1. WARSTWA MACHINE LEARNING & WAGI
│   ├── server.py                    # Dedykowany mikroserwis inferencji PyTorch (port 8001)
│   ├── app.py                       # Zachowany prototyp Streamlit
│   ├── casting_data/                # Oryginalny zbiór danych (def_front, ok_front)
│   ├── data/                        # Podziały train/test oraz próbki wzorcowe (sample_images)
│   ├── models_weights/              # Wagi modeli (.pth: Custom CNN, ResNet18, MobileNetV2)
│   ├── reports/                     # Raporty ewaluacyjne, metryki JSON i wykresy PNG
│   ├── src/                         # Moduły PyTorch: architektury, datasety, walidacja 5x2 CV
│   ├── train_final_models.py        # Skrypt treningowy finalnych wag
│   ├── evaluate_models.py           # Skrypt ewaluacji testowej i analizy błędów
│   ├── requirements.txt             # Zależności ML (PyTorch, torchvision, Pillow, FastAPI)
│   └── Dockerfile                   # Kontener mikroserwisu ML
│
├── backend/                         # 2. BACKEND FASTAPI (NETFLIX DISPATCH DDD)
│   ├── app/
│   │   ├── core/ / config.py        # Konfiguracja środowiska i adresy serwisów
│   │   ├── database.py              # Sesja SQLAlchemy i model bazowy
│   │   ├── api.py                   # Główny router agregujący domeny
│   │   ├── main.py                  # Inicjalizacja FastAPI, CORS, cykl życia lifespan
│   │   │
│   │   ├── inspection/              # Bounded Context: Inspekcje Jakości & Reguły Decyzyjne
│   │   │   ├── models.py            # Model bazy SQLite (InspectionRecord)
│   │   │   ├── schemas.py           # Schematy Pydantic DTO (request/response)
│   │   │   ├── service.py           # Logika biznesowa, wywołania serwisu ML, progi tau
│   │   │   └── views.py             # Endpointy API (/api/v1/inspections)
│   │   │
│   │   ├── model_registry/          # Bounded Context: Rejestr Modeli & Benchmarki
│   │   │   ├── schemas.py           # Metadane i schematy benchmarków
│   │   │   ├── service.py           # Klient komunikacji z serwisem ML
│   │   │   └── views.py             # Endpointy API (/api/v1/models)
│   │   │
│   │   ├── analytics/               # Bounded Context: Raporty & Wykresy
│   │   │   ├── service.py           # Pobieranie raportów ewaluacji i statystyk
│   │   │   └── views.py             # Endpointy API (/api/v1/analytics)
│   │   │
│   │   ├── samples/                 # Bounded Context: Próbki Referencyjne
│   │   │   ├── service.py           # Pobieranie i strumieniowanie próbek
│   │   │   └── views.py             # Endpointy API (/api/v1/samples)
│   │   │
│   │   └── simulation/              # Bounded Context: Symulacja Strumienia Produkcyjnego
│   │       ├── schemas.py           # Schematy DTO (Ground Truth vs Predykcja, TP/TN/FP/FN)
│   │       ├── service.py           # Krok symulacji, weryfikacja trafności, audyt w SQLite
│   │       └── views.py             # Endpointy API (/api/v1/simulation)
│   ├── requirements.txt             # Zależności backendu (FastAPI, SQLAlchemy, httpx - BEZ PyTorch)
│   └── Dockerfile                   # Szybki kontener backendu (< 10 s budowania)
│
└── frontend/                        # 3. NOWOCZESNY FRONTEND NEXT.JS (INDUSTRIAL AI)
    ├── src/
    │   ├── app/
    │   │   ├── layout.tsx           # Layout z nawigacją i motywem ciemnym
    │   │   ├── page.tsx             # Główna stacja inspekcyjna AOI (Live Viewport)
    │   │   ├── simulation/page.tsx  # Symulacja napływu 1200 odlewów ze zbioru testowego z Ground Truth
    │   │   ├── models/page.tsx      # Porównanie architektur i live benchmark CPU
    │   │   ├── analytics/page.tsx   # Raporty naukowe, macierze i krzywe ROC/PR
    │   │   ├── history/page.tsx     # Dziennik audytu i historia z bazy SQLite
    │   │   └── globals.css          # Design system Industrial AI Dark Mode (Tailwind CSS + shadcn)
    │   ├── components/              # Komponenty shadcn/ui: Button, Card, Badge, Tabs, Slider, Progress
    │   └── lib/                     # Klient API i definicje typów TypeScript
    ├── package.json
    └── Dockerfile                   # Kontener Next.js
```

---

## 2. Szybkie Uruchomienie (Docker Compose)

Wszystkie 3 usługi (ML, Backend, Frontend) uruchamiane są jednym poleceniem:

```bash
docker compose up --build
```

Po uruchomieniu poszczególne moduły dostępne są pod adresami:
- **Frontend (Next.js Dashboard):** [http://localhost:3000](http://localhost:3000)
- **Backend API (FastAPI Docs):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ML Inference Service (PyTorch Docs):** [http://localhost:8001/docs](http://localhost:8001/docs)

---

## 3. Uruchomienie Lokalne (Development bez Dockera)

### Krok 1: Mikroserwis ML (PyTorch)
```bash
cd ml
pip install -r requirements.txt
python3 -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### Krok 2: Backend FastAPI (Netflix Dispatch DDD)
```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Krok 3: Frontend Next.js
```bash
cd frontend
npm install
npm run dev
```
Aplikacja otworzy się na [http://localhost:3000](http://localhost:3000).

---

## 4. Kluczowe Metryki Inżynierskie & Zrealizowane Cele

| Model | Rola / Architektura | Recall (Wada) | Precision | F1-Score | Czas CPU | Rozmiar | FN (Wady Przeoczone) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ResNet18** | Transfer Learning (ImageNet) | **100.0%** | **99.50%** | **0.9975** | 62.5 ms | 42.7 MB | **0 / 600** |
| **MobileNetV2** | Optimum Edge AI | **100.0%** | **98.20%** | **0.9909** | 59.2 ms | 8.6 MB | **0 / 600** |
| **Custom CNN** | Autorska Sieć Bazowa | **100.0%** | **97.72%** | **0.9885** | 34.5 ms | 1.6 MB | **0 / 600** |

---

## 5. Cechy Architektury Netflix Dispatch w Backendzie

Backend został zorganizowany według wzorca **Domain-Driven Design (DDD)** spopularyzowanego przez projekt **Netflix Dispatch**:
1. **Separacja domenowa (Bounded Contexts):** Każda domena biznesowa (`inspection`, `model_registry`, `analytics`, `samples`) posiada własne modele, schematy Pydantic DTO, usługi domenowe (`service.py`) oraz routery FastAPI (`views.py`).
2. **Czysta izolacja ML:** Backend nie ładuje biblioteki PyTorch ani ciężkich wag, lecz deleguje zadania inferencji przez dedykowanego klienta HTTP (`httpx`) do mikroserwisu ML.
3. **Persystencja i Audyt:** Wyniki inspekcji, telemetria, zastosowane progi decyzyjne $\tau$ oraz werdykty produkcyjne są trwale rejestrowane w bazie danych SQLite / PostgreSQL za pośrednictwem SQLAlchemy.
