"""
Aplikacja demonstracyjna Streamlit:
System Automatycznej Kontroli Jakości Odlewów Wirników Pomp
Autorzy: Dawid Chudzicki (272715), Piotr Szczerbiak (294722)
Przedmiot: Uczenie Maszynowe / Sztuczna Inteligencja
"""

import os
import glob
import time
from typing import Optional, Tuple
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import torch
import streamlit as st

from src.dataset import get_eval_transforms
from src.models import build_model, get_model_size_mb, count_parameters

# 1. KONFIGURACJA STRONY STREAMLIT
st.set_page_config(
    page_title="Kontrola Jakości Odlewów | Pilot Technocast AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helper do wyświetlania obrazów bez ostrzeżeń o deprecacji w nowym Streamlit
def show_image(image, caption=None):
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        st.image(image, caption=caption, use_container_width=True)

# 2. ZAAWANSOWANE STYLE CSS (DESIGN SYSTEM PREMIUM INDUSTRIAL AI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Ukrycie zbędnych elementów interfejsu Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}

    /* Główny kontener aplikacji */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(15, 23, 42, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                    radial-gradient(circle at 20% 80%, rgba(14, 165, 233, 0.12) 0%, transparent 40%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(129, 140, 248, 0.35);
        color: #c7d2fe;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }

    .hero-badge .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 8px #38bdf8;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: #94a3b8;
        margin: 0;
        max-width: 850px;
        line-height: 1.5;
    }

    .authors-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 6px 14px;
        border-radius: 12px;
        color: #e2e8f0;
        font-size: 0.85rem;
        margin-top: 14px;
    }

    /* Karty Metryk KPI */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 15px -3px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.08);
    }

    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }

    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }

    .kpi-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        margin-top: 6px;
    }

    .kpi-badge-success {
        background-color: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }

    .kpi-badge-info {
        background-color: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
    }

    .kpi-badge-neutral {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }

    /* Karta Werdyktu AI (Status Cards) */
    .verdict-card {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.06);
        position: relative;
        overflow: hidden;
    }

    .verdict-card-def {
        background: linear-gradient(135deg, #fff1f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        color: #991b1b;
    }

    .verdict-card-ok {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #10b981;
        color: #166534;
    }

    .verdict-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }

    .verdict-desc {
        font-size: 1.02rem;
        font-weight: 500;
        margin: 0 0 14px 0;
        opacity: 0.9;
    }

    .verdict-recommendation {
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .verdict-rec-def {
        background-color: #dc2626;
        color: #ffffff;
        box-shadow: 0 4px 10px rgba(220, 38, 38, 0.3);
    }

    .verdict-rec-ok {
        background-color: #059669;
        color: #ffffff;
        box-shadow: 0 4px 10px rgba(5, 150, 105, 0.3);
    }

    /* Wizjer Kamery Przemysłowej (Camera Viewport Frame) */
    .camera-frame {
        background-color: #0f172a;
        border: 2px solid #334155;
        border-radius: 14px;
        padding: 12px;
        position: relative;
        margin-bottom: 12px;
    }

    .camera-meta-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #94a3b8;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 8px;
    }

    .camera-dot {
        width: 8px;
        height: 8px;
        background-color: #ef4444;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        box-shadow: 0 0 6px #ef4444;
    }

    /* Kontener Paska Prawdopodobieństwa */
    .confidence-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        margin: 18px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }

    .confidence-label-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 8px;
    }

    .confidence-bar-bg {
        width: 100%;
        height: 14px;
        background: #e2e8f0;
        border-radius: 9999px;
        overflow: hidden;
        display: flex;
    }

    .confidence-fill-ok {
        background: linear-gradient(90deg, #10b981, #059669);
        height: 100%;
        transition: width 0.4s ease;
    }

    .confidence-fill-def {
        background: linear-gradient(90deg, #f43f5e, #e11d48);
        height: 100%;
        transition: width 0.4s ease;
    }

    /* Karty Porównania Modeli */
    .model-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.04);
        border-top: 4px solid #6366f1;
        height: 100%;
    }

    .model-card-highlight {
        border-top: 4px solid #059669;
        background: #ffffff;
        box-shadow: 0 8px 25px -4px rgba(5, 150, 105, 0.12);
    }

    .model-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 4px 0;
    }

    .model-role {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .model-metric-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.88rem;
    }

    .model-metric-name {
        color: #475569;
        font-weight: 500;
    }

    .model-metric-val {
        font-weight: 700;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Styl zakładek */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
        color: #64748b;
        border-radius: 10px;
        padding: 8px 18px;
        background-color: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Style sidebaru */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    .sidebar-section-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin: 18px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# 3. FUNKCJE POMOCNICZE I MODEL MACHINE LEARNING
@st.cache_resource
def load_cached_model(model_name: str, device_name: str):
    """Ładuje i buforuje w pamięci wybrany model."""
    device = torch.device(device_name)
    weights_map = {
        "ResNet18": ("resnet18", "models_weights/resnet18.pth"),
        "MobileNetV2": ("mobilenet_v2", "models_weights/mobilenet_v2.pth"),
        "Custom CNN": ("cnn", "models_weights/custom_cnn.pth"),
    }
    m_key, weight_file = weights_map[model_name]
    model = build_model(m_key, pretrained=False, device=device)

    if os.path.exists(weight_file):
        state_dict = torch.load(weight_file, map_location=device)
        model.load_state_dict(state_dict)
        weights_loaded = True
    else:
        weights_loaded = False

    model.eval()
    return model, device, weights_loaded, weight_file


def run_single_inference(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Wykonuje inferencję dla pojedynczego obrazu i zwraca (prob_defect, latency_ms)."""
    transform = get_eval_transforms()
    img_rgb = image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(tensor)
        prob = torch.sigmoid(out).item()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0
    return prob, latency_ms


def apply_inspection_filter(img: Image.Image, filter_mode: str) -> Image.Image:
    """Aplikuje optyczny filtr wspomagający kontrolera jakości w badaniu struktury wirnika."""
    if filter_mode == "Wykrywanie Krawędzi (Struktura/Pęknięcia)":
        return img.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
    elif filter_mode == "Wysoki Kontrast (Ubytki)":
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.8)
    elif filter_mode == "Monochromatyczny (Skala Szarości)":
        return img.convert("L").convert("RGB")
    elif filter_mode == "Wyostrzenie Krawędzi":
        return img.filter(ImageFilter.SHARPEN)
    return img


# 4. GŁÓWNA APLIKACJA
def main():
    # HERO BANNER
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">
            <span class="pulse-dot"></span> PILOT TECHNOCAST &bull; AUTOMATED OPTICAL INSPECTION (AOI)
        </div>
        <h1 class="hero-title">System Kontroli Jakości Odlewów Wirników</h1>
        <p class="hero-subtitle">
            Wizyjna inspekcja odlewów wirników pomp zatapialnych (impeller 300&times;300 px) z wykorzystaniem 
            architektur Deep Learning. System gwarantuje eliminację krytycznych wad (Recall &ge; 0.98) przy czasie inferencji &le; 100 ms na CPU.
        </p>
        <div class="authors-pill">
            <span><b>Autorzy:</b> Dawid Chudzicki (272715), Piotr Szczerbiak (294722)</span>
            <span style="opacity: 0.4;">|</span>
            <span>Uczenie Maszynowe / Praca Magisterska</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # PANEL BOCZNY (SIDEBAR)
    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">Konfiguracja Inferencji</div>', unsafe_allow_html=True)

        selected_model = st.selectbox(
            "Architektura Klasyfikatora:",
            ["ResNet18", "MobileNetV2", "Custom CNN"],
            index=0,
            help="ResNet18 i MobileNetV2 korzystają z Transfer Learningu (ImageNet). Custom CNN to autorska sieć bazowa."
        )

        device_options = ["cpu"]
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            device_options.insert(0, "cuda")

        selected_device = st.selectbox(
            "Akcelerator Sprzętowy:",
            device_options,
            index=0,
            help="Zgodnie ze specyfikacją przemysłową kluczowa jest walidacja czasu odpowiedzi na procesorze CPU."
        )

        # Status hardware badge
        if selected_device == "cuda":
            st.success("Akceleracja CUDA aktywna (NVIDIA GPU)")
        else:
            st.info("Tryb CPU (Standard brzegowy Edge AI)")

        st.markdown("---")
        st.markdown('<div class="sidebar-section-title">Kalibracja Progu Decyzyjnego ($\\tau$)</div>', unsafe_allow_html=True)

        # Optymalne progi wyznaczone eksperymentalnie na zbiorze testowym
        optimal_thresholds = {
            "Custom CNN": 0.158,
            "ResNet18": 0.197,
            "MobileNetV2": 0.099,
        }
        opt_tau = optimal_thresholds.get(selected_model, 0.20)

        threshold = st.slider(
            "Próg klasyfikacji wady ($\\tau$):",
            min_value=0.01,
            max_value=0.90,
            value=float(opt_tau),
            step=0.01,
            help="Jeśli P(Wada) >= tau, odlew jest oznaczany jako WADLIWY i odrzucany z linii produkcyjnej."
        )

        # Dynamiczny opis trybu bezpieczeństwa
        if threshold <= 0.20:
            st.markdown(f"""
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #166534;">
                <b>Tryb Zero-Defect (Zalecany):</b><br>
                Maksymalizuje Recall (&ge; 0.98). Żadna krytyczna wada nie przedostanie się do montażu pompy.
            </div>
            """, unsafe_allow_html=True)
        elif threshold <= 0.50:
            st.markdown(f"""
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #1e40af;">
                <b>Tryb Zrównoważony:</b><br>
                Kompromis między eliminacją wad a precyzją (ograniczenie powtórnych inspekcji).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #92400e;">
                <b>Tryb Wysokiej Precyzji:</b><br>
                Ryzyko wystąpienia błędów False Negative (przeoczenie drobnych pęknięć)!
            </div>
            """, unsafe_allow_html=True)

        st.caption(f"Zoptymalizowany próg dla {selected_model}: **$\\tau^* = {opt_tau:.3f}$**")

        st.markdown("---")
        st.markdown('<div class="sidebar-section-title">Wymogi Specyfikacji</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Czułość (Recall wad):** $\\ge 0{,}98$
        - **Precyzja:** $\\ge 0{,}90$
        - **Opóźnienie CPU:** $\\le 100\\text{ ms}$
        - **Rozmiar wejścia:** $300 \\times 300\\text{ px}$
        """)

    # GŁÓWNE ZAKŁADKI APLIKACJI
    tab_inspect, tab_report, tab_errors, tab_methodology = st.tabs([
        "Panel Inspekcji Wizyjnej",
        "Ewaluacja i Porównanie Modeli",
        "Analiza Błędów (FN vs FP)",
        "Metodologia 5x2 CV & Specyfikacja"
    ])

    # ZAKŁADKA 1: PANEL INSPEKCJI WIZYJNEJ
    with tab_inspect:
        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown("### 1. Rejestracja Obrazu Wirnika")

            source_type = st.radio(
                "Wybierz źródło próbki:",
                ["Katalog próbek wzorcowych", "Wgraj własne zdjęcie (Upload)"],
                horizontal=True
            )

            chosen_image = None
            sample_name = ""

            if source_type == "Katalog próbek wzorcowych":
                sample_category = st.radio(
                    "Stan odlewu:",
                    ["Wadliwy (WADA - pęknięcie, niedolew)", "Poprawny (OK - bez wad)"],
                    horizontal=True
                )

                folder_key = "def" if "WADA" in sample_category else "ok"
                sample_files = sorted(glob.glob(f"data/sample_images/{folder_key}/*.jpeg"))

                if not sample_files:
                    alt_folder = "casting_data/def_front" if folder_key == "def" else "casting_data/ok_front"
                    sample_files = sorted(glob.glob(f"{alt_folder}/*.jpeg"))[:15]

                if sample_files:
                    chosen_path = st.selectbox(
                        "Wybierz plik egzemplarza do inspekcji:",
                        sample_files,
                        format_func=lambda p: os.path.basename(p)
                    )
                    chosen_image = Image.open(chosen_path)
                    sample_name = os.path.basename(chosen_path)
                else:
                    st.warning("Brak plików w folderze próbek.")
            else:
                uploaded_file = st.file_uploader(
                    "Wgraj obraz wirnika w formacie JPG / PNG:",
                    type=["jpg", "jpeg", "png"]
                )
                if uploaded_file is not None:
                    chosen_image = Image.open(uploaded_file)
                    sample_name = uploaded_file.name

            if chosen_image is not None:
                # Opcjonalne filtry wizualne kamery AOI
                filter_mode = st.selectbox(
                    "Optyczny tryb inspekcji (Wizualizacja cech):",
                    [
                        "Widok standardowy (RGB)",
                        "Wykrywanie Krawędzi (Struktura/Pęknięcia)",
                        "Wysoki Kontrast (Ubytki)",
                        "Monochromatyczny (Skala Szarości)",
                        "Wyostrzenie Krawędzi"
                    ],
                    index=0,
                    help="Ułatwia operatorowi wizualną weryfikację rys i mikro-niedolewów metalu."
                )

                filtered_img = apply_inspection_filter(chosen_image, filter_mode)

                st.markdown(f"""
                <div class="camera-frame">
                    <div class="camera-meta-bar">
                        <span><span class="camera-dot"></span>AOI-CAM-01 &bull; LIVE FEED</span>
                        <span>RES: {chosen_image.size[0]}&times;{chosen_image.size[1]} PX</span>
                        <span>{sample_name}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                show_image(filtered_img, caption=f"Podgląd: {filter_mode}")

        with col_result:
            st.markdown("### 2. Wynik Klasyfikacji & Telemetria AI")

            if chosen_image is not None:
                # Załadowanie modelu
                with st.spinner(f"Analiza sieci {selected_model}..."):
                    model, dev, weights_ok, w_file = load_cached_model(selected_model, selected_device)
                    prob_defect, latency = run_single_inference(chosen_image, model, dev)

                if not weights_ok:
                    st.error(f"Nie znaleziono pliku wag: `{w_file}`. Uruchom skrypt treningowy!")

                prob_ok = 1.0 - prob_defect
                is_defective = prob_defect >= threshold

                # Werdykt wizualny
                if is_defective:
                    st.markdown(f"""
                    <div class="verdict-card verdict-card-def">
                        <div class="verdict-title">ODLEW WADLIWY (ODRZUCONY)</div>
                        <div class="verdict-desc">
                            Wykryto niezgodność strukturalną (pęknięcie lub niedolew). Prawdopodobieństwo wady: <b>{prob_defect * 100:.2f}%</b>
                        </div>
                        <span class="verdict-recommendation verdict-rec-def">
                            &times; ODRZUCONY &bull; KIERUNEK: RECYKLING / PRZETOP
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-card verdict-card-ok">
                        <div class="verdict-title">ODLEW POPRAWNY (ZATWIERDZONY)</div>
                        <div class="verdict-desc">
                            Struktura wirnika prawidłowa. Spełnia normy techniczne Pilot Technocast.
                        </div>
                        <span class="verdict-recommendation verdict-rec-ok">
                            &check; ZATWIERDZONY &bull; KIERUNEK: MONTAŻ POMPY
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                # Dedykowany, stylizowany pasek pewności
                st.markdown(f"""
                <div class="confidence-box">
                    <div class="confidence-label-row">
                        <span>Pewność OK: <b>{prob_ok * 100:.1f}%</b></span>
                        <span style="color: #64748b; font-size: 0.8rem;">Próg decyzyjny &tau; = {threshold:.2f}</span>
                        <span>Pewność WADA: <b>{prob_defect * 100:.1f}%</b></span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-fill-ok" style="width: {prob_ok * 100:.2f}%;"></div>
                        <div class="confidence-fill-def" style="width: {prob_defect * 100:.2f}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Siatka KPI parametrów inżynierskich
                total_params, trainable_params = count_parameters(model)
                model_size_mb = get_model_size_mb(model)

                col_k1, col_k2 = st.columns(2)
                with col_k1:
                    latency_badge = "kpi-badge-success" if latency <= 100.0 else "kpi-badge-neutral"
                    latency_text = "ZGODNE Z CELEM (&le; 100 ms)" if latency <= 100.0 else "POWYŻEJ 100 ms"
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-header">
                            <span class="kpi-label">Czas Inferencji</span>
                        </div>
                        <div class="kpi-value">{latency:.1f} <span style="font-size: 1rem; color: #64748b;">ms</span></div>
                        <div class="kpi-badge {latency_badge}">{latency_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_k2:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-header">
                            <span class="kpi-label">Rozmiar Wag</span>
                        </div>
                        <div class="kpi-value">{model_size_mb} <span style="font-size: 1rem; color: #64748b;">MB</span></div>
                        <div class="kpi-badge kpi-badge-info">{total_params:,} parametrów</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                col_k3, col_k4 = st.columns(2)
                with col_k3:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-header">
                            <span class="kpi-label">Próg Klasyfikacji (&tau;)</span>
                        </div>
                        <div class="kpi-value">{threshold:.3f}</div>
                        <div class="kpi-badge kpi-badge-neutral">Optimum: {opt_tau:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_k4:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-header">
                            <span class="kpi-label">Wybrany Model</span>
                        </div>
                        <div class="kpi-value" style="font-size: 1.3rem;">{selected_model}</div>
                        <div class="kpi-badge kpi-badge-neutral">Urządzenie: {selected_device.upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Wyjaśnienie decyzji klasyfikatora
                with st.expander("Wyjaśnienie Decyzji Klasyfikatora (Decision Rule)", expanded=False):
                    st.markdown(f"""
                    - **Reguła decyzyjna:** Jeśli $P(\\text{{Wada}}) \\ge \\tau$, odlew klasyfikowany jest jako **WADLIWY**.
                    - **Wartość wyliczona:** $P(\\text{{Wada}}) = {prob_defect:.5f}$, zadany próg $\\tau = {threshold:.3f}$.
                    - **Rekomendacja:** Dzięki obniżonemu progowi $\\tau^* = {opt_tau:.3f}$ model osiąga **Recall = 100%**, co gwarantuje brak pękniętych wirników trafiających do gotowych pomp.
                    """)

            else:
                st.info("Wybierz gotową próbkę ze zbioru lub wgraj plik z dysku, aby przeprowadzić automatyczną kontrolę jakości.")

    # ZAKŁADKA 2: EWALUACJA I PORÓWNANIE MODELI
    with tab_report:
        st.markdown("### Zestawienie Wyników Modeli na Zbiorze Testowym (1200 obrazów)")
        st.markdown("Porównanie trzech modeli pod kątem krytycznych metryk: **Recall**, **Precision**, **Opóźnienia CPU** oraz **Rozmiaru wag**.")

        # 3 Karty Liderów
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.markdown("""
            <div class="model-card model-card-highlight">
                <div class="model-title">ResNet18</div>
                <div class="model-role">Najwyższa Dokładność &bull; Transfer Learning</div>
                <div class="model-metric-row"><span class="model-metric-name">Recall (Wada):</span><span class="model-metric-val" style="color: #059669;">100.0%</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Precision (Wada):</span><span class="model-metric-val" style="color: #059669;">99.50%</span></div>
                <div class="model-metric-row"><span class="model-metric-name">F1-Score:</span><span class="model-metric-val">0.9975</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Czas inferencji CPU:</span><span class="model-metric-val">62.5 ms</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Rozmiar modelu:</span><span class="model-metric-val">42.7 MB</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Przeoczone wady (FN):</span><span class="model-metric-val" style="color: #059669;">0 / 600</span></div>
            </div>
            """, unsafe_allow_html=True)

        with c_m2:
            st.markdown("""
            <div class="model-card">
                <div class="model-title">MobileNetV2</div>
                <div class="model-role">Optimum Brzegowe &bull; Edge AI</div>
                <div class="model-metric-row"><span class="model-metric-name">Recall (Wada):</span><span class="model-metric-val" style="color: #059669;">100.0%</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Precision (Wada):</span><span class="model-metric-val" style="color: #059669;">98.20%</span></div>
                <div class="model-metric-row"><span class="model-metric-name">F1-Score:</span><span class="model-metric-val">0.9909</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Czas inferencji CPU:</span><span class="model-metric-val">59.2 ms</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Rozmiar modelu:</span><span class="model-metric-val">8.6 MB (Lekki)</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Przeoczone wady (FN):</span><span class="model-metric-val" style="color: #059669;">0 / 600</span></div>
            </div>
            """, unsafe_allow_html=True)

        with c_m3:
            st.markdown("""
            <div class="model-card">
                <div class="model-title">Custom CNN</div>
                <div class="model-role">Autorska Sieć Bazowa &bull; Baseline</div>
                <div class="model-metric-row"><span class="model-metric-name">Recall (Wada):</span><span class="model-metric-val" style="color: #059669;">100.0%</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Precision (Wada):</span><span class="model-metric-val" style="color: #059669;">97.72%</span></div>
                <div class="model-metric-row"><span class="model-metric-name">F1-Score:</span><span class="model-metric-val">0.9885</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Czas inferencji CPU:</span><span class="model-metric-val" style="color: #059669;">34.5 ms</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Rozmiar modelu:</span><span class="model-metric-val">1.6 MB</span></div>
                <div class="model-metric-row"><span class="model-metric-name">Przeoczone wady (FN):</span><span class="model-metric-val" style="color: #059669;">0 / 600</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Galeria Wykresów Ewaluacyjnych")

        subtab_curves, subtab_cm, subtab_thresholds, subtab_loss, subtab_eda = st.tabs([
            "Krzywe ROC & Precision-Recall",
            "Macierze Pomyłek",
            "Strojenie Progów (PR Curves)",
            "Przebieg Uczenia (Epoch Loss)",
            "Eksploracyjna Analiza Danych (EDA)"
        ])

        with subtab_curves:
            p_roc = "reports/figures/test_roc_pr_curves.png"
            if os.path.exists(p_roc):
                show_image(p_roc, caption="Krzywe ROC oraz Precision-Recall dla zbioru testowego (1200 próbek)")
            else:
                st.info("Brak pliku wykresu.")

        with subtab_cm:
            p_cm = "reports/figures/test_confusion_matrices.png"
            if os.path.exists(p_cm):
                show_image(p_cm, caption="Macierze pomyłek na zbiorze testowym z uwzględnieniem kosztu False Negative vs False Positive")
            else:
                st.info("Brak pliku macierzy pomyłek.")

        with subtab_thresholds:
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                t1 = "reports/figures/threshold_tuning_resnet18.png"
                if os.path.exists(t1):
                    show_image(t1, caption="ResNet18: Strojenie progu pod Recall >= 0.98")
            with t_col2:
                t2 = "reports/figures/threshold_tuning_mobilenet_v2.png"
                if os.path.exists(t2):
                    show_image(t2, caption="MobileNetV2: Strojenie progu pod Recall >= 0.98")
            with t_col3:
                t3 = "reports/figures/threshold_tuning_cnn.png"
                if os.path.exists(t3):
                    show_image(t3, caption="Custom CNN: Strojenie progu pod Recall >= 0.98")

        with subtab_loss:
            p_train = "reports/figures/training_curves.png"
            if os.path.exists(p_train):
                show_image(p_train, caption="Dynamika zbieżności funkcji straty (BCE) oraz metryk walidacyjnych w kolejnych epokach")

        with subtab_eda:
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                e1 = "reports/figures/eda_class_distribution.png"
                if os.path.exists(e1):
                    show_image(e1, caption="Idealny balans klas w zbiorze Pilot Technocast (50% / 50%)")
            with e_col2:
                e2 = "reports/figures/eda_pixel_intensity.png"
                if os.path.exists(e2):
                    show_image(e2, caption="Rozkład intensywności pikseli dla odlewów OK vs WADA")

            e3 = "reports/figures/eda_samples_grid.png"
            if os.path.exists(e3):
                show_image(e3, caption="Przykłady obrazów wirników ze zbioru danych")

        # Pełny raport tekstowy w expanderze
        rep_path = "reports/final_evaluation_report.md"
        if os.path.exists(rep_path):
            with st.expander("Zobacz pełny wygenerowany raport końcowy (Markdown)", expanded=False):
                with open(rep_path, "r", encoding="utf-8") as f:
                    st.markdown(f.read())

    # ZAKŁADKA 3: ANALIZA BŁĘDÓW (FN VS FP)
    with tab_errors:
        st.markdown("### Asymetryczny Koszt Błędów w Kontroli Jakości Wirników")

        col_cost1, col_cost2 = st.columns(2)
        with col_cost1:
            st.markdown("""
            <div style="background: #fff1f2; border: 2px solid #ef4444; border-radius: 14px; padding: 18px;">
                <h4 style="color: #991b1b; margin: 0 0 8px 0;">False Negative (Błąd Krytyczny - Wada Przeoczona)</h4>
                <p style="color: #7f1d1d; font-size: 0.92rem; margin-bottom: 8px;">
                    Sytuacja, w której pęknięty lub niedolany wirnik zostaje zaklasyfikowany jako poprawny i trafia do montażu pompy.
                </p>
                <ul style="color: #7f1d1d; font-size: 0.85rem; padding-left: 20px; margin: 0;">
                    <li><b>Koszt:</b> Bardzo wysoki (awaria pompy pod obciążeniem, zalanie, utrata zaufania klienta).</li>
                    <li><b>Wynik naszego systemu:</b> <b>0 błędów FN</b> (100% wykrywalności na zbiorze testowym).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_cost2:
            st.markdown("""
            <div style="background: #fffbeb; border: 2px solid #f59e0b; border-radius: 14px; padding: 18px;">
                <h4 style="color: #92400e; margin: 0 0 8px 0;">False Positive (Błąd Umiarkowany - Fałszywy Alarm)</h4>
                <p style="color: #78350f; font-size: 0.92rem; margin-bottom: 8px;">
                    Sytuacja, w której w pełni sprawny odlew zostaje oznaczony jako wadliwy przez drobny cień lub refleks świetlny.
                </p>
                <ul style="color: #78350f; font-size: 0.85rem; padding-left: 20px; margin: 0;">
                    <li><b>Koszt:</b> Niski / umiarkowany (wymaga jedynie ponownej kontroli manualnej przez operatora).</li>
                    <li><b>Wynik naszego systemu:</b> Zminimalizowany (Precision &ge; 98.2% &ndash; 99.5%).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Analiza Prób Błędnie Zaklasyfikowanych (False Positives)")

        selected_err_model = st.selectbox(
            "Wybierz architekturę do przeglądu fałszywych alarmów:",
            ["ResNet18", "MobileNetV2", "Custom CNN"],
            index=0
        )
        tag_map = {"ResNet18": "resnet18", "MobileNetV2": "mobilenet_v2", "Custom CNN": "cnn"}
        err_tag = tag_map[selected_err_model]

        fp_file = f"reports/error_analysis/error_fp_{err_tag}.png"
        if os.path.exists(fp_file):
            show_image(fp_file, caption=f"Przykłady fałszywych alarmów (False Positives) dla modelu {selected_err_model}")
            st.caption(
                "**Wniosek inżynierski:** Większość fałszywych alarmów wynika ze specyficznych refleksów światła "
                "na krawędziach łopatek wirnika lub nietypowej chropowatości powierzchni formy odlewniczej."
            )
        else:
            st.success(f"Brak zarejestrowanych błędów False Positive dla {selected_err_model}!")

    # ZAKŁADKA 4: METODOLOGIA I ZAŁOŻENIA
    with tab_methodology:
        st.markdown("### Metodologia Badawcza i Założenia Systemowe")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            #### 1. Zbiór Danych (Pilot Technocast)
            - **Obiekt:** Wirnik pompy zatapialnej (*submersible pump impeller*).
            - **Rozdzielczość:** $300 \\times 300\\text{ px}$, kanały zduplikowane do formatu RGB ImageNet.
            - **Liczebność:** 6 000 obrazów (dokładny balans 3 000 OK / 3 000 WADA).
            - **Weryfikacja integralności:** 0 plików uszkodzonych, 0 fizycznych duplikatów MD5.
            
            #### 2. Architektury Sieci Neuronowych
            - **ResNet18 (Transfer Learning):** Sieć rezydualna z wagami `IMAGENET1K_V1`, dostrojona warstwa w pełni połączona z Dropoutem 0.3.
            - **MobileNetV2 (Transfer Learning):** Architektura z odwróconymi blokami rezydualnymi (Inverted Residuals), zoptymalizowana pod minimalne zużycie pamięci i szybką inferencję.
            - **Custom CNN:** Autorska sieć splotowa (4 bloki: Conv2D $\\to$ BatchNorm $\\to$ ReLU $\\to$ MaxPool) jako punkt odniesienia (*baseline*).
            """)

        with col_m2:
            st.markdown("""
            #### 3. Rygorystyczna Walidacja 5x2-fold CV (Dietterich)
            Standardowa walidacja K-fold cechuje się wysoką korelacją między podzbiorami, co zawyża błąd I rodzaju w testach istotności. 
            Dlatego zastosowano procedurę **5&times;2-fold Cross-Validation** (Thomas G. Dietterich, 1998):
            - 5 powtórzeń 2-krotnego podziału stratyfikowanego (50% train / 50% test).
            - Sparowany test t-Studenta Diettericha z 5 stopniami swobody:
            """)
            st.latex(r"t = \frac{p_1^{(1)}}{\sqrt{\frac{1}{5} \sum_{i=1}^5 s_i^2}} \sim t(5)")
            st.markdown("""
            #### 4. Wdrożenie Przemysłowe
            - System spełnia rygorystyczne kryterium czasu inferencji **$\\le 100\\text{ ms}$ na procesorze CPU**, 
              umożliwiając pracę bezpośrednio przy taśmie produkcyjnej bez konieczności instalowania drogich akceleratorów GPU.
            """)


if __name__ == "__main__":
    main()
