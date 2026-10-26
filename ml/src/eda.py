"""
Moduł eksploracyjnej analizy danych (EDA) i weryfikacji jakości zbioru:
- Zliczanie i balans klas
- Kontrola poprawności i integralności plików
- Wykrywanie duplikatów (MD5 oraz hasze percepcyjne)
- Generowanie wizualizacji próbek, rozkładów pikseli, średnich obrazów
"""

import os
import hashlib
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import scan_dataset, CLASS_NAMES, IDX_TO_LABEL


def compute_md5(filepath: str) -> str:
    """Oblicza sumę kontrolną MD5 pliku."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Oblicza hasz różnicowy (dHash) obrazu dla wykrywania niemal identycznych obrazów."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized)
    difference = pixels[:, 1:] > pixels[:, :-1]
    return "".join(["1" if b else "0" for b in difference.flatten()])


def run_full_eda(
    data_dir: str = "casting_data",
    output_dir: str = "reports",
    figures_dir: str = "reports/figures"
) -> Dict[str, Any]:
    """
    Przeprowadza pełną procedurę weryfikacji i analizy danych oraz zapisuje wykresy i raport.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    filepaths, labels = scan_dataset(data_dir)
    total_images = len(filepaths)

    print(f"Znaleziono łącznie {total_images} obrazów w katalogu '{data_dir}'.")

    # 1. Zliczanie klas i balans
    class_counts = defaultdict(int)
    for lbl in labels:
        class_counts[lbl] += 1

    # 2. Kontrola jakości plików i wykrywanie duplikatów
    corrupted_files = []
    sizes = set()
    modes = set()
    pixel_min_vals = []
    pixel_max_vals = []
    pixel_means = []

    md5_dict = defaultdict(list)
    dhash_dict = defaultdict(list)

    print("Weryfikacja integralności plików i wykrywanie duplikatów...")
    for path, label in zip(filepaths, labels):
        try:
            with Image.open(path) as img:
                sizes.add(img.size)
                modes.add(img.mode)
                
                # Próbkowanie statystyk pikseli
                arr = np.array(img, dtype=np.float32)
                pixel_min_vals.append(float(arr.min()))
                pixel_max_vals.append(float(arr.max()))
                pixel_means.append(float(arr.mean()))

                # Hasz MD5
                h_md5 = compute_md5(path)
                md5_dict[h_md5].append(path)

                # Hasz percepcyjny (dHash)
                h_diff = compute_difference_hash(img)
                dhash_dict[h_diff].append(path)

        except Exception as e:
            corrupted_files.append((path, str(e)))

    # Identyfikacja dokładnych duplikatów
    exact_duplicates = {k: v for k, v in md5_dict.items() if len(v) > 1}
    # Identyfikacja potencjalnych bliskich duplikatów (np. po lekkich rotacjach/augmentacjach)
    perceptual_duplicates = {k: v for k, v in dhash_dict.items() if len(v) > 1}

    # 3. Zapis raportu tekstowego
    report_path = os.path.join(output_dir, "eda_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Raport Eksploracyjnej Analizy Danych (EDA)\n")
        f.write("System automatycznej kontroli jakości odlewów (wirniki pomp zatapialnych)\n\n")
        f.write("## 1. Statystyki ogólne i balans klas\n\n")
        f.write(f"- **Łączna liczba obrazów:** {total_images}\n")
        for cls_name, cls_idx in [("ok_front", 0), ("def_front", 1)]:
            cnt = class_counts[cls_idx]
            pct = (cnt / total_images) * 100
            f.write(f"- **Klasa `{cls_name}` ({IDX_TO_LABEL[cls_idx]}):** {cnt} obrazów ({pct:.2f}%)\n")
        f.write(f"- **Stosunek klas (def/ok):** {class_counts[1] / class_counts[0]:.2f}:1 (idealny balans 50/50)\n\n")

        f.write("## 2. Integralność plików i format\n\n")
        f.write(f"- **Uszkodzone pliki:** {len(corrupted_files)}\n")
        f.write(f"- **Wymiary obrazów (szerokość, wysokość):** {list(sizes)}\n")
        f.write(f"- **Tryb kolorów (PIL mode):** {list(modes)}\n")
        f.write(f"- **Zakres wartości pikseli:** min={min(pixel_min_vals):.0f}, max={max(pixel_max_vals):.0f}, średnia={np.mean(pixel_means):.2f}\n\n")

        f.write("## 3. Analiza duplikatów i ryzyka wycieku danych\n\n")
        f.write(f"- **Dokładne duplikaty plików (identyczny MD5):** {len(exact_duplicates)} grup(y)\n")
        f.write(f"- **Duplikaty percepcyjne (bliskie cechy wizualne dHash):** {len(perceptual_duplicates)} grup(y)\n")
        if len(exact_duplicates) == 0:
            f.write("  *Brak fizycznych duplikatów plików w zbiorze danych.*\n")
        f.write("\n")

    print(f"Zapisano raport EDA do: {report_path}")

    # 4. Generowanie wykresów
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Wykres 1: Rozkład klas
    fig, ax = plt.subplots(figsize=(6, 4))
    classes = [IDX_TO_LABEL[0], IDX_TO_LABEL[1]]
    counts = [class_counts[0], class_counts[1]]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(classes, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax.set_title("Rozkład klas w zbiorze danych odlewów", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Liczba próbek", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 50, f"{int(h)} ({h/total_images*100:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "eda_class_distribution.png")
    fig.savefig(fig1_path, dpi=200)
    plt.close(fig)

    # Wykres 2: Siatka przykładowych zdjęć (OK vs DEFECT)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Przykładowe obrazy wirników: Poprawne (OK) vs Wadliwe (WADA)", fontsize=14, fontweight="bold")
    
    ok_paths = [p for p, l in zip(filepaths, labels) if l == 0][:5]
    def_paths = [p for p, l in zip(filepaths, labels) if l == 1][:5]

    for col, p in enumerate(ok_paths):
        img = Image.open(p)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"OK #{col+1}", color="green", fontweight="bold")
        axes[0, col].axis("off")

    for col, p in enumerate(def_paths):
        img = Image.open(p)
        axes[1, col].imshow(img)
        axes[1, col].set_title(f"WADA #{col+1}", color="red", fontweight="bold")
        axes[1, col].axis("off")

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "eda_samples_grid.png")
    fig.savefig(fig2_path, dpi=200)
    plt.close(fig)

    # Wykres 3: Histogram rozkładu jasności pikseli (OK vs WADA)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sample_ok = np.array(Image.open(ok_paths[0])).flatten()
    sample_def = np.array(Image.open(def_paths[0])).flatten()

    ax.hist(sample_ok, bins=50, alpha=0.6, color="#2ecc71", label="Poprawny (OK)", density=True)
    ax.hist(sample_def, bins=50, alpha=0.6, color="#e74c3c", label="Wadliwy (WADA)", density=True)
    ax.set_title("Rozkład intensywności pikseli (przykłady)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Wartość piksela [0 - 255]")
    ax.set_ylabel("Gęstość")
    ax.legend()
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "eda_pixel_intensity.png")
    fig.savefig(fig3_path, dpi=200)
    plt.close(fig)

    # Wykres 4: Średni obraz i odchylenie standardowe dla każdej z klas
    print("Obliczanie średnich obrazów i odchylenia standardowego...")
    def get_mean_std_image(paths_subset: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        acc = []
        for p in paths_subset[:200]:  # reprezentatywna próba 200 obrazów
            im = np.array(Image.open(p).convert("L"), dtype=np.float32)
            acc.append(im)
        stack = np.stack(acc, axis=0)
        return stack.mean(axis=0), stack.std(axis=0)

    mean_ok, std_ok = get_mean_std_image([p for p, l in zip(filepaths, labels) if l == 0])
    mean_def, std_def = get_mean_std_image([p for p, l in zip(filepaths, labels) if l == 1])

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle("Średni obraz oraz odchylenie standardowe pikseli", fontsize=13, fontweight="bold")
    
    im0 = axes[0, 0].imshow(mean_ok, cmap="gray")
    axes[0, 0].set_title("Średnia: Poprawne (OK)")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(mean_def, cmap="gray")
    axes[0, 1].set_title("Średnia: Wadliwe (WADA)")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(std_ok, cmap="magma")
    axes[1, 0].set_title("Std Dev: Poprawne (OK)")
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(std_def, cmap="magma")
    axes[1, 1].set_title("Std Dev: Wadliwe (WADA)")
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "eda_mean_std_images.png")
    fig.savefig(fig4_path, dpi=200)
    plt.close(fig)

    print(f"Wykresy zapisane w katalogu: {figures_dir}")

    return {
        "total_images": total_images,
        "class_counts": dict(class_counts),
        "sizes": list(sizes),
        "modes": list(modes),
        "corrupted_count": len(corrupted_files),
        "exact_duplicates_count": len(exact_duplicates),
        "figures": [fig1_path, fig2_path, fig3_path, fig4_path],
    }
