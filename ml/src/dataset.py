"""
Dataset loader and preprocessing transforms for casting inspection dataset.
"""

import os
from typing import List, Tuple, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Definiujemy mapowanie klas:
# 0 -> ok_front (odlew bez wad, klasa negatywna)
# 1 -> def_front (odlew wadliwy, klasa pozytywna w kontekście wykrywania wad)
CLASS_NAMES = ["ok_front", "def_front"]
CLASS_TO_IDX = {"ok_front": 0, "def_front": 1}
IDX_TO_LABEL = {0: "Poprawny (OK)", 1: "Wadliwy (DEF)"}

# Średnia i odchylenie standardowe ImageNet dla normalizacji
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: Tuple[int, int] = (300, 300)) -> transforms.Compose:
    """Zwraca potok transformacji i augmentacji dla zbioru treningowego."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: Tuple[int, int] = (300, 300)) -> transforms.Compose:
    """Zwraca potok transformacji dla walidacji, testu oraz inferencji."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Odwraca normalizację tensora (C, H, W) do zakresu [0, 1] dla wizualizacji."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0.0, 1.0)


def scan_dataset(data_dir: str = "casting_data") -> Tuple[List[str], List[int]]:
    """
    Skanuje folder ze zbiorem danych i zwraca listę ścieżek do plików oraz odpowiadające im etykiety.
    """
    filepaths = []
    labels = []

    for class_name, label in CLASS_TO_IDX.items():
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_folder):
            raise FileNotFoundError(f"Katalog klasy {class_folder} nie istnieje!")
        
        filenames = sorted(os.listdir(class_folder))
        for fname in filenames:
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                full_path = os.path.join(class_folder, fname)
                filepaths.append(full_path)
                labels.append(label)

    return filepaths, labels


class CastingDataset(Dataset):
    """
    PyTorch Dataset do wczytywania obrazów odlewów.
    Zapewnia konwersję do 3-kanałowego formatu RGB (zgodnie z wymaganiami modeli pretrenowanych).
    """

    def __init__(
        self,
        filepaths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None
    ):
        self.filepaths = list(filepaths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.filepaths[idx]
        label = self.labels[idx]

        # Wczytujemy obraz PIL i upewniamy się, że ma 3 kanały RGB
        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, label, path
