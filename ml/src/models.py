"""
Architektury sieci neuronowych do binarnej klasyfikacji odlewów:
1. Custom CNN (autorska architektura od podstaw)
2. ResNet18 (Transfer Learning)
3. MobileNetV2 (Transfer Learning)
"""

import os
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models
try:
    from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
    HAS_NEW_WEIGHTS = True
except ImportError:
    HAS_NEW_WEIGHTS = False


class CustomCNN(nn.Module):
    """
    Własna architektura sieci splotowej zaprojektowana od podstaw.
    Zawiera 4 bloki konwolucyjne z Batch Normalization, aktywacją ReLU,
    warstwami Max Pooling oraz Dropoutem zapobiegającym przeuczeniu.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, dropout_rate: float = 0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Blok 1: 300x300 -> 150x150
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 2: 150x150 -> 75x75
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 3: 75x75 -> 37x37
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 4: 37x37 -> 18x18
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.global_pool(feat)
        out = self.classifier(pooled)
        return out.squeeze(-1)  # Kształt (B,)


def build_resnet18(pretrained: bool = True, num_classes: int = 1, dropout_rate: float = 0.3) -> nn.Module:
    """Tworzy model ResNet18 z dostosowaną warstwą wyjściową (Transfer Learning)."""
    if pretrained:
        weights = ResNet18_Weights.DEFAULT if HAS_NEW_WEIGHTS else True
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    # Definiujemy forward tak, aby zwracał (B,)
    orig_forward = model.forward
    def forward_wrapper(x: torch.Tensor) -> torch.Tensor:
        out = orig_forward(x)
        return out.squeeze(-1)
    model.forward = forward_wrapper

    return model


def build_mobilenet_v2(pretrained: bool = True, num_classes: int = 1, dropout_rate: float = 0.3) -> nn.Module:
    """Tworzy model MobileNetV2 z dostosowaną warstwą klasyfikującą (Transfer Learning)."""
    if pretrained:
        weights = MobileNet_V2_Weights.DEFAULT if HAS_NEW_WEIGHTS else True
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    orig_forward = model.forward
    def forward_wrapper(x: torch.Tensor) -> torch.Tensor:
        out = orig_forward(x)
        return out.squeeze(-1)
    model.forward = forward_wrapper

    return model


def build_model(
    model_name: str,
    pretrained: bool = True,
    device: str = "cpu",
    dropout_rate: float = 0.3
) -> nn.Module:
    """
    Fabryka modeli zwracająca odpowiednią architekturę na wybranym urządzeniu.
    Dozwolone nazwy: 'cnn', 'resnet18', 'mobilenet_v2'.
    """
    name_clean = model_name.strip().lower()

    if name_clean in ["cnn", "custom_cnn", "customcnn"]:
        model = CustomCNN(dropout_rate=dropout_rate)
    elif name_clean in ["resnet", "resnet18"]:
        model = build_resnet18(pretrained=pretrained, dropout_rate=dropout_rate)
    elif name_clean in ["mobilenet", "mobilenet_v2", "mobilenetv2"]:
        model = build_mobilenet_v2(pretrained=pretrained, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Nieznana nazwa modelu: {model_name}. Dozwolone: 'cnn', 'resnet18', 'mobilenet_v2'.")

    return model.to(device)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Zwraca: (liczba_wszystkich_parametrów, liczba_trenowalnych_parametrów)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model: nn.Module) -> float:
    """Oblicza orientacyjny rozmiar wag modelu w megabajtach (MB)."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return round(size_all_mb, 2)
