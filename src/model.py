"""Model zoo. The reference CNN is given; the alternative is part of assignment A6."""
import torch.nn as nn


class ReferenceCNN(nn.Module):
    """Small reference CNN: 3 conv blocks + MLP head (for 224x224 input)."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), # 28
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),   # resolution-agnostic: works for any input size
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, num_classes):
    if name == "reference_cnn":
        return ReferenceCNN(num_classes)
    # TODO(A6.6): add a justified alternative, for example name == "resnet18":
    #   a torchvision ResNet-18 with ImageNet weights and the last layer replaced by
    #   nn.Linear(in_features, num_classes). Or choose another lightweight model and
    #   justify the choice in the report. Count parameters and FLOPs for both models.
    #   Remember: change ONE variable at a time when you compare against the reference CNN.
    raise ValueError(f"unknown model: {name}")
