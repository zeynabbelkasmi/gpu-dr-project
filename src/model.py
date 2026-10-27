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


def resnet18(num_classes, pretrained=True):
    """A6.6: torchvision ResNet-18, ImageNet weights, last layer replaced.

    Why this model: (1) residual connections and batch norm train deeper features than the
    3-block reference CNN, which matters for small DR lesions (microaneurysms, exudates);
    (2) ImageNet pre-training compensates for the small training set (~2,100 images) and
    the inputs are already normalized with the ImageNet mean/std (preprocess normalize);
    (3) it stays lightweight: 11.2 M parameters and 3.6 GFLOPs (1.8 GMACs) per 224x224 image,
    against 0.6 M and 1.0 GFLOPs for the reference CNN, and its convolutions are
    cuDNN/Tensor-Core friendly, so AMP and channels_last can pay off.
    Parameters and FLOPs of both models: python -m src.model
    """
    import torchvision
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def resnet50(num_classes, pretrained=True):
    """Round 2: deeper residual network (23.5 M parameters, 8.2 GFLOPs with a 5-class head), same ImageNet recipe as ResNet-18."""
    import torchvision
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = torchvision.models.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def efficientnet_b0(num_classes, pretrained=True):
    """Round 2: EfficientNet-B0 (4.0 M parameters, 0.8 GFLOPs with a 5-class head), better ImageNet accuracy per FLOP than
    ResNet-18; depthwise convolutions are less Tensor-Core friendly, so it may not be faster."""
    import torchvision
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.efficientnet_b0(weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


MODELS = {"resnet18": resnet18, "resnet50": resnet50, "efficientnet_b0": efficientnet_b0}


def build_model(name, num_classes, pretrained=True):
    if name == "reference_cnn":
        return ReferenceCNN(num_classes)
    if name in MODELS:
        return MODELS[name](num_classes, pretrained)
    raise ValueError(f"unknown model: {name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, size=(224, 224)):
    """Multiply-accumulates x 2 of the conv and linear layers for one image (forward pass)."""
    import torch
    flops = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            k = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
            flops.append(2 * k * out.numel())
        elif isinstance(mod, nn.Linear):
            flops.append(2 * mod.in_features * mod.out_features)

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, *size))
    for h in hs:
        h.remove()
    return sum(flops)


if __name__ == "__main__":
    for name in ("reference_cnn", "resnet18", "efficientnet_b0", "resnet50"):
        m = build_model(name, 5, pretrained=False)
        print(f"{name:14s} params={count_params(m) / 1e6:6.2f} M  "
              f"FLOPs={count_flops(m) / 1e9:6.2f} G per 224x224 image")
