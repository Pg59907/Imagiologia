"""
Model factory for ERCP classification.

Uses the `timm` library to load pretrained CNN/ViT backbones and
adapts the classification head to 4 classes.

Recommended models to try (in order):
    tf_efficientnetv2_s   – best accuracy/speed trade-off  ← default
    convnext_tiny         – modern architecture, often beats EfficientNet
    tf_efficientnetv2_m   – larger, more accurate, slower
    resnet50              – matches the baseline architecture
    densenet121           – baseline architecture
"""

import torch
import torch.nn as nn
import timm


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.3,
) -> nn.Module:
    """
    Create a timm model with a custom classification head.

    Args:
        model_name  : any valid timm model string (run `timm.list_models()`)
        num_classes : number of output classes (4 for ERCP)
        pretrained  : load ImageNet weights
        drop_rate   : dropout before the classifier head

    Returns:
        nn.Module ready for training
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Return the target layer for Grad-CAM visualisation.

    The target layer should be the *last convolutional feature-extraction
    block* before the global pooling / classifier head.  Grad-CAM computes
    gradient-weighted class activation maps from the feature maps at this
    layer, so choosing it correctly is critical.

    Args:
        model      : instantiated timm model
        model_name : timm model name string (used to select the right layer)

    Returns:
        nn.Module  – the target layer for hook registration
    """
    name = model_name.lower()

    if "efficientnet" in name:
        # EfficientNet / EfficientNetV2: last MBConv block sequence
        return model.blocks[-1]

    elif "convnext" in name:
        # ConvNeXt: last stage
        return model.stages[-1]

    elif "resnet" in name or "resnext" in name or "wide_resnet" in name:
        # ResNet family: last residual layer
        return model.layer4

    elif "densenet" in name:
        # DenseNet: last dense block
        return model.features.denseblock4

    elif "mobilenet" in name:
        # MobileNet: last inverted-residual block
        return model.blocks[-1]

    elif "vit" in name or "deit" in name or "swin" in name:
        # Vision Transformers: last transformer block
        # Note: standard Grad-CAM produces rough maps for ViTs.
        # Consider Attention Rollout for better results.
        if hasattr(model, "blocks"):
            return model.blocks[-1]
        elif hasattr(model, "layers"):
            return model.layers[-1]
        else:
            raise ValueError(f"Cannot auto-detect target layer for ViT model: {model_name}")

    else:
        # Generic fallback: walk all modules and return the last Conv2d
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise ValueError(
                f"Cannot find a Conv2d layer for Grad-CAM in model: {model_name}. "
                "Please specify the target layer manually."
            )
        print(f"[WARNING] Using generic Conv2d fallback for Grad-CAM target layer.")
        return last_conv
