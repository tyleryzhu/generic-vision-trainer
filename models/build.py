from functools import partial
import torch.nn as nn

from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .revvit import ReversibleViT
from .vit import VisionTransformer


def build_model(config):
    "Model builder."

    model_type = config.MODEL.NAME

    if model_type == "resnet18":
        model = ResNet18(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == "resnet34":
        model = ResNet34(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == "resnet50":
        model = ResNet50(num_classes=config.MODEL.NUM_CLASSES)
    elif model_type == "vit":
        model = VisionTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=config.MODEL.NUM_CLASSES,
        )
    elif model_type == "revvit":
        model = ReversibleViT(
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            depth=config.MODEL.VIT.DEPTH,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            image_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            enable_amp=config.TRAIN.ENABLE_AMP,
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model
