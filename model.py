from torch import nn
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import ResNet50_Weights

def MaskRCNN(in_channels=5, num_classes=2, trainable_backbone_layers=3, image_mean=None, image_std=None, weights=None, **kwargs):
    if image_mean is None:
        # image_mean = [0.485, 0.456, 0.406, 0.5, 0.5]
        image_mean = np.array([130.0, 135.0, 135.0, 118.0, 118.0],dtype=float) / 255.0
    if image_std is None:
        # image_std = [0.229, 0.224, 0.225, 0.225, 0.225]
        image_std = np.array([44.0, 40.0, 40.0, 30.0, 21.0],dtype=float) / 255.0

    if weights is None:
        backbone_weight = None
    elif weights == 'DEFAULT':
        backbone_weight = ResNet50_Weights.DEFAULT

    model = maskrcnn_resnet50_fpn_v2(
        weights_backbone=backbone_weight,  # ResNet50 weights
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
        image_mean=image_mean,
        image_std=image_std
    )
    model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).requires_grad_(True)

    return model
