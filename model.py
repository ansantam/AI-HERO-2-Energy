from torch import nn
import numpy as np
from torchvision.models.detection import MaskRCNN as torchMaskRCNN
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor

def MaskRCNN(in_channels=5, num_classes=2, trainable_backbone_layers=3, image_mean=None, image_std=None, weights=None, backbone:str="resnet50"):
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406, 0.5, 0.5] # ResNet mean
        # our mean [0.4924735  0.47618312 0.47336233 0.47657516 0.47533598]
        # image_mean = np.array([130.0, 135.0, 135.0, 118.0, 118.0],dtype=float) / 255.0
    if image_std is None:
        image_std = [0.229, 0.224, 0.225, 0.225, 0.225]
        # our std [0.19806709 0.20018329 0.19319048 0.14501901 0.08164792]
        # image_std = np.array([44.0, 40.0, 40.0, 30.0, 21.0],dtype=float) / 255.0
    if in_channels == 3:
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

    if weights is None:
        if backbone == 'resnet50':
            backbone_model = resnet50()
        elif backbone == 'resnet18':
            backbone_model = resnet18()
    elif weights == 'DEFAULT':
        if backbone == 'resnet50':
            backbone_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            # backbone_model.out_channels = 256
        elif backbone == 'resnet18':
            backbone_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            # backbone_model.out_channels = 256
    backbone_model = _resnet_fpn_extractor(backbone_model, trainable_layers=trainable_backbone_layers)
    model = torchMaskRCNN(
        backbone=backbone_model,
        num_classes=num_classes,
        image_mean=image_mean,
        image_std=image_std,
    )

    # model = maskrcnn_resnet50_fpn_v2(
    #     weights_backbone=backbone_weight,  # ResNet50 weights
    #     num_classes=num_classes,
    #     trainable_backbone_layers=trainable_backbone_layers,
    #     image_mean=image_mean,
    #     image_std=image_std
    # )
    if in_channels != 3:
        model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).requires_grad_(True)

    return model
