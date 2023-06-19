from torch import nn

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2


def MaskRCNN(in_channels=5, num_classes=2, trainable_backbone_layers=5, image_mean=None, image_std=None, **kwargs):
    if image_mean is None:
        # image_mean = [0.485, 0.456, 0.406, 0.5, 0.5]
        image_mean = [130.0, 135.0, 135.0, 118.0, 118.0] / 255.0
    if image_std is None:
        # image_std = [0.229, 0.224, 0.225, 0.225, 0.225]
        image_std = [44.0, 40.0, 40.0, 30.0, 21.0] / 255.0

    model = maskrcnn_resnet50_fpn_v2(
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
        image_mean=image_mean,
        image_std=image_std
    )
    model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).requires_grad_(True)

    return model
