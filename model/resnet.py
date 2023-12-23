from torchvision.models import resnet101
from torchvision.models.segmentation.deeplabv3 import _deeplabv3_resnet, \
    DeepLabV3


def deeplabv3_resnet101(
        *,
        progress: bool = True,
        num_classes=None,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    # from torchvision.models.segmentation.deeplabv3 import DeepLabV3
    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
        :members:
    """

    backbone = resnet101(weights=None,
                         replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, True)

    return model
