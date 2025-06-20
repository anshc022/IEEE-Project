"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn

"""Model head modules."""
__all__ = ("Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect")
class Detect(nn.Module):
    """YOLO Detect head for detection models."""
    dynamic = ...
    export = ...
    format = ...
    end2end = ...
    max_det = ...
    shape = ...
    anchors = ...
    strides = ...
    legacy = ...
    def __init__(self, nc=..., ch=...) -> None:
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        ...
    
    def forward(self, x): # -> dict[str, Any] | Tensor | tuple[Tensor, dict[str, Any]] | tuple[Tensor, Tensor | Any] | tuple[tuple[Tensor, Tensor | Any] | Tensor, Any]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        ...
    
    def forward_end2end(self, x): # -> dict[str, Any] | Tensor | tuple[Tensor, dict[str, Any]]:
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        ...
    
    def bias_init(self): # -> None:
        """Initialize Detect() biases, WARNING: requires stride availability."""
        ...
    
    def decode_bboxes(self, bboxes, anchors, xywh=...): # -> Tensor:
        """Decode bounding boxes."""
        ...
    
    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = ...): # -> Tensor:
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        ...
    


class Segment(Detect):
    """YOLO Segment head for segmentation models."""
    def __init__(self, nc=..., nm=..., npr=..., ch=...) -> None:
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        ...
    
    def forward(self, x): # -> tuple[dict[str, Any] | Tensor | tuple[Tensor, dict[str, Any]] | Any | tuple[Tensor, Tensor | Any] | tuple[tuple[Tensor, Tensor | Any] | Tensor, Any], Tensor, Any] | tuple[Any, Any] | tuple[Any, tuple[Any | Tensor | dict[str, Any], Tensor, Any]]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        ...
    


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""
    def __init__(self, nc=..., ne=..., ch=...) -> None:
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        ...
    
    def forward(self, x): # -> tuple[dict[str, Any] | Tensor | tuple[Tensor, dict[str, Any]] | Any | tuple[Tensor, Tensor | Any] | tuple[tuple[Tensor, Tensor | Any] | Tensor, Any], Tensor] | tuple[Any, tuple[Any | Tensor | dict[str, Any], Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        ...
    
    def decode_bboxes(self, bboxes, anchors): # -> Tensor:
        """Decode rotated bounding boxes."""
        ...
    


class Pose(Detect):
    """YOLO Pose head for keypoints models."""
    def __init__(self, nc=..., kpt_shape=..., ch=...) -> None:
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        ...
    
    def forward(self, x): # -> tuple[dict[str, Any] | Tensor | tuple[Tensor, dict[str, Any]] | Any | tuple[Tensor, Tensor | Any] | tuple[tuple[Tensor, Tensor | Any] | Tensor, Any], Tensor] | tuple[Any, tuple[Any | Tensor | dict[str, Any], Tensor]]:
        """Perform forward pass through YOLO model and return predictions."""
        ...
    
    def kpts_decode(self, bs, kpts): # -> Tensor:
        """Decodes keypoints."""
        ...
    


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""
    export = ...
    def __init__(self, c1, c2, k=..., s=..., p=..., g=...) -> None:
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        ...
    
    def forward(self, x): # -> Any | tuple[Any, Any]:
        """Performs a forward pass of the YOLO model on input image data."""
        ...
    


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""
    def __init__(self, nc=..., embed=..., with_bn=..., ch=...) -> None:
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        ...
    
    def forward(self, x, text): # -> Tensor | tuple[Tensor, Any]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        ...
    
    def bias_init(self): # -> None:
        """Initialize Detect() biases, WARNING: requires stride availability."""
        ...
    


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = ...
    def __init__(self, nc=..., ch=..., hd=..., nq=..., ndp=..., nh=..., ndl=..., d_ffn=..., dropout=..., act=..., eval_idx=..., nd=..., label_noise_ratio=..., box_noise_scale=..., learnt_init_query=...) -> None:
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        ...
    
    def forward(self, x, batch=...): # -> tuple[Any, Any, Any, Any, dict[str, Any] | None] | Tensor | tuple[Tensor, tuple[Any, Any, Any, Any, dict[str, Any] | None]]:
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        ...
    


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """
    end2end = ...
    def __init__(self, nc=..., ch=...) -> None:
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        ...
    


