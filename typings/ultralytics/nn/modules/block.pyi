"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn

"""Block modules."""
__all__ = ("DFL", "HGBlock", "HGStem", "SPP", "SPPF", "C1", "C2", "C3", "C2f", "C2fAttn", "ImagePoolingAttn", "ContrastiveHead", "BNContrastiveHead", "C3x", "C3TR", "C3Ghost", "GhostBottleneck", "Bottleneck", "BottleneckCSP", "Proto", "RepC3", "ResNetLayer", "RepNCSPELAN4", "ELAN1", "ADown", "AConv", "SPPELAN", "CBFuse", "CBLinear", "C3k2", "C2fPSA", "C2PSA", "RepVGGDW", "CIB", "C2fCIB", "Attention", "PSA", "SCDown", "TorchVision")
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """
    def __init__(self, c1=...) -> None:
        """Initialize a convolutional layer with a given number of input channels."""
        ...
    
    def forward(self, x): # -> Any:
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        ...
    


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""
    def __init__(self, c1, c_=..., c2=...) -> None:
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        ...
    
    def forward(self, x): # -> Any:
        """Performs a forward pass through layers using an upsampled input image."""
        ...
    


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """
    def __init__(self, c1, cm, c2) -> None:
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass of a PPHGNetV2 backbone layer."""
        ...
    


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """
    def __init__(self, c1, cm, c2, k=..., n=..., lightconv=..., shortcut=..., act=...) -> None:
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass of a PPHGNetV2 backbone layer."""
        ...
    


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""
    def __init__(self, c1, c2, k=...) -> None:
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        ...
    


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
    def __init__(self, c1, c2, k=...) -> None:
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass through Ghost Convolution block."""
        ...
    


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""
    def __init__(self, c1, c2, n=...) -> None:
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        ...
    
    def forward(self, x): # -> Any:
        """Applies cross-convolutions to input in the C3 module."""
        ...
    


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=...) -> None:
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        ...
    


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=...) -> None:
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass through C2f layer."""
        ...
    
    def forward_split(self, x): # -> Any:
        """Forward pass using split() instead of chunk()."""
        ...
    


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=...) -> None:
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        ...
    


class C3x(C3):
    """C3 module with cross-convolutions."""
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=...) -> None:
        """Initialize C3TR instance and set default parameters."""
        ...
    


class RepC3(nn.Module):
    """Rep C3."""
    def __init__(self, c1, c2, n=..., e=...) -> None:
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass of RT-DETR neck layer."""
        ...
    


class C3TR(C3):
    """C3 module with TransformerBlock()."""
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=...) -> None:
        """Initialize C3Ghost module with GhostBottleneck()."""
        ...
    


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=...) -> None:
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        ...
    


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""
    def __init__(self, c1, c2, k=..., s=...) -> None:
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        ...
    
    def forward(self, x): # -> Any:
        """Applies skip connection and concatenation to input tensor."""
        ...
    


class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=..., g=..., k=..., e=...) -> None:
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        ...
    
    def forward(self, x): # -> Any:
        """Applies the YOLO FPN to input data."""
        ...
    


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=...) -> None:
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        ...
    
    def forward(self, x): # -> Any:
        """Applies a CSP bottleneck with 3 convolutions."""
        ...
    


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""
    def __init__(self, c1, c2, s=..., e=...) -> None:
        """Initialize convolution with given parameters."""
        ...
    
    def forward(self, x): # -> Tensor:
        """Forward pass through the ResNet block."""
        ...
    


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""
    def __init__(self, c1, c2, s=..., is_first=..., n=..., e=...) -> None:
        """Initializes the ResNetLayer given arguments."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass through the ResNet layer."""
        ...
    


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""
    def __init__(self, c1, c2, nh=..., ec=..., gc=..., scale=...) -> None:
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        ...
    
    def forward(self, x, guide):
        """Forward process."""
        ...
    


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""
    def __init__(self, c1, c2, n=..., ec=..., nh=..., gc=..., shortcut=..., g=..., e=...) -> None:
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        ...
    
    def forward(self, x, guide): # -> Any:
        """Forward pass through C2f layer."""
        ...
    
    def forward_split(self, x, guide): # -> Any:
        """Forward pass using split() instead of chunk()."""
        ...
    


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""
    def __init__(self, ec=..., ch=..., ct=..., nh=..., k=..., scale=...) -> None:
        """Initializes ImagePoolingAttn with specified arguments."""
        ...
    
    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        ...
    


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""
    def __init__(self) -> None:
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        ...
    
    def forward(self, x, w): # -> Tensor:
        """Forward function of contrastive learning."""
        ...
    


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """
    def __init__(self, embed_dims: int) -> None:
        """Initialize ContrastiveHead with region-text similarity parameters."""
        ...
    
    def forward(self, x, w): # -> Tensor:
        """Forward function of contrastive learning."""
        ...
    


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""
    def __init__(self, c1, c2, shortcut=..., g=..., k=..., e=...) -> None:
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        ...
    


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=...) -> None:
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        ...
    


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""
    def __init__(self, c1, c2, c3, c4, n=...) -> None:
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass through RepNCSPELAN4 layer."""
        ...
    
    def forward_split(self, x): # -> Any:
        """Forward pass using split() instead of chunk()."""
        ...
    


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""
    def __init__(self, c1, c2, c3, c4) -> None:
        """Initializes ELAN1 layer with specified channel sizes."""
        ...
    


class AConv(nn.Module):
    """AConv."""
    def __init__(self, c1, c2) -> None:
        """Initializes AConv module with convolution layers."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass through AConv layer."""
        ...
    


class ADown(nn.Module):
    """ADown."""
    def __init__(self, c1, c2) -> None:
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        ...
    
    def forward(self, x): # -> Tensor:
        """Forward pass through ADown layer."""
        ...
    


class SPPELAN(nn.Module):
    """SPP-ELAN."""
    def __init__(self, c1, c2, c3, k=...) -> None:
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass through SPPELAN layer."""
        ...
    


class CBLinear(nn.Module):
    """CBLinear."""
    def __init__(self, c1, c2s, k=..., s=..., p=..., g=...) -> None:
        """Initializes the CBLinear module, passing inputs unchanged."""
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass through CBLinear layer."""
        ...
    


class CBFuse(nn.Module):
    """CBFuse."""
    def __init__(self, idx) -> None:
        """Initializes CBFuse module with layer index for selective feature fusion."""
        ...
    
    def forward(self, xs): # -> Tensor:
        """Forward pass through CBFuse layer."""
        ...
    


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=...) -> None:
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass through C2f layer."""
        ...
    


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=..., c3k=..., e=..., g=..., shortcut=...) -> None:
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        ...
    


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
    def __init__(self, c1, c2, n=..., shortcut=..., g=..., e=..., k=...) -> None:
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        ...
    


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""
    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        ...
    
    def forward(self, x): # -> Any:
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        ...
    
    def forward_fuse(self, x): # -> Any:
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        ...
    
    @torch.no_grad()
    def fuse(self): # -> None:
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        ...
    


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """
    def __init__(self, c1, c2, shortcut=..., e=..., lk=...) -> None:
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        ...
    
    def forward(self, x): # -> Any:
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        ...
    


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """
    def __init__(self, c1, c2, n=..., shortcut=..., lk=..., g=..., e=...) -> None:
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        ...
    


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """
    def __init__(self, dim, num_heads=..., attn_ratio=...) -> None:
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        ...
    
    def forward(self, x): # -> Any:
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        ...
    


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """
    def __init__(self, c, attn_ratio=..., num_heads=..., shortcut=...) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        ...
    
    def forward(self, x): # -> Any:
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        ...
    


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """
    def __init__(self, c1, c2, e=...) -> None:
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        ...
    
    def forward(self, x): # -> Any:
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        ...
    


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """
    def __init__(self, c1, c2, n=..., e=...) -> None:
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        ...
    
    def forward(self, x): # -> Any:
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        ...
    


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """
    def __init__(self, c1, c2, n=..., e=...) -> None:
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        ...
    


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """
    def __init__(self, c1, c2, k, s) -> None:
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        ...
    
    def forward(self, x): # -> Any:
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        ...
    


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """
    def __init__(self, model, weights=..., unwrap=..., truncate=..., split=...) -> None:
        """Load the model and weights from torchvision."""
        ...
    
    def forward(self, x): # -> list[Any] | Any:
        """Forward pass through the model."""
        ...
    


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """
    def __init__(self, dim, num_heads, area=...) -> None:
        """
        Initializes an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided, default is 1.
        """
        ...
    
    def forward(self, x): # -> Any:
        """Processes the input tensor 'x' through the area-attention."""
        ...
    


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """
    def __init__(self, dim, num_heads, mlp_ratio=..., area=...) -> None:
        """
        Initializes an Area-attention block module for efficient feature extraction in YOLO models.

        This module implements an area-attention mechanism combined with a feed-forward network for processing feature
        maps. It uses a novel area-based attention approach that is more efficient than traditional self-attention
        while maintaining effectiveness.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        ...
    
    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        ...
    


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """
    def __init__(self, c1, c2, n=..., a2=..., area=..., residual=..., mlp_ratio=..., e=..., g=..., shortcut=...) -> None:
        """
        Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        ...
    
    def forward(self, x): # -> Any:
        """Forward pass through R-ELAN layer."""
        ...
    


