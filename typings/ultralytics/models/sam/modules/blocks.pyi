"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Type, Union
from torch import Tensor, nn
from .transformer import Attention, TwoWayAttentionBlock, TwoWayTransformer

class DropPath(nn.Module):
    """
    Implements stochastic depth regularization for neural networks during training.

    Attributes:
        drop_prob (float): Probability of dropping a path during training.
        scale_by_keep (bool): Whether to scale the output by the keep probability.

    Methods:
        forward: Applies stochastic depth to input tensor during training, with optional scaling.

    Examples:
        >>> drop_path = DropPath(drop_prob=0.2, scale_by_keep=True)
        >>> x = torch.randn(32, 64, 224, 224)
        >>> output = drop_path(x)
    """
    def __init__(self, drop_prob=..., scale_by_keep=...) -> None:
        """Initialize DropPath module for stochastic depth regularization during training."""
        ...
    
    def forward(self, x):
        """Applies stochastic depth to input tensor during training, with optional scaling."""
        ...
    


class MaskDownSampler(nn.Module):
    """
    A mask downsampling and embedding module for efficient processing of input masks.

    This class implements a mask downsampler that progressively reduces the spatial dimensions of input masks
    while expanding their channel dimensions using convolutional layers, layer normalization, and activation
    functions.

    Attributes:
        encoder (nn.Sequential): A sequential container of convolutional layers, layer normalization, and
            activation functions for downsampling and embedding masks.

    Methods:
        forward: Downsamples and encodes input mask to embed_dim channels.

    Examples:
        >>> mask_downsampler = MaskDownSampler(embed_dim=256, kernel_size=4, stride=4, padding=0, total_stride=16)
        >>> input_mask = torch.randn(1, 1, 256, 256)
        >>> output = mask_downsampler(input_mask)
        >>> print(output.shape)
        torch.Size([1, 256, 16, 16])
    """
    def __init__(self, embed_dim=..., kernel_size=..., stride=..., padding=..., total_stride=..., activation=...) -> None:
        """Initializes a mask downsampler module for progressive downsampling and channel expansion."""
        ...
    
    def forward(self, x): # -> Any:
        """Downsamples and encodes input mask to embed_dim channels using convolutional layers and LayerNorm2d."""
        ...
    


class CXBlock(nn.Module):
    """
    ConvNeXt Block for efficient feature extraction in convolutional neural networks.

    This block implements a modified version of the ConvNeXt architecture, offering improved performance and
    flexibility in feature extraction.

    Attributes:
        dwconv (nn.Conv2d): Depthwise or standard 2D convolution layer.
        norm (LayerNorm2d): Layer normalization applied to channels.
        pwconv1 (nn.Linear): First pointwise convolution implemented as a linear layer.
        act (nn.GELU): GELU activation function.
        pwconv2 (nn.Linear): Second pointwise convolution implemented as a linear layer.
        gamma (nn.Parameter | None): Learnable scale parameter for layer scaling.
        drop_path (nn.Module): DropPath layer for stochastic depth regularization.

    Methods:
        forward: Processes the input tensor through the ConvNeXt block.

    Examples:
        >>> import torch
        >>> x = torch.randn(1, 64, 56, 56)
        >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """
    def __init__(self, dim, kernel_size=..., padding=..., drop_path=..., layer_scale_init_value=..., use_dwconv=...) -> None:
        """
        Initialize a ConvNeXt Block for efficient feature extraction in convolutional neural networks.

        This block implements a modified version of the ConvNeXt architecture, offering improved performance and
        flexibility in feature extraction.

        Args:
            dim (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding size for the convolution.
            drop_path (float): Stochastic depth rate.
            layer_scale_init_value (float): Initial value for Layer Scale.
            use_dwconv (bool): Whether to use depthwise convolution.

        Examples:
            >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
            >>> x = torch.randn(1, 64, 32, 32)
            >>> output = block(x)
            >>> print(output.shape)
            torch.Size([1, 64, 32, 32])
        """
        ...
    
    def forward(self, x):
        """Applies ConvNeXt block operations to input tensor, including convolutions and residual connection."""
        ...
    


class Fuser(nn.Module):
    """
    A module for fusing features through multiple layers of a neural network.

    This class applies a series of identical layers to an input tensor, optionally projecting the input first.

    Attributes:
        proj (nn.Module): An optional input projection layer. Identity if no projection is needed.
        layers (nn.ModuleList): A list of identical layers to be applied sequentially.

    Methods:
        forward: Applies the fuser to an input tensor.

    Examples:
        >>> layer = CXBlock(dim=256)
        >>> fuser = Fuser(layer, num_layers=3, dim=256, input_projection=True)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = fuser(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """
    def __init__(self, layer, num_layers, dim=..., input_projection=...) -> None:
        """
        Initializes the Fuser module for feature fusion through multiple layers.

        This module creates a sequence of identical layers and optionally applies an input projection.

        Args:
            layer (nn.Module): The layer to be replicated in the fuser.
            num_layers (int): The number of times to replicate the layer.
            dim (int | None): The dimension for input projection, if used.
            input_projection (bool): Whether to use input projection.

        Examples:
            >>> layer = nn.Linear(64, 64)
            >>> fuser = Fuser(layer, num_layers=3, dim=64, input_projection=True)
            >>> input_tensor = torch.randn(1, 64)
            >>> output = fuser(input_tensor)
        """
        ...
    
    def forward(self, x): # -> Any:
        """Applies a series of layers to the input tensor, optionally projecting it first."""
        ...
    


class SAM2TwoWayAttentionBlock(TwoWayAttentionBlock):
    """
    A two-way attention block for performing self-attention and cross-attention in both directions.

    This block extends the TwoWayAttentionBlock and consists of four main components: self-attention on
    sparse inputs, cross-attention from sparse to dense inputs, an MLP block on sparse inputs, and
    cross-attention from dense to sparse inputs.

    Attributes:
        self_attn (Attention): Self-attention layer for queries.
        norm1 (nn.LayerNorm): Layer normalization after the first attention block.
        cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization after the second attention block.
        mlp (MLP): MLP block for transforming query embeddings.
        norm3 (nn.LayerNorm): Layer normalization after the MLP block.
        norm4 (nn.LayerNorm): Layer normalization after the third attention block.
        cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
        skip_first_layer_pe (bool): Flag to skip positional encoding in the first layer.

    Methods:
        forward: Processes input through the attention blocks and MLP.

    Examples:
        >>> block = SAM2TwoWayAttentionBlock(embedding_dim=256, num_heads=8)
        >>> sparse_input = torch.randn(1, 100, 256)
        >>> dense_input = torch.randn(1, 256, 16, 16)
        >>> sparse_output, dense_output = block(sparse_input, dense_input)
    """
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = ..., activation: Type[nn.Module] = ..., attention_downsample_rate: int = ..., skip_first_layer_pe: bool = ...) -> None:
        """
        Initializes a SAM2TwoWayAttentionBlock for performing self-attention and cross-attention in two directions.

        This block extends the TwoWayAttentionBlock and consists of four main components: self-attention on sparse
        inputs, cross-attention from sparse to dense inputs, an MLP block on sparse inputs, and cross-attention
        from dense to sparse inputs.

        Args:
            embedding_dim (int): The channel dimension of the embeddings.
            num_heads (int): The number of heads in the attention layers.
            mlp_dim (int): The hidden dimension of the MLP block.
            activation (Type[nn.Module]): The activation function of the MLP block.
            attention_downsample_rate (int): The downsample rate for attention computations.
            skip_first_layer_pe (bool): Whether to skip the positional encoding in the first layer.

        Examples:
            >>> block = SAM2TwoWayAttentionBlock(embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> sparse_inputs = torch.randn(1, 100, 256)
            >>> dense_inputs = torch.randn(1, 256, 32, 32)
            >>> sparse_outputs, dense_outputs = block(sparse_inputs, dense_inputs)
        """
        ...
    


class SAM2TwoWayTransformer(TwoWayTransformer):
    """
    A Two-Way Transformer module for simultaneous attention to image and query points.

    This class extends the TwoWayTransformer, implementing a specialized transformer decoder that attends to an
    input image using queries with supplied positional embeddings. It is particularly useful for tasks like
    object detection, image segmentation, and point cloud processing.

    Attributes:
        depth (int): Number of layers in the transformer.
        embedding_dim (int): Channel dimension for input embeddings.
        num_heads (int): Number of heads for multihead attention.
        mlp_dim (int): Internal channel dimension for the MLP block.
        layers (nn.ModuleList): List of SAM2TwoWayAttentionBlock layers comprising the transformer.
        final_attn_token_to_image (Attention): Final attention layer from queries to image.
        norm_final_attn (nn.LayerNorm): Layer normalization applied to final queries.

    Methods:
        forward: Processes input image embeddings and query embeddings through the transformer.

    Examples:
        >>> transformer = SAM2TwoWayTransformer(depth=5, embedding_dim=256, num_heads=8, mlp_dim=2048)
        >>> image_embedding = torch.randn(1, 256, 64, 64)
        >>> query_embedding = torch.randn(1, 100, 256)
        >>> output = transformer(image_embedding, query_embedding)
        >>> print(output[0].shape, output[1].shape)
        torch.Size([1, 100, 256]) torch.Size([1, 256, 64, 64])
    """
    def __init__(self, depth: int, embedding_dim: int, num_heads: int, mlp_dim: int, activation: Type[nn.Module] = ..., attention_downsample_rate: int = ...) -> None:
        """
        Initializes a SAM2TwoWayTransformer instance.

        This transformer decoder attends to an input image using queries with supplied positional embeddings.
        It is designed for tasks like object detection, image segmentation, and point cloud processing.

        Args:
            depth (int): Number of layers in the transformer.
            embedding_dim (int): Channel dimension for the input embeddings.
            num_heads (int): Number of heads for multihead attention. Must divide embedding_dim.
            mlp_dim (int): Channel dimension internal to the MLP block.
            activation (Type[nn.Module]): Activation function to use in the MLP block.
            attention_downsample_rate (int): Downsampling rate for attention computations.

        Examples:
            >>> transformer = SAM2TwoWayTransformer(depth=5, embedding_dim=256, num_heads=8, mlp_dim=2048)
            >>> transformer
            SAM2TwoWayTransformer(
              (layers): ModuleList(
                (0-4): 5 x SAM2TwoWayAttentionBlock(...)
              )
              (final_attn_token_to_image): Attention(...)
              (norm_final_attn): LayerNorm(...)
            )
        """
        ...
    


class RoPEAttention(Attention):
    """
    Implements rotary position encoding for attention mechanisms in transformer architectures.

    This class extends the base Attention class by incorporating Rotary Position Encoding (RoPE) to enhance
    the positional awareness of the attention mechanism.

    Attributes:
        compute_cis (Callable): Function to compute axial complex numbers for rotary encoding.
        freqs_cis (Tensor): Precomputed frequency tensor for rotary encoding.
        rope_k_repeat (bool): Flag to repeat query RoPE to match key length for cross-attention to memories.

    Methods:
        forward: Applies rotary position encoding and computes attention between query, key, and value tensors.

    Examples:
        >>> rope_attn = RoPEAttention(embedding_dim=256, num_heads=8, rope_theta=10000.0, feat_sizes=(32, 32))
        >>> q = torch.randn(1, 1024, 256)
        >>> k = torch.randn(1, 1024, 256)
        >>> v = torch.randn(1, 1024, 256)
        >>> output = rope_attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 1024, 256])
    """
    def __init__(self, *args, rope_theta=..., rope_k_repeat=..., feat_sizes=..., **kwargs) -> None:
        """Initializes RoPEAttention with rotary position encoding for enhanced positional awareness."""
        ...
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = ...) -> Tensor:
        """Applies rotary position encoding and computes attention between query, key, and value tensors."""
        ...
    


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = ...) -> torch.Tensor:
    """Applies pooling and optional normalization to a tensor, handling spatial dimension permutations."""
    ...

class MultiScaleAttention(nn.Module):
    """
    Implements multiscale self-attention with optional query pooling for efficient feature extraction.

    This class provides a flexible implementation of multiscale attention, allowing for optional
    downsampling of query features through pooling. It's designed to enhance the model's ability to
    capture multiscale information in visual tasks.

    Attributes:
        dim (int): Input dimension of the feature map.
        dim_out (int): Output dimension of the attention module.
        num_heads (int): Number of attention heads.
        scale (float): Scaling factor for dot-product attention.
        q_pool (nn.Module | None): Optional pooling module for query features.
        qkv (nn.Linear): Linear projection for query, key, and value.
        proj (nn.Linear): Output projection.

    Methods:
        forward: Applies multiscale attention to the input tensor.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> x = torch.randn(1, 64, 64, 256)
        >>> msa = MultiScaleAttention(dim=256, dim_out=256, num_heads=8)
        >>> output = msa(x)
        >>> print(output.shape)
        torch.Size([1, 64, 64, 256])
    """
    def __init__(self, dim: int, dim_out: int, num_heads: int, q_pool: nn.Module = ...) -> None:
        """Initializes multiscale attention with optional query pooling for efficient feature extraction."""
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies multiscale attention with optional query pooling to extract multiscale features."""
        ...
    


class MultiScaleBlock(nn.Module):
    """
    A multiscale attention block with window partitioning and query pooling for efficient vision transformers.

    This class implements a multiscale attention mechanism with optional window partitioning and downsampling,
    designed for use in vision transformer architectures.

    Attributes:
        dim (int): Input dimension of the block.
        dim_out (int): Output dimension of the block.
        norm1 (nn.Module): First normalization layer.
        window_size (int): Size of the window for partitioning.
        pool (nn.Module | None): Pooling layer for query downsampling.
        q_stride (Tuple[int, int] | None): Stride for query pooling.
        attn (MultiScaleAttention): Multi-scale attention module.
        drop_path (nn.Module): Drop path layer for regularization.
        norm2 (nn.Module): Second normalization layer.
        mlp (MLP): Multi-layer perceptron module.
        proj (nn.Linear | None): Projection layer for dimension mismatch.

    Methods:
        forward: Processes input tensor through the multiscale block.

    Examples:
        >>> block = MultiScaleBlock(dim=256, dim_out=512, num_heads=8, window_size=7)
        >>> x = torch.randn(1, 56, 56, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 28, 28, 512])
    """
    def __init__(self, dim: int, dim_out: int, num_heads: int, mlp_ratio: float = ..., drop_path: float = ..., norm_layer: Union[nn.Module, str] = ..., q_stride: Tuple[int, int] = ..., act_layer: nn.Module = ..., window_size: int = ...) -> None:
        """Initializes a multiscale attention block with window partitioning and optional query pooling."""
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through multiscale attention and MLP, with optional windowing and downsampling."""
        ...
    


class PositionEmbeddingSine(nn.Module):
    """
    A module for generating sinusoidal positional embeddings for 2D inputs like images.

    This class implements sinusoidal position encoding for 2D spatial positions, which can be used in
    transformer-based models for computer vision tasks.

    Attributes:
        num_pos_feats (int): Number of positional features (half of the embedding dimension).
        temperature (int): Temperature parameter for the sinusoidal functions.
        normalize (bool): Whether to normalize the positional embeddings.
        scale (float): Scaling factor for the embeddings when normalize is True.
        cache (Dict): Cache for storing precomputed embeddings.

    Methods:
        _encode_xy: Encodes 2D positions using sine and cosine functions.
        encode_boxes: Encodes box coordinates and dimensions into positional embeddings.
        encode_points: Encodes 2D point coordinates with sinusoidal positional embeddings.
        forward: Generates sinusoidal position embeddings for 2D inputs.

    Examples:
        >>> pos_emb = PositionEmbeddingSine(num_pos_feats=128)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> embeddings = pos_emb(x)
        >>> print(embeddings.shape)
        torch.Size([1, 256, 224, 224])
    """
    def __init__(self, num_pos_feats, temperature: int = ..., normalize: bool = ..., scale: Optional[float] = ...) -> None:
        """Initializes sinusoidal position embeddings for 2D image inputs."""
        ...
    
    @torch.no_grad()
    def encode_boxes(self, x, y, w, h): # -> Tensor:
        """Encodes box coordinates and dimensions into positional embeddings for detection."""
        ...
    
    encode = ...
    @torch.no_grad()
    def encode_points(self, x, y, labels): # -> Tensor:
        """Encodes 2D points with sinusoidal embeddings and appends labels."""
        ...
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor): # -> Tensor:
        """Generates sinusoidal position embeddings for 2D inputs like images."""
        ...
    


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.

    This class generates positional embeddings for input coordinates using random spatial frequencies. It is
    particularly useful for transformer-based models that require position information.

    Attributes:
        positional_encoding_gaussian_matrix (torch.Tensor): A buffer containing random values for encoding.

    Methods:
        _pe_encoding: Positionally encodes points that are normalized to [0,1].
        forward: Generates positional encoding for a grid of the specified size.
        forward_with_coords: Positionally encodes points that are not normalized to [0,1].

    Examples:
        >>> pe = PositionEmbeddingRandom(num_pos_feats=64)
        >>> size = (32, 32)
        >>> encoding = pe(size)
        >>> print(encoding.shape)
        torch.Size([128, 32, 32])
    """
    def __init__(self, num_pos_feats: int = ..., scale: Optional[float] = ...) -> None:
        """Initializes random spatial frequency position embedding for transformers."""
        ...
    
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generates positional encoding for a grid using random spatial frequencies."""
        ...
    
    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Positionally encodes input coordinates, normalizing them to [0,1] based on the given image size."""
        ...
    


class Block(nn.Module):
    """
    Transformer block with support for window attention and residual propagation.

    This class implements a transformer block that can use either global or windowed self-attention,
    followed by a feed-forward network. It supports relative positional embeddings and is designed
    for use in vision transformer architectures.

    Attributes:
        norm1 (nn.Module): First normalization layer.
        attn (REAttention): Self-attention layer with optional relative positional encoding.
        norm2 (nn.Module): Second normalization layer.
        mlp (MLPBlock): Multi-layer perceptron block.
        window_size (int): Size of attention window. If 0, global attention is used.

    Methods:
        forward: Processes input through the transformer block.

    Examples:
        >>> import torch
        >>> block = Block(dim=256, num_heads=8, window_size=7)
        >>> x = torch.randn(1, 56, 56, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 56, 56, 256])
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = ..., qkv_bias: bool = ..., norm_layer: Type[nn.Module] = ..., act_layer: Type[nn.Module] = ..., use_rel_pos: bool = ..., rel_pos_zero_init: bool = ..., window_size: int = ..., input_size: Optional[Tuple[int, int]] = ...) -> None:
        """
        Initializes a transformer block with optional window attention and relative positional embeddings.

        This constructor sets up a transformer block that can use either global or windowed self-attention,
        followed by a feed-forward network. It supports relative positional embeddings and is designed
        for use in vision transformer architectures.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in the self-attention layer.
            mlp_ratio (float): Ratio of mlp hidden dimension to embedding dimension.
            qkv_bias (bool): If True, adds a learnable bias to query, key, value projections.
            norm_layer (Type[nn.Module]): Type of normalization layer to use.
            act_layer (Type[nn.Module]): Type of activation function to use in the MLP block.
            use_rel_pos (bool): If True, uses relative positional embeddings in attention.
            rel_pos_zero_init (bool): If True, initializes relative positional parameters to zero.
            window_size (int): Size of attention window. If 0, uses global attention.
            input_size (Optional[Tuple[int, int]]): Input resolution for calculating relative positional parameter size.

        Examples:
            >>> block = Block(dim=256, num_heads=8, window_size=7)
            >>> x = torch.randn(1, 56, 56, 256)
            >>> output = block(x)
            >>> print(output.shape)
            torch.Size([1, 56, 56, 256])
        """
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through transformer block with optional windowed self-attention and residual connection."""
        ...
    


class REAttention(nn.Module):
    """
    Rotary Embedding Attention module for efficient self-attention in transformer architectures.

    This class implements a multi-head attention mechanism with rotary positional embeddings, designed
    for use in vision transformer models. It supports optional query pooling and window partitioning
    for efficient processing of large inputs.

    Attributes:
        compute_cis (Callable): Function to compute axial complex numbers for rotary encoding.
        freqs_cis (Tensor): Precomputed frequency tensor for rotary encoding.
        rope_k_repeat (bool): Flag to repeat query RoPE to match key length for cross-attention to memories.
        q_proj (nn.Linear): Linear projection for query.
        k_proj (nn.Linear): Linear projection for key.
        v_proj (nn.Linear): Linear projection for value.
        out_proj (nn.Linear): Output projection.
        num_heads (int): Number of attention heads.
        internal_dim (int): Internal dimension for attention computation.

    Methods:
        forward: Applies rotary position encoding and computes attention between query, key, and value tensors.

    Examples:
        >>> rope_attn = REAttention(embedding_dim=256, num_heads=8, rope_theta=10000.0, feat_sizes=(32, 32))
        >>> q = torch.randn(1, 1024, 256)
        >>> k = torch.randn(1, 1024, 256)
        >>> v = torch.randn(1, 1024, 256)
        >>> output = rope_attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 1024, 256])
    """
    def __init__(self, dim: int, num_heads: int = ..., qkv_bias: bool = ..., use_rel_pos: bool = ..., rel_pos_zero_init: bool = ..., input_size: Optional[Tuple[int, int]] = ...) -> None:
        """
        Initializes a Relative Position Attention module for transformer-based architectures.

        This module implements multi-head attention with optional relative positional encodings, designed
        specifically for vision tasks in transformer models.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads. Default is 8.
            qkv_bias (bool): If True, adds a learnable bias to query, key, value projections. Default is True.
            use_rel_pos (bool): If True, uses relative positional encodings. Default is False.
            rel_pos_zero_init (bool): If True, initializes relative positional parameters to zero. Default is True.
            input_size (Tuple[int, int] | None): Input resolution for calculating relative positional parameter size.
                Required if use_rel_pos is True. Default is None.

        Examples:
            >>> attention = REAttention(dim=256, num_heads=8, input_size=(32, 32))
            >>> x = torch.randn(1, 32, 32, 256)
            >>> output = attention(x)
            >>> print(output.shape)
            torch.Size([1, 32, 32, 256])
        """
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies multi-head attention with optional relative positional encoding to input tensor."""
        ...
    


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding module for vision transformer architectures.

    This module converts an input image into a sequence of patch embeddings using a convolutional layer.
    It is commonly used as the first layer in vision transformer architectures to transform image data
    into a suitable format for subsequent transformer blocks.

    Attributes:
        proj (nn.Conv2d): Convolutional layer for projecting image patches to embeddings.

    Methods:
        forward: Applies patch embedding to the input tensor.

    Examples:
        >>> patch_embed = PatchEmbed(kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = patch_embed(x)
        >>> print(output.shape)
        torch.Size([1, 768, 14, 14])
    """
    def __init__(self, kernel_size: Tuple[int, int] = ..., stride: Tuple[int, int] = ..., padding: Tuple[int, int] = ..., in_chans: int = ..., embed_dim: int = ...) -> None:
        """
        Initializes the PatchEmbed module for converting image patches to embeddings.

        This module is typically used as the first layer in vision transformer architectures to transform
        image data into a suitable format for subsequent transformer blocks.

        Args:
            kernel_size (Tuple[int, int]): Size of the convolutional kernel for patch extraction.
            stride (Tuple[int, int]): Stride of the convolutional operation.
            padding (Tuple[int, int]): Padding applied to the input before convolution.
            in_chans (int): Number of input image channels.
            embed_dim (int): Dimensionality of the output patch embeddings.

        Examples:
            >>> patch_embed = PatchEmbed(kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> output = patch_embed(x)
            >>> print(output.shape)
            torch.Size([1, 768, 14, 14])
        """
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes patch embedding by applying convolution and transposing resulting tensor."""
        ...
    


