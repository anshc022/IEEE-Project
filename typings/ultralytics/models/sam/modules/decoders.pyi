"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Optional, Tuple, Type
from torch import nn

class MaskDecoder(nn.Module):
    """
    Decoder module for generating masks and their associated quality scores using a transformer architecture.

    This class predicts masks given image and prompt embeddings, utilizing a transformer to process the inputs and
    generate mask predictions along with their quality scores.

    Attributes:
        transformer_dim (int): Channel dimension for the transformer module.
        transformer (nn.Module): Transformer module used for mask prediction.
        num_multimask_outputs (int): Number of masks to predict for disambiguating masks.
        iou_token (nn.Embedding): Embedding for the IoU token.
        num_mask_tokens (int): Number of mask tokens.
        mask_tokens (nn.Embedding): Embedding for the mask tokens.
        output_upscaling (nn.Sequential): Neural network sequence for upscaling the output.
        output_hypernetworks_mlps (nn.ModuleList): Hypernetwork MLPs for generating masks.
        iou_prediction_head (nn.Module): MLP for predicting mask quality.

    Methods:
        forward: Predicts masks given image and prompt embeddings.
        predict_masks: Internal method for mask prediction.

    Examples:
        >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer_module)
        >>> masks, iou_pred = decoder(
        ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=True
        ... )
        >>> print(f"Predicted masks shape: {masks.shape}, IoU predictions shape: {iou_pred.shape}")
    """
    def __init__(self, transformer_dim: int, transformer: nn.Module, num_multimask_outputs: int = ..., activation: Type[nn.Module] = ..., iou_head_depth: int = ..., iou_head_hidden_dim: int = ...) -> None:
        """
        Initializes the MaskDecoder module for generating masks and their quality scores.

        Args:
            transformer_dim (int): Channel dimension for the transformer module.
            transformer (nn.Module): Transformer module used for mask prediction.
            num_multimask_outputs (int): Number of masks to predict for disambiguating masks.
            activation (Type[nn.Module]): Type of activation to use when upscaling masks.
            iou_head_depth (int): Depth of the MLP used to predict mask quality.
            iou_head_hidden_dim (int): Hidden dimension of the MLP used to predict mask quality.

        Examples:
            >>> transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8), num_layers=6)
            >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer)
            >>> print(decoder)
        """
        ...
    
    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor, multimask_output: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks given image and prompt embeddings.

        Args:
            image_embeddings (torch.Tensor): Embeddings from the image encoder.
            image_pe (torch.Tensor): Positional encoding with the shape of image_embeddings.
            sparse_prompt_embeddings (torch.Tensor): Embeddings of the points and boxes.
            dense_prompt_embeddings (torch.Tensor): Embeddings of the mask inputs.
            multimask_output (bool): Whether to return multiple masks or a single mask.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): A tuple containing:
                - masks (torch.Tensor): Batched predicted masks.
                - iou_pred (torch.Tensor): Batched predictions of mask quality.

        Examples:
            >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer_module)
            >>> image_emb = torch.rand(1, 256, 64, 64)
            >>> image_pe = torch.rand(1, 256, 64, 64)
            >>> sparse_emb = torch.rand(1, 2, 256)
            >>> dense_emb = torch.rand(1, 256, 64, 64)
            >>> masks, iou_pred = decoder(image_emb, image_pe, sparse_emb, dense_emb, multimask_output=True)
            >>> print(f"Masks shape: {masks.shape}, IoU predictions shape: {iou_pred.shape}")
        """
        ...
    
    def predict_masks(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks and quality scores using image and prompt embeddings via transformer architecture."""
        ...
    


class SAM2MaskDecoder(nn.Module):
    """
    Transformer-based decoder for predicting instance segmentation masks from image and prompt embeddings.

    This class extends the functionality of the MaskDecoder, incorporating additional features such as
    high-resolution feature processing, dynamic multimask output, and object score prediction.

    Attributes:
        transformer_dim (int): Channel dimension of the transformer.
        transformer (nn.Module): Transformer used to predict masks.
        num_multimask_outputs (int): Number of masks to predict when disambiguating masks.
        iou_token (nn.Embedding): Embedding for IOU token.
        num_mask_tokens (int): Total number of mask tokens.
        mask_tokens (nn.Embedding): Embedding for mask tokens.
        pred_obj_scores (bool): Whether to predict object scores.
        obj_score_token (nn.Embedding): Embedding for object score token.
        use_multimask_token_for_obj_ptr (bool): Whether to use multimask token for object pointer.
        output_upscaling (nn.Sequential): Upscaling layers for output.
        use_high_res_features (bool): Whether to use high-resolution features.
        conv_s0 (nn.Conv2d): Convolutional layer for high-resolution features (s0).
        conv_s1 (nn.Conv2d): Convolutional layer for high-resolution features (s1).
        output_hypernetworks_mlps (nn.ModuleList): List of MLPs for output hypernetworks.
        iou_prediction_head (MLP): MLP for IOU prediction.
        pred_obj_score_head (nn.Linear | MLP): Linear layer or MLP for object score prediction.
        dynamic_multimask_via_stability (bool): Whether to use dynamic multimask via stability.
        dynamic_multimask_stability_delta (float): Delta value for dynamic multimask stability.
        dynamic_multimask_stability_thresh (float): Threshold for dynamic multimask stability.

    Methods:
        forward: Predicts masks given image and prompt embeddings.
        predict_masks: Predicts instance segmentation masks from image and prompt embeddings.
        _get_stability_scores: Computes mask stability scores based on IoU between thresholds.
        _dynamic_multimask_via_stability: Dynamically selects the most stable mask output.

    Examples:
        >>> image_embeddings = torch.rand(1, 256, 64, 64)
        >>> image_pe = torch.rand(1, 256, 64, 64)
        >>> sparse_prompt_embeddings = torch.rand(1, 2, 256)
        >>> dense_prompt_embeddings = torch.rand(1, 256, 64, 64)
        >>> decoder = SAM2MaskDecoder(256, transformer)
        >>> masks, iou_pred, sam_tokens_out, obj_score_logits = decoder.forward(
        ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, True, False
        ... )
    """
    def __init__(self, transformer_dim: int, transformer: nn.Module, num_multimask_outputs: int = ..., activation: Type[nn.Module] = ..., iou_head_depth: int = ..., iou_head_hidden_dim: int = ..., use_high_res_features: bool = ..., iou_prediction_use_sigmoid=..., dynamic_multimask_via_stability=..., dynamic_multimask_stability_delta=..., dynamic_multimask_stability_thresh=..., pred_obj_scores: bool = ..., pred_obj_scores_mlp: bool = ..., use_multimask_token_for_obj_ptr: bool = ...) -> None:
        """
        Initializes the SAM2MaskDecoder module for predicting instance segmentation masks.

        This decoder extends the functionality of MaskDecoder, incorporating additional features such as
        high-resolution feature processing, dynamic multimask output, and object score prediction.

        Args:
            transformer_dim (int): Channel dimension of the transformer.
            transformer (nn.Module): Transformer used to predict masks.
            num_multimask_outputs (int): Number of masks to predict when disambiguating masks.
            activation (Type[nn.Module]): Type of activation to use when upscaling masks.
            iou_head_depth (int): Depth of the MLP used to predict mask quality.
            iou_head_hidden_dim (int): Hidden dimension of the MLP used to predict mask quality.
            use_high_res_features (bool): Whether to use high-resolution features.
            iou_prediction_use_sigmoid (bool): Whether to use sigmoid for IOU prediction.
            dynamic_multimask_via_stability (bool): Whether to use dynamic multimask via stability.
            dynamic_multimask_stability_delta (float): Delta value for dynamic multimask stability.
            dynamic_multimask_stability_thresh (float): Threshold for dynamic multimask stability.
            pred_obj_scores (bool): Whether to predict object scores.
            pred_obj_scores_mlp (bool): Whether to use MLP for object score prediction.
            use_multimask_token_for_obj_ptr (bool): Whether to use multimask token for object pointer.

        Examples:
            >>> transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8), num_layers=6)
            >>> decoder = SAM2MaskDecoder(transformer_dim=256, transformer=transformer)
            >>> print(decoder)
        """
        ...
    
    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor, multimask_output: bool, repeat_image: bool, high_res_features: Optional[List[torch.Tensor]] = ...) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks given image and prompt embeddings.

        Args:
            image_embeddings (torch.Tensor): Embeddings from the image encoder with shape (B, C, H, W).
            image_pe (torch.Tensor): Positional encoding with the shape of image_embeddings (B, C, H, W).
            sparse_prompt_embeddings (torch.Tensor): Embeddings of the points and boxes with shape (B, N, C).
            dense_prompt_embeddings (torch.Tensor): Embeddings of the mask inputs with shape (B, C, H, W).
            multimask_output (bool): Whether to return multiple masks or a single mask.
            repeat_image (bool): Flag to repeat the image embeddings.
            high_res_features (List[torch.Tensor] | None): Optional high-resolution features.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing:
                - masks (torch.Tensor): Batched predicted masks with shape (B, N, H, W).
                - iou_pred (torch.Tensor): Batched predictions of mask quality with shape (B, N).
                - sam_tokens_out (torch.Tensor): Batched SAM token for mask output with shape (B, N, C).
                - object_score_logits (torch.Tensor): Batched object score logits with shape (B, 1).

        Examples:
            >>> image_embeddings = torch.rand(1, 256, 64, 64)
            >>> image_pe = torch.rand(1, 256, 64, 64)
            >>> sparse_prompt_embeddings = torch.rand(1, 2, 256)
            >>> dense_prompt_embeddings = torch.rand(1, 256, 64, 64)
            >>> decoder = SAM2MaskDecoder(256, transformer)
            >>> masks, iou_pred, sam_tokens_out, obj_score_logits = decoder.forward(
            ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, True, False
            ... )
        """
        ...
    
    def predict_masks(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor, repeat_image: bool, high_res_features: Optional[List[torch.Tensor]] = ...) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts instance segmentation masks from image and prompt embeddings using a transformer."""
        ...
    


