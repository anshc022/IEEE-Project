"""
This type stub file was generated by pyright.
"""

import torch
from typing import Tuple

def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num): # -> tuple[Any | dict[Any, Any], dict[Any, Any]]:
    """
    Selects the closest conditioning frames to a given frame index.

    Args:
        frame_idx (int): Current frame index.
        cond_frame_outputs (Dict[int, Any]): Dictionary of conditioning frame outputs keyed by frame indices.
        max_cond_frame_num (int): Maximum number of conditioning frames to select.

    Returns:
        (Tuple[Dict[int, Any], Dict[int, Any]]): A tuple containing two dictionaries:
            - selected_outputs: Selected items from cond_frame_outputs.
            - unselected_outputs: Items not selected from cond_frame_outputs.

    Examples:
        >>> frame_idx = 5
        >>> cond_frame_outputs = {1: "a", 3: "b", 7: "c", 9: "d"}
        >>> max_cond_frame_num = 2
        >>> selected, unselected = select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num)
        >>> print(selected)
        {3: 'b', 7: 'c'}
        >>> print(unselected)
        {1: 'a', 9: 'd'}
    """
    ...

def get_1d_sine_pe(pos_inds, dim, temperature=...): # -> Tensor:
    """Generates 1D sinusoidal positional embeddings for given positions and dimensions."""
    ...

def init_t_xy(end_x: int, end_y: int): # -> tuple[Tensor, Tensor]:
    """Initializes 1D and 2D coordinate tensors for a grid of specified dimensions."""
    ...

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = ...): # -> Tensor:
    """Computes axial complex exponential positional encodings for 2D spatial positions in a grid."""
    ...

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor): # -> Tensor:
    """Reshapes frequency tensor for broadcasting with input tensor, ensuring dimensional compatibility."""
    ...

def apply_rotary_enc(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, repeat_freqs_k: bool = ...): # -> tuple[Tensor, Tensor]:
    """Applies rotary positional encoding to query and key tensors using complex-valued frequency components."""
    ...

def window_partition(x, window_size): # -> tuple[Tensor | Any, tuple[Any, Any]]:
    """
    Partitions input tensor into non-overlapping windows with padding if needed.

    Args:
        x (torch.Tensor): Input tensor with shape (B, H, W, C).
        window_size (int): Size of each window.

    Returns:
        (Tuple[torch.Tensor, Tuple[int, int]]): A tuple containing:
            - windows (torch.Tensor): Partitioned windows with shape (B * num_windows, window_size, window_size, C).
            - (Hp, Wp) (Tuple[int, int]): Padded height and width before partition.

    Examples:
        >>> x = torch.randn(1, 16, 16, 3)
        >>> windows, (Hp, Wp) = window_partition(x, window_size=4)
        >>> print(windows.shape, Hp, Wp)
        torch.Size([16, 4, 4, 3]) 16 16
    """
    ...

def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Unpartitions windowed sequences into original sequences and removes padding.

    This function reverses the windowing process, reconstructing the original input from windowed segments
    and removing any padding that was added during the windowing process.

    Args:
        windows (torch.Tensor): Input tensor of windowed sequences with shape (B * num_windows, window_size,
            window_size, C), where B is the batch size, num_windows is the number of windows, window_size is
            the size of each window, and C is the number of channels.
        window_size (int): Size of each window.
        pad_hw (Tuple[int, int]): Padded height and width (Hp, Wp) of the input before windowing.
        hw (Tuple[int, int]): Original height and width (H, W) of the input before padding and windowing.

    Returns:
        (torch.Tensor): Unpartitioned sequences with shape (B, H, W, C), where B is the batch size, H and W
            are the original height and width, and C is the number of channels.

    Examples:
        >>> windows = torch.rand(32, 8, 8, 64)  # 32 windows of size 8x8 with 64 channels
        >>> pad_hw = (16, 16)  # Padded height and width
        >>> hw = (15, 14)  # Original height and width
        >>> x = window_unpartition(windows, window_size=8, pad_hw=pad_hw, hw=hw)
        >>> print(x.shape)
        torch.Size([1, 15, 14, 64])
    """
    ...

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Extracts relative positional embeddings based on query and key sizes.

    Args:
        q_size (int): Size of the query.
        k_size (int): Size of the key.
        rel_pos (torch.Tensor): Relative position embeddings with shape (L, C), where L is the maximum relative
            distance and C is the embedding dimension.

    Returns:
        (torch.Tensor): Extracted positional embeddings according to relative positions, with shape (q_size,
            k_size, C).

    Examples:
        >>> q_size, k_size = 8, 16
        >>> rel_pos = torch.randn(31, 64)  # 31 = 2 * max(8, 16) - 1
        >>> extracted_pos = get_rel_pos(q_size, k_size, rel_pos)
        >>> print(extracted_pos.shape)
        torch.Size([8, 16, 64])
    """
    ...

def add_decomposed_rel_pos(attn: torch.Tensor, q: torch.Tensor, rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor, q_size: Tuple[int, int], k_size: Tuple[int, int]) -> torch.Tensor:
    """
    Adds decomposed Relative Positional Embeddings to the attention map.

    This function calculates and applies decomposed Relative Positional Embeddings as described in the MVITv2
    paper. It enhances the attention mechanism by incorporating spatial relationships between query and key
    positions.

    Args:
        attn (torch.Tensor): Attention map with shape (B, q_h * q_w, k_h * k_w).
        q (torch.Tensor): Query tensor in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (torch.Tensor): Relative position embeddings for height axis with shape (Lh, C).
        rel_pos_w (torch.Tensor): Relative position embeddings for width axis with shape (Lw, C).
        q_size (Tuple[int, int]): Spatial sequence size of query q as (q_h, q_w).
        k_size (Tuple[int, int]): Spatial sequence size of key k as (k_h, k_w).

    Returns:
        (torch.Tensor): Updated attention map with added relative positional embeddings, shape
            (B, q_h * q_w, k_h * k_w).

    Examples:
        >>> B, C, q_h, q_w, k_h, k_w = 1, 64, 8, 8, 8, 8
        >>> attn = torch.rand(B, q_h * q_w, k_h * k_w)
        >>> q = torch.rand(B, q_h * q_w, C)
        >>> rel_pos_h = torch.rand(2 * max(q_h, k_h) - 1, C)
        >>> rel_pos_w = torch.rand(2 * max(q_w, k_w) - 1, C)
        >>> q_size, k_size = (q_h, q_w), (k_h, k_w)
        >>> updated_attn = add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size)
        >>> print(updated_attn.shape)
        torch.Size([1, 64, 64])

    References:
        https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py
    """
    ...

