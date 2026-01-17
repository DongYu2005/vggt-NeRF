"""
NLP Head for predicting NeRF MLP parameters.

This head fuses multi-layer transformer features (similar to DPTHead) 
and predicts the parameters for a NeRF MLP network.

NeRF MLP Structure:
- Linear layer: (63, 128) - positional encoded position input
- Sigma output layer: (128, 1) - density prediction
- Color output layer: (128 + 27, 3) - RGB color prediction with direction encoding
"""

import os
from typing import List, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import create_uv_grid, position_grid_to_embed


class NeRFMLP(nn.Module):
    """
    Simple NeRF MLP module that can be initialized with predicted parameters.
    
    Structure:
        - pos_linear: (63, 128) - processes positional encoded position
        - sigma_linear: (128, 1) - outputs density
        - color_linear: (128 + 27, 3) - outputs RGB color
    """
    
    # Parameter dimensions
    POS_IN_DIM = 63      # 3 + 3 * 2 * 10 (position + positional encoding)
    HIDDEN_DIM = 128
    DIR_IN_DIM = 27      # 3 + 3 * 2 * 4 (direction + positional encoding)
    SIGMA_OUT_DIM = 1
    COLOR_OUT_DIM = 3
    
    # Parameter counts
    POS_LINEAR_PARAMS = POS_IN_DIM * HIDDEN_DIM + HIDDEN_DIM  # 63*128 + 128 = 8192
    SIGMA_LINEAR_PARAMS = HIDDEN_DIM * SIGMA_OUT_DIM + SIGMA_OUT_DIM  # 128*1 + 1 = 129
    COLOR_LINEAR_PARAMS = (HIDDEN_DIM + DIR_IN_DIM) * COLOR_OUT_DIM + COLOR_OUT_DIM  # 155*3 + 3 = 468
    
    TOTAL_PARAMS = POS_LINEAR_PARAMS + SIGMA_LINEAR_PARAMS + COLOR_LINEAR_PARAMS  # 8789
    
    def __init__(self):
        super().__init__()
        self.pos_linear = nn.Linear(self.POS_IN_DIM, self.HIDDEN_DIM)
        self.sigma_linear = nn.Linear(self.HIDDEN_DIM, self.SIGMA_OUT_DIM)
        self.color_linear = nn.Linear(self.HIDDEN_DIM + self.DIR_IN_DIM, self.COLOR_OUT_DIM)
        
    def forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the NeRF MLP.
        
        Args:
            pos_encoded: Positional encoded position, shape [..., 63]
            dir_encoded: Positional encoded direction, shape [..., 27]
            
        Returns:
            sigma: Density, shape [..., 1]
            color: RGB color, shape [..., 3]
        """
        h = F.relu(self.pos_linear(pos_encoded))
        sigma = self.sigma_linear(h)
        
        # Concatenate hidden features with direction encoding for color prediction
        h_with_dir = torch.cat([h, dir_encoded], dim=-1)
        color = torch.sigmoid(self.color_linear(h_with_dir))
        
        return sigma, color
    
    def load_from_params(self, params: torch.Tensor) -> None:
        """
        Load MLP parameters from a flattened parameter tensor.
        
        Args:
            params: Flattened parameters tensor of shape [TOTAL_PARAMS] or [B, TOTAL_PARAMS]
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)
            
        # Extract parameters for each layer
        idx = 0
        
        # pos_linear: weight (128, 63), bias (128)
        pos_weight_size = self.POS_IN_DIM * self.HIDDEN_DIM
        pos_weight = params[:, idx:idx + pos_weight_size].view(-1, self.HIDDEN_DIM, self.POS_IN_DIM)
        idx += pos_weight_size
        pos_bias = params[:, idx:idx + self.HIDDEN_DIM]
        idx += self.HIDDEN_DIM
        
        # sigma_linear: weight (1, 128), bias (1)
        sigma_weight_size = self.HIDDEN_DIM * self.SIGMA_OUT_DIM
        sigma_weight = params[:, idx:idx + sigma_weight_size].view(-1, self.SIGMA_OUT_DIM, self.HIDDEN_DIM)
        idx += sigma_weight_size
        sigma_bias = params[:, idx:idx + self.SIGMA_OUT_DIM]
        idx += self.SIGMA_OUT_DIM
        
        # color_linear: weight (3, 155), bias (3)
        color_in_dim = self.HIDDEN_DIM + self.DIR_IN_DIM
        color_weight_size = color_in_dim * self.COLOR_OUT_DIM
        color_weight = params[:, idx:idx + color_weight_size].view(-1, self.COLOR_OUT_DIM, color_in_dim)
        idx += color_weight_size
        color_bias = params[:, idx:idx + self.COLOR_OUT_DIM]
        
        # Set parameters (using first batch element if batched)
        self.pos_linear.weight.data = pos_weight[0]
        self.pos_linear.bias.data = pos_bias[0]
        self.sigma_linear.weight.data = sigma_weight[0]
        self.sigma_linear.bias.data = sigma_bias[0]
        self.color_linear.weight.data = color_weight[0]
        self.color_linear.bias.data = color_bias[0]


class NLPHead(nn.Module):
    """
    NLP (NeRF Linear Parameters) Head for predicting NeRF MLP parameters.
    
    This head processes features from a vision transformer backbone and produces
    parameters for a NeRF MLP network. It fuses multi-scale features similar to DPTHead.
    
    Args:
        dim_in (int): Input dimension (channels) from transformer.
        patch_size (int, optional): Patch size. Default is 14.
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        mlp_hidden_dim (int, optional): Hidden dimension for parameter prediction MLP. Default is 512.
    """
    
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        mlp_hidden_dim: int = 512,
    ) -> None:
        super(NLPHead, self).__init__()
        
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.intermediate_layer_idx = intermediate_layer_idx
        
        # Total number of NeRF MLP parameters to predict
        self.nerf_param_count = NeRFMLP.TOTAL_PARAMS  # 8789
        
        self.norm = nn.LayerNorm(dim_in)
        
        # Projection layers for each output channel from tokens
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) 
            for oc in out_channels
        ])
        
        # Resize layers for feature maps (different scales)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0], out_channels=out_channels[0], 
                kernel_size=4, stride=4, padding=0
            ),
            nn.ConvTranspose2d(
                in_channels=out_channels[1], out_channels=out_channels[1], 
                kernel_size=2, stride=2, padding=0
            ),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3], out_channels=out_channels[3], 
                kernel_size=3, stride=2, padding=1
            ),
        ])
        
        # Scratch layers for feature fusion
        self.scratch = _make_scratch(out_channels, features, expand=False)
        
        # Fusion blocks
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        
        # Output convolution for fused features
        self.scratch.output_conv = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        
        # Global pooling and MLP for parameter prediction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP to predict NeRF parameters from pooled features
        self.param_predictor = nn.Sequential(
            nn.Linear(features, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, self.nerf_param_count),
        )
        
        # Initialize the last layer with small weights for stable training
        nn.init.normal_(self.param_predictor[-1].weight, std=0.01)
        nn.init.zeros_(self.param_predictor[-1].bias)
        
    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> torch.Tensor:
        """
        Forward pass through the NLP head.
        
        Args:
            aggregated_tokens_list (List[Tensor]):   [B, S, P, 2C]
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
            frames_chunk_size (int, optional): Number of frames to process in each chunk. Default: 8.
            
        Returns:
            Tensor: Predicted NeRF MLP parameters with shape [B, nerf_param_count]
        """
        B, S, _, H, W = images.shape
        
        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)
        
        # Process frames in chunks and aggregate
        assert frames_chunk_size > 0
        
        all_params = []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            chunk_params = self._forward_impl(
                aggregated_tokens_list, images, patch_start_idx, 
                frames_start_idx, frames_end_idx
            )
            all_params.append(chunk_params)
        
        # Average parameters across chunks
        return torch.stack(all_params, dim=0).mean(dim=0)
    
    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> torch.Tensor:
        """
        Implementation of the forward pass.
        
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.
            
        Returns:
            Tensor: Predicted NeRF MLP parameters with shape [B, nerf_param_count]
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
        
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        out = []
        dpt_idx = 0
        
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            
            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            
            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            
            out.append(x)
            dpt_idx += 1
        
        # Fuse features from multiple layers
        fused = self._scratch_forward(out)  # [B*S, features, H', W']
        
        # Apply output convolution
        fused = self.scratch.output_conv(fused)  # [B*S, features, H', W']
        
        # Global average pooling across spatial dimensions
        pooled = self.global_pool(fused)  # [B*S, features, 1, 1]
        pooled = pooled.view(B * S, -1)  # [B*S, features]
        
        # Average across all frames for each batch
        pooled = pooled.view(B, S, -1).mean(dim=1)  # [B, features]
        
        # Predict NeRF MLP parameters
        params = self.param_predictor(pooled)  # [B, nerf_param_count]
        
        return params
    
    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed
    
    def _scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.
        
        Args:
            features (List[Tensor]): List of feature maps from different layers.
            
        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features
        
        # 统一为 [B*S, output_channel, H', W'] 
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # 融合的过程中调整H,W统一到
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        
        out = self.scratch.refinenet1(out, layer_1_rn)
        # 最终H,W变成layer_1_rn的H,W,为input[0]*features (256,256)
        
        del layer_1_rn, layer_1
        
        return out
    
    def create_nerf_mlp(self, params: torch.Tensor) -> NeRFMLP:
        """
        Create a NeRF MLP initialized with predicted parameters.
        
        Args:
            params: Predicted parameters tensor of shape [B, nerf_param_count] or [nerf_param_count]
            
        Returns:
            NeRFMLP: Initialized NeRF MLP module
        """
        mlp = NeRFMLP()
        mlp.load_from_params(params)
        return mlp


def run_nerf_training(
    nerf_mlp: NeRFMLP,
    images: torch.Tensor,
    camera_params: Optional[torch.Tensor] = None,
    num_iterations: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Execute one iteration of NeRF training/rendering.
    
    This is a placeholder function that will be filled in later with the actual
    NeRF training logic.
    
    Args:
        nerf_mlp: The NeRF MLP model with predicted parameters
        images: Target images with shape [B, S, 3, H, W]
        camera_params: Camera parameters for each view (optional)
        num_iterations: Number of training iterations
        
    Returns:
        Dict containing:
            - 'rendered_images': Rendered images
            - 'loss': Training loss
    """
    # TODO: Implement NeRF training logic
    # This should include:
    # 1. Ray generation from camera parameters
    # 2. Point sampling along rays
    # 3. Positional encoding of points and directions
    # 4. MLP forward pass to get sigma and color
    # 5. Volume rendering to get pixel colors
    # 6. Loss computation against target images
    
    B, S, C, H, W = images.shape
    
    # Placeholder outputs
    rendered_images = torch.zeros_like(images)
    loss = torch.tensor(0.0, device=images.device)
    
    return {
        'rendered_images': rendered_images,
        'loss': loss,
    }


################################################################################
# Modules (adapted from DPTHead)
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features: int, activation: nn.Module, bn: bool, groups: int = 1):
        super().__init__()
        
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.norm1 = None
        self.norm2 = None
        
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: int = None,
        has_residual: bool = True,
        groups: int = 1,
    ):
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None) -> torch.Tensor:
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output
