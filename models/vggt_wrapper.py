import torch
import torch.nn as nn
from vggt.models.vggt import VGGT
import os
class VGGTFeatureExtractor(nn.Module):
    """
    Wrapper around the official VGGT model to extract dense feature maps.
    支持加载预训练参数并冻结权重。
    """
    def __init__(self, config):
        super().__init__()
        
        print("Initializing VGGT model...")

        # 1. 准备配置参数 (关掉不用的 Head 以节省显存)
        # 这些参数会传递给模型初始化
        model_kwargs = {
            "img_size": getattr(config, 'vggt_img_size', 518),
            "patch_size": getattr(config, 'vggt_patch_size', 14),
            "embed_dim": getattr(config, 'vggt_embed_dim', 1024),
            "enable_camera": False, 
            "enable_point": False, 
            "enable_depth": False, 
            "enable_track": False, 
            "enable_nlp": False
        }

        # 2. 加载模型
        # 逻辑：如果 config 里指定了 vggt_ckpt 路径，就用本地文件；否则从 Hugging Face 下载
        if hasattr(config, 'vggt_ckpt') and config.vggt_ckpt is not None:
            ckpt_path = config.vggt_ckpt
            print(f"🚀 Checkpoint path provided: {ckpt_path}")
            if os.path.isdir(ckpt_path):
                    print(f"📂 Detected Folder. Loading via VGGT.from_pretrained()...")
                    self.model = VGGT.from_pretrained(ckpt_path, **model_kwargs)
                    print("✅ HuggingFace weights loaded successfully from folder!")
            elif hasattr(config, 'vggt_ckpt') and config.vggt_ckpt is not None:
                print(f"Loading VGGT from LOCAL file: {config.vggt_ckpt}")
                # 本地模式：手动初始化 + 手动加载权重 (复用你之前的逻辑)
                self.model = VGGT(**model_kwargs)
                try:
                    ckpt = torch.load(config.vggt_ckpt, map_location='cpu')
                    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
                    # 过滤并加载
                    feature_weights = {k: v for k, v in state_dict.items() if 'aggregator' in k}
                    self.model.load_state_dict(feature_weights, strict=False)
                    print("Local weights loaded successfully!")
                except Exception as e:
                    print(f"ERROR loading local weights: {e}")
            else:
                print(f"❌ Error: Path does not exist: {ckpt_path}")
                # 这种情况下为了防止随机训练，建议直接抛错
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        else:
            print("Loading VGGT from Hugging Face (Automatic Download: facebook/VGGT-1B)...")
            # 自动模式：from_pretrained 会自动下载权重、初始化结构、并应用 model_kwargs
            # 一步到位，不需要手动 torch.load
            local_model_path = "./checkpoints/VGGT-1B"
            # 检查一下路径对不对，防止手滑写错
            if os.path.exists(local_model_path):
                print(f"Loading VGGT from local path: {local_model_path}")
                self.model = VGGT.from_pretrained(local_model_path, **model_kwargs)
            else:
                print(f"Error: Local path {local_model_path} not found!")
                # 可以在这里抛出异常或者 fallback

        # 3. 冻结参数 (Freezing)
        if getattr(config, 'freeze_vggt', True):
            print("Freezing VGGT weights (Feature Extraction Mode).")
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("VGGT weights are trainable (Fine-tuning Mode).")
            self.model.train()

    def forward(self, images):
        """
        Args:
            images: [B, S, 3, H, W] Input images (Normalized [0,1])
        Returns:
            feature_maps: [B, S, C, H_feat, W_feat]
        """
        # 确保输入维度正确
        if images.dim() == 4: # [B, 3, H, W] -> [B, 1, 3, H, W]
            images = images.unsqueeze(1)
            
        B, S, C, H, W = images.shape
        
        # 如果冻结了，使用 no_grad 上下文以节省显存
        # 如果没冻结，则正常计算梯度
        is_frozen = not next(self.model.parameters()).requires_grad
        
        with torch.set_grad_enabled(not is_frozen):
            # 1. 运行 Aggregator
            output_list, patch_start_idx = self.model.aggregator(images)
            
            # 2. 取最后一层特征
            last_layer_tokens = output_list[-1]
            if last_layer_tokens.dim() == 4:
                # [B, S, Tokens, Dim] -> [B*S, Tokens, Dim]
                tokens_flat = last_layer_tokens.view(B * S, -1, last_layer_tokens.shape[-1])
            else:
                tokens_flat = last_layer_tokens
            # 3. 剥离 Special Tokens
            patch_tokens = tokens_flat[:, patch_start_idx:, :]
            
            # 4. Reshape
            patch_size = self.model.aggregator.patch_size
            H_feat = H // patch_size
            W_feat = W // patch_size
            feat_dim = patch_tokens.shape[-1]
            
            feature_maps = patch_tokens.view(B * S, H_feat, W_feat, feat_dim)
            feature_maps = feature_maps.permute(0, 3, 1, 2)
            feature_maps = feature_maps.view(B, S, feat_dim, H_feat, W_feat)
            
        return feature_maps