import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import timm
import os

# ---------- Mamba 导入 ----------
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("mamba_ssm not found, please install it")

# ---------- 场景分支 ----------
class SceneTokenExtractor(nn.Module):
    def __init__(self, output_dim=256, freeze=True, image_size=224, patch_size=14):
        super().__init__()
        self.dino = timm.create_model(
            'vit_small_patch14_dinov2.lvd142m',
            pretrained=False,
            num_classes=0,
            img_size=image_size
        )
        # 从本地加载权重
        local_weight_path = '/home/user/.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m/snapshots/4610ca143709d58a633b6397a74412c2c3842454/model.safetensors'
        if os.path.exists(local_weight_path):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(local_weight_path)
                if 'pos_embed' in state_dict:
                    pos_ckpt = state_dict['pos_embed']
                    pos_model = self.dino.pos_embed.data
                    if pos_ckpt.shape != pos_model.shape:
                        n_prefix = self.dino.num_prefix_tokens
                        cls_token = pos_ckpt[:, :n_prefix, :]
                        patch_ckpt = pos_ckpt[:, n_prefix:, :]
                        h_ckpt = int(patch_ckpt.shape[1] ** 0.5)
                        h_model = image_size // patch_size
                        patch_ckpt = patch_ckpt.reshape(1, h_ckpt, h_ckpt, -1).permute(0, 3, 1, 2)
                        patch_new = F.interpolate(patch_ckpt, size=(h_model, h_model), mode='bilinear', align_corners=False)
                        patch_new = patch_new.permute(0, 2, 3, 1).reshape(1, -1, self.dino.embed_dim)
                        new_pos_embed = torch.cat([cls_token, patch_new], dim=1)
                        state_dict['pos_embed'] = new_pos_embed
                self.dino.load_state_dict(state_dict, strict=True)
                print(f"✅ DINOv2 权重已从本地加载")
            except Exception as e:
                print(f"❌ 加载本地权重失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ 本地权重文件不存在，将使用随机初始化")

        if freeze:
            for param in self.dino.parameters():
                param.requires_grad = False

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dino_dim = self.dino.embed_dim
        self.scene_fc = nn.Sequential(
            nn.LayerNorm(self.dino_dim),
            nn.Linear(self.dino_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        max_side = max(H, W)
        x_square = torch.zeros(B, C, max_side, max_side, device=x.device)
        x_square[:, :, :H, :W] = x
        x_resized = F.interpolate(
            x_square, size=(self.image_size, self.image_size),
            mode='bilinear', align_corners=False
        )
        features = self.dino.forward_features(x_resized)
        patch_tokens = features[:, 1:, :]  # (B, N_patch, 384)
        return patch_tokens  # 只返回 patch tokens，不进行池化


# ---------- 多模态对齐模块（新增场景对齐）----------
class AlignBlockWithScene(nn.Module):
    def __init__(self, head_dim=1280, body_channels=1280, traj_dim=32, scene_dim=384,
                 align_dim=256, num_heads=4):
        super().__init__()
        self.proj_head = nn.Linear(head_dim, align_dim)
        self.proj_body = nn.Conv2d(body_channels, align_dim, kernel_size=1)
        self.proj_traj = nn.Linear(traj_dim, align_dim)
        self.proj_scene = nn.Linear(scene_dim, align_dim)

        self.cross_attn_body = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)
        self.cross_attn_scene = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)

        self.fuse_proj = nn.Sequential(
            nn.Linear(align_dim * 4, align_dim),
            nn.LayerNorm(align_dim),
            nn.GELU()
        )

    def forward(self, head_feat, body_feat_map, traj_feat, scene_tokens):
        B, T, _ = head_feat.shape
        # 投影
        H = self.proj_head(head_feat)                       # (B, T, 256)
        body_proj = self.proj_body(body_feat_map)           # (B*T, 256, h, w)
        T_proj = self.proj_traj(traj_feat)                  # (B, T, 256)

        # 身体对齐
        body_tokens = body_proj.flatten(2).transpose(1, 2)  # (B*T, N_body, 256)
        head_query = H.reshape(B * T, 1, 256)
        aligned_body, _ = self.cross_attn_body(query=head_query, key=body_tokens, value=body_tokens)
        aligned_body = aligned_body.squeeze(1)               # (B*T, 256)

        # 场景对齐
        scene_proj = self.proj_scene(scene_tokens)           # (B*T, N_scene, 256)
        aligned_scene, _ = self.cross_attn_scene(query=head_query, key=scene_proj, value=scene_proj)
        aligned_scene = aligned_scene.squeeze(1)             # (B*T, 256)

        # 轨迹特征（无需对齐，直接投影）
        traj_proj = T_proj.reshape(B * T, 256)

        # 原始头部特征（作为 query 自身）
        head_proj = H.reshape(B * T, 256)

        # 拼接四个对齐后的特征
        fused = torch.cat([head_proj, aligned_body, traj_proj, aligned_scene], dim=-1)  # (B*T, 1024)
        fused = self.fuse_proj(fused)   # (B*T, 256)
        return fused.reshape(B, T, 256)


# ---------- 基础模块（保持不变）----------
class SharedFeatureNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model = EfficientNet.from_name('efficientnet-b0', include_top=False)
        self.stem = nn.Sequential(*list(model.children())[:2])
    def forward(self, image):
        return self.stem(image)

class BodyNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.body_branch = EfficientNet.from_name('efficientnet-b0', include_top=False)
        self.body_branch._conv_stem = nn.Identity()
        self.body_branch._bn0 = nn.Identity()
    def forward(self, feature):
        return self.body_branch.extract_features(feature)

class HeadNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.head_branch = EfficientNet.from_name('efficientnet-b0', include_top=False)
        self.head_branch._conv_stem = nn.Identity()
        self.head_branch._bn0 = nn.Identity()
    def forward(self, feature):
        return self.head_branch(feature)

class TrajNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16), nn.LeakyReLU(), nn.Linear(16, 32),
        )
    def forward(self, dv):
        return self.fc(dv)

class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)
    def forward(self, x):
        return x + self.mamba(self.norm(x))

class BidirectionalMamba(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, d_model=256, num_layers=2):
        super().__init__()
        if not HAS_MAMBA:
            raise RuntimeError("mamba_ssm is required")
        self.proj_in = nn.Linear(input_dim, d_model)
        self.mamba_fwd = nn.Sequential(*[MambaBlock(d_model) for _ in range(num_layers)])
        self.mamba_bwd = nn.Sequential(*[MambaBlock(d_model) for _ in range(num_layers)])
        self.proj_out = nn.Linear(2 * d_model, hidden_dim)
    def forward(self, x):
        x = self.proj_in(x)
        out_fwd = self.mamba_fwd(x)
        x_rev = torch.flip(x, dims=[1])
        out_rev = self.mamba_bwd(x_rev)
        out_rev = torch.flip(out_rev, dims=[1])
        out = torch.cat([out_fwd, out_rev], dim=-1)
        return self.proj_out(out)

class DirectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ---------- 新 HBNet ----------
class HBNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.featurenet = SharedFeatureNet()
        self.headnet = HeadNet()
        self.bodynet = BodyNet()
        self.trajnet = TrajNet()
        self.scene_extractor = SceneTokenExtractor(output_dim=256, freeze=True)

        # 多模态对齐（含场景）
        self.align_block = AlignBlockWithScene(
            head_dim=1280, body_channels=1280, traj_dim=32, scene_dim=384,
            align_dim=256, num_heads=4
        )

        # 双向 Mamba 时序
        self.temporal_mamba = BidirectionalMamba(input_dim=256, hidden_dim=64, d_model=256, num_layers=2)

        # 方向输出头
        self.head_dir_head = DirectionHead(in_dim=64, hidden_dim=16)
        self.body_dir_head = DirectionHead(in_dim=64, hidden_dim=16)

        self.automatic_optimization = False

    def forward(self, image, head_mask, body_dv):
        B, T, C, H, W = image.shape
        image_flat = image.reshape(B * T, C, H, W)
        head_mask_flat = head_mask.reshape(B * T, 1, H, W)

        # 共享低层特征
        shared_feat = self.featurenet(image_flat)
        attention_map = F.interpolate(
            head_mask_flat, size=shared_feat.shape[-2:],
            mode='bilinear', align_corners=False
        )
        head_input = shared_feat * attention_map

        # 头/体分支
        head_feat = self.headnet(head_input)               # (B*T, 1280)
        body_feat_map = self.bodynet(shared_feat)          # (B*T, 1280, h, w)
        traj_feat = self.trajnet(body_dv)                  # (B, T, 32)


        self.scene_extractor.eval()
        with torch.no_grad():
            scene_tokens = self.scene_extractor(image_flat)

        # 恢复时序维度
        head_feat = head_feat.reshape(B, T, 1280)

        # 多模态对齐 + 融合（含场景）
        fused_feat = self.align_block(head_feat, body_feat_map, traj_feat, scene_tokens)  # (B, T, 256)

        # Mamba 时序建模
        temporal_feat = self.temporal_mamba(fused_feat)    # (B, T, 64)

        # 方向预测
        head_dir = self.head_dir_head(temporal_feat)
        body_dir = self.body_dir_head(temporal_feat)

        # 在 HBNet.forward 末尾，替换原来的返回值
        return {
            'head_dir': head_dir,
            'body_dir': body_dir,
            'head_feat': head_feat,  # 头部原始全局特征 (B, T, 1280)
            'body_feat': body_feat_map.mean(dim=[2, 3]).reshape(B, T, 1280),  # 身体原始全局特征 (B, T, 1280)
        }