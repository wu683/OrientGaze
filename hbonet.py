import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
import timm
import os
from .bimamba import BidirectionalMamba


# ==========================================================
# Multimodal Feature Extraction
# ==========================================================
class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = EfficientNet.from_name('efficientnet-b0', include_top=False)
        self.stem = nn.Sequential(*list(model.children())[:2])
    def forward(self, image):
        return self.stem(image)
class HeadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_branch = EfficientNet.from_name('efficientnet-b0', include_top=False)
        self.head_branch._conv_stem = nn.Identity()
        self.head_branch._bn0 = nn.Identity()
    def forward(self, feature):
        return self.head_branch(feature)

class BodyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body_branch = EfficientNet.from_name('efficientnet-b0', include_top=False)
        self.body_branch._conv_stem = nn.Identity()
        self.body_branch._bn0 = nn.Identity()
    def forward(self, feature):
        return self.body_branch.extract_features(feature)

class TrajectoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16), nn.LeakyReLU(), nn.Linear(16, 32),
        )
    def forward(self, dv):
        return self.fc(dv)

# ---------- 场景分支 ----------
class SceneEncoder(nn.Module):
    def __init__(self,  freeze=True, image_size=224, patch_size=14):
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

# ---------- 多模态对齐模块----------
class HCCA(nn.Module):
    def __init__(self, head_dim=1280, body_channels=1280,  scene_dim=384,
                 align_dim=256, num_heads=4):
        super().__init__()
        self.proj_head = nn.Linear(head_dim, align_dim)
        self.proj_body = nn.Conv2d(body_channels, align_dim, kernel_size=1)
        self.proj_scene = nn.Linear(scene_dim, align_dim)

        self.cross_attn_body = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)
        self.cross_attn_scene = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)


    def forward(self, head_feat, body_feat_map,  scene_tokens):
        B, T, _ = head_feat.shape
        # 投影
        H = self.proj_head(head_feat)                       # (B, T, 256)
        body_proj = self.proj_body(body_feat_map)           # (B*T, 256, h, w)

        # 身体对齐
        body_tokens = body_proj.flatten(2).transpose(1, 2)  # (B*T, N_body, 256)
        head_query = H.reshape(B * T, 1, 256)
        aligned_body, _ = self.cross_attn_body(query=head_query, key=body_tokens, value=body_tokens)
        aligned_body = aligned_body.squeeze(1)               # (B*T, 256)

        # 场景对齐
        scene_proj = self.proj_scene(scene_tokens)           # (B*T, N_scene, 256)
        aligned_scene, _ = self.cross_attn_scene(query=head_query, key=scene_proj, value=scene_proj)
        aligned_scene = aligned_scene.squeeze(1)             # (B*T, 256)

        # 原始头部特征（作为 query 自身）
        head_proj = H.reshape(B * T, 256)
        return {
                    "head": head_proj.reshape(B,T,256),
                    "align_body": aligned_body.reshape(B,T,256),
                    "align_scene": aligned_scene.reshape(B,T,256)
               }



class MotionAwareTemporalFusion(nn.Module):
    def __init__(
            self,
            dim=256,
            traj_dim=32,
            hidden_dim=64
    ):
        super().__init__()

        # Trajectory projection
        self.proj_traj = nn.Linear(traj_dim, dim)

        # 多模态融合
        self.fusion = nn.Sequential(
            nn.Linear(
                dim * 4 ,
                dim
            ),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        # Temporal modeling
        self.bimamba = BidirectionalMamba(
            input_dim=dim,
            hidden_dim=hidden_dim,
            d_model=256,
            num_layers=2
        )

    def forward(
            self,
            aligned_feat,
            traj_feat
    ):
        head_feat = aligned_feat["head"]
        body_feat = aligned_feat["align_body"]
        scene_feat = aligned_feat["align_scene"]
        # Trajectory: 32 → 256
        traj_feat = self.proj_traj(traj_feat)
        x = torch.cat(
            [ head_feat, body_feat, scene_feat, traj_feat ],
            dim=-1
        )
        x = self.fusion(x)
        x = self.bimamba(x)
        return x


class DirectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)



# ---------- 新 HBNet ----------
class HBONet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.shared_backbone = SharedBackbone()
        self.head_encoder = HeadEncoder()
        self.body_encoder = BodyEncoder()
        self.traj_encoder = TrajectoryEncoder()
        self.scene_encoder = SceneEncoder(freeze=True)

        # 多模态对齐（含场景）
        self.hcca = HCCA(
            head_dim=1280, body_channels=1280, scene_dim=384,
            align_dim=256, num_heads=4
        )

        # 双向 Mamba 时序
        self.mtf = MotionAwareTemporalFusion(dim=256,traj_dim=32, hidden_dim=64)

        # 方向输出头
        self.head_dir_head = DirectionHead(in_dim=64, hidden_dim=16)
        self.body_dir_head = DirectionHead(in_dim=64, hidden_dim=16)

        self.automatic_optimization = False

    def forward(self, image, head_mask, body_dv):
        B, T, C, H, W = image.shape
        image_flat = image.reshape(B * T, C, H, W)
        head_mask_flat = head_mask.reshape(B * T, 1, H, W)

        # 共享低层特征
        shared_feat = self.shared_backbone(image_flat)
        attention_map = F.interpolate(
            head_mask_flat, size=shared_feat.shape[-2:],
            mode='bilinear', align_corners=False
        )
        head_input = shared_feat * attention_map

        # 头/体分支
        head_feat = self.head_encoder(head_input)               # (B*T, 1280)
        body_feat_map = self.body_encoder(shared_feat)          # (B*T, 1280, h, w)
        traj_feat = self.traj_encoder(body_dv)                  # (B, T, 32)


        self.scene_encoder.eval()
        with torch.no_grad():
            scene_tokens = self.scene_encoder(image_flat)

        # 恢复时序维度
        head_feat = head_feat.reshape(B, T, 1280)

        # 多模态对齐
        aligned_feat = self.hcca(head_feat, body_feat_map, scene_tokens)  # (B, T, 256)

        # Mamba 时序建模
        temporal_feat = self.mtf(aligned_feat,traj_feat)    # (B, T, 64)

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