import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet


# ==========================================================
# Shared Backbone
# ==========================================================
class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        model = EfficientNet.from_name(
            'efficientnet-b0',
            include_top=False
        )

        self.stem = nn.Sequential(
            *list(model.children())[:2]
        )

    def forward(self, image):
        return self.stem(image)


# ==========================================================
# Head Encoder
# ==========================================================
class HeadEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.head_branch = EfficientNet.from_name(
            'efficientnet-b0',
            include_top=False
        )

        self.head_branch._conv_stem = nn.Identity()
        self.head_branch._bn0 = nn.Identity()

    def forward(self, feature):
        return self.head_branch(feature)


# ==========================================================
# Body Encoder
# ==========================================================
class BodyEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.body_branch = EfficientNet.from_name(
            'efficientnet-b0',
            include_top=False
        )

        self.body_branch._conv_stem = nn.Identity()
        self.body_branch._bn0 = nn.Identity()

    def forward(self, feature):
        return self.body_branch.extract_features(feature)


# ==========================================================
# Trajectory Encoder
# ==========================================================
class TrajectoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
        )

    def forward(self, dv):
        return self.fc(dv)


# ==========================================================
# HCCA
# Head-Body Cross-modal Alignment
#
# 注意：
# 这里是仅 HCCA 消融实验，因此：
#   Head -> Body
# 保留
#
# Scene 分支：
#   Head -> Scene
# 已完全去掉
# ==========================================================
class HCCA(nn.Module):

    def __init__(
            self,
            head_dim=1280,
            body_channels=1280,
            align_dim=256,
            num_heads=4
    ):
        super().__init__()

        # --------------------------------------------------
        # Head: 1280 -> 256
        # --------------------------------------------------
        self.proj_head = nn.Linear(
            head_dim,
            align_dim
        )

        # --------------------------------------------------
        # Body feature map:
        # 1280 -> 256
        # --------------------------------------------------
        self.proj_body = nn.Conv2d(
            body_channels,
            align_dim,
            kernel_size=1
        )

        # --------------------------------------------------
        # Head -> Body
        # --------------------------------------------------
        self.cross_attn_body = nn.MultiheadAttention(
            embed_dim=align_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(
            self,
            head_feat,
            body_feat_map
    ):

        B, T, _ = head_feat.shape

        # ==================================================
        # Head projection
        # ==================================================

        H = self.proj_head(
            head_feat
        )

        # H:
        # (B,T,256)

        # ==================================================
        # Body projection
        # ==================================================

        body_proj = self.proj_body(
            body_feat_map
        )

        # body_proj:
        # (B*T,256,h,w)

        # ==================================================
        # Body spatial features -> tokens
        # ==================================================

        body_tokens = body_proj.flatten(
            2
        ).transpose(
            1,
            2
        )

        # (B*T,N_body,256)

        # ==================================================
        # Head as Query
        # ==================================================

        head_query = H.reshape(
            B * T,
            1,
            256
        )

        # (B*T,1,256)

        # ==================================================
        # Head-Body Cross Attention
        # ==================================================

        aligned_body, _ = self.cross_attn_body(
            query=head_query,
            key=body_tokens,
            value=body_tokens
        )

        aligned_body = aligned_body.squeeze(
            1
        )

        # (B*T,256)

        aligned_body = aligned_body.reshape(
            B,
            T,
            256
        )

        # ==================================================
        # Return
        # ==================================================

        return {
            'head': H,
            'align_body': aligned_body
        }


# ==========================================================
# HCCA + BiLSTM Temporal Fusion
# ==========================================================
class HCCATemporalFusion(nn.Module):

    def __init__(
            self,
            dim=256,
            traj_dim=32,
            hidden_dim=64
    ):
        super().__init__()

        # ==================================================
        # HCCA
        # ==================================================

        self.hcca = HCCA(
            head_dim=1280,
            body_channels=1280,
            align_dim=dim,
            num_heads=4
        )

        # ==================================================
        # Trajectory projection
        # ==================================================

        self.proj_traj = nn.Linear(
            traj_dim,
            dim
        )

        # ==================================================
        # 三个模态
        #
        # Head      : 256
        # Body      : 256
        # Trajectory: 256
        #
        # Concatenated dimension = 768
        # ==================================================

        self.fusion = nn.Sequential(

            nn.Linear(
                dim * 3,
                dim
            ),

            nn.LayerNorm(dim),

            nn.GELU()
        )

        # ==================================================
        # BiLSTM
        #
        # input  = 256
        # hidden = 64
        # bidirectional
        #
        # output = 64 * 2 = 128
        # ==================================================

        self.lstm = nn.LSTM(

            input_size=dim,

            hidden_size=hidden_dim,

            num_layers=2,

            batch_first=True,

            bidirectional=True
        )

        self.output_dim = hidden_dim * 2

    def forward(
            self,
            head_feat,
            body_feat_map,
            traj_feat
    ):

        # ==================================================
        # HCCA
        # ==================================================

        aligned_feat = self.hcca(
            head_feat,
            body_feat_map
        )

        head_feat = aligned_feat[
            'head'
        ]

        body_feat = aligned_feat[
            'align_body'
        ]

        # ==================================================
        # Trajectory projection
        # ==================================================

        traj_feat = self.proj_traj(
            traj_feat
        )

        # ==================================================
        # Multimodal fusion
        # ==================================================

        x = torch.cat(
            [
                head_feat,
                body_feat,
                traj_feat
            ],
            dim=-1
        )

        # (B,T,768)

        x = self.fusion(
            x
        )

        # (B,T,256)

        # ==================================================
        # Temporal modeling
        # ==================================================

        x, _ = self.lstm(
            x
        )

        # (B,T,128)

        return x


# ==========================================================
# Direction Head
# ==========================================================
class DirectionHead(nn.Module):

    def __init__(
            self,
            in_dim,
            hidden_dim=16
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                in_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                3
            )
        )

    def forward(self, x):

        return F.normalize(
            self.net(x),
            dim=-1
        )


# ==========================================================
# HBONet
# Only HCCA + BiLSTM
#
# Scene removed
# ==========================================================
class HBONet(pl.LightningModule):

    def __init__(self):

        super().__init__()

        # ==================================================
        # Shared Backbone
        # ==================================================

        self.shared_backbone = SharedBackbone()

        # ==================================================
        # Head / Body
        # ==================================================

        self.head_encoder = HeadEncoder()

        self.body_encoder = BodyEncoder()

        # ==================================================
        # Trajectory
        # ==================================================

        self.traj_encoder = TrajectoryEncoder()

        # ==================================================
        # HCCA + BiLSTM
        # ==================================================

        self.temporal_fusion = HCCATemporalFusion(
            dim=256,
            traj_dim=32,
            hidden_dim=64
        )

        # ==================================================
        # BiLSTM output:
        #
        # 64 * 2 = 128
        # ==================================================

        self.head_dir_head = DirectionHead(
            in_dim=128,
            hidden_dim=16
        )

        self.body_dir_head = DirectionHead(
            in_dim=128,
            hidden_dim=16
        )

        self.automatic_optimization = False

    def forward(
            self,
            image,
            head_mask,
            body_dv
    ):

        B, T, C, H, W = image.shape

        # ==================================================
        # Flatten temporal dimension
        # ==================================================

        image_flat = image.reshape(
            B * T,
            C,
            H,
            W
        )

        head_mask_flat = head_mask.reshape(
            B * T,
            1,
            H,
            W
        )

        # ==================================================
        # Shared low-level feature
        # ==================================================

        shared_feat = self.shared_backbone(
            image_flat
        )

        # ==================================================
        # Head
        # ==================================================

        attention_map = F.interpolate(
            head_mask_flat,
            size=shared_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        head_input = (
            shared_feat *
            attention_map
        )

        head_feat = self.head_encoder(
            head_input
        )

        # (B*T,1280)

        # ==================================================
        # Body
        # ==================================================

        body_feat_map = self.body_encoder(
            shared_feat
        )

        # (B*T,1280,h,w)

        # ==================================================
        # Trajectory
        # ==================================================

        traj_feat = self.traj_encoder(
            body_dv
        )

        # (B,T,32)

        # ==================================================
        # Restore temporal dimension
        # ==================================================

        head_feat = head_feat.reshape(
            B,
            T,
            1280
        )

        # ==================================================
        # HCCA + BiLSTM
        # ==================================================

        temporal_feat = self.temporal_fusion(
            head_feat,
            body_feat_map,
            traj_feat
        )

        # (B,T,128)

        # ==================================================
        # Direction prediction
        # ==================================================

        head_dir = self.head_dir_head(
            temporal_feat
        )

        body_dir = self.body_dir_head(
            temporal_feat
        )

        # ==================================================
        # Output
        # ==================================================

        return {

            'head_dir':
                head_dir,

            'body_dir':
                body_dir,

            # 原始头部特征
            # 供 GazeNet 的 MMD 使用

            'head_feat':
                head_feat,

            # 原始身体全局特征
            # 供 GazeNet 的 MMD 使用

            'body_feat':
                body_feat_map.mean(
                    dim=[2, 3]
                ).reshape(
                    B,
                    T,
                    1280
                )
        }