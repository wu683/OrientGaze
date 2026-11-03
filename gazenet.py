import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from .utils import compute_mae
from .loss import compute_basic_cos_loss, compute_mmd_loss

from.bimamba import BidirectionalMamba
HAS_MAMBA = True


class DirectionHead(nn.Module):
    def __init__(self, in_dim, n_frames, hidden_dim=128):
        super().__init__()
        self.n_frames = n_frames
        self.net = nn.Sequential(
            nn.Linear(in_dim * n_frames, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * n_frames),
        )

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        x = x.reshape(B, T * C)   # [B, T*C]
        x = self.net(x)           # [B, T*3]
        x = x.reshape(B, T, 3)    # [B, T, 3]
        return F.normalize(x, dim=-1)


class GazeNet(pl.LightningModule):
    def __init__(
            self,
            hbonet,
            n_frames=7,
            gaze_hidden_dim=128,
            temporal_hidden_dim=256,
            learning_rate=1e-4
    ):
        super().__init__()
        self.hbonet = hbonet

        self.temporal_mamba = BidirectionalMamba(
            input_dim=14,
            hidden_dim=temporal_hidden_dim,
            num_layers=2
        )

        self.gaze_head = DirectionHead(
            in_dim=temporal_hidden_dim,
            n_frames=n_frames,
            hidden_dim=gaze_hidden_dim
        )

        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['hbonet'])

        self.automatic_optimization = False
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.final_test_results = {}

    def forward(self, image, head_mask, body_dv):
        hbonet_out = self.hbonet(image, head_mask,body_dv)
        mu_h = hbonet_out["head_dir"]
        mu_b = hbonet_out["body_dir"]
        v_t = body_dv[:, :, :2]

        delta_mu_h = torch.zeros_like(mu_h)
        delta_mu_b = torch.zeros_like(mu_b)

        delta_mu_h[:, 1:] = mu_h[:, 1:] - mu_h[:, :-1]
        delta_mu_b[:, 1:] = mu_b[:, 1:] - mu_b[:, :-1]

        state = torch.cat([mu_h,mu_b,v_t,delta_mu_h,delta_mu_b], dim=-1)
        temporal_feature = self.temporal_mamba(state)

        gaze = self.gaze_head(temporal_feature)

        return {
            "gaze_direction": gaze,
            "head_direction": mu_h,
            "body_direction": mu_b,
            "head_feat": hbonet_out["head_feat"],
            "body_feat": hbonet_out["body_feat"]
        }

    def training_step(self, batch, batch_idx):
        image, head_mask, body_dv, gt_gaze, head_gt, body_gt, fb_label = batch
        output = self(image, head_mask, body_dv)
        gaze_dir = output['gaze_direction']
        head_dir = output['head_direction']
        body_dir = output['body_direction']

        loss_head = compute_basic_cos_loss({'direction': head_dir}, head_gt)
        loss_body = compute_basic_cos_loss({'direction': body_dir}, body_gt)

        is_back = fb_label.float()  # [B, T]
        cos_gaze = F.cosine_similarity(
            gaze_dir, gt_gaze, dim=-1
        )  # [B, T]


        loss_per_frame = 1 - cos_gaze  # [B, T]
        weight = 1.0 + is_back  # front=1, back=2
        loss_gaze = (weight * loss_per_frame).mean()

        loss_task = 0.35 * loss_head + 0.35 * loss_body + 0.30 * loss_gaze

        head_feat = output['head_feat']
        body_feat = output['body_feat']
        loss_align = compute_mmd_loss(head_feat, body_feat)
        total_loss = loss_task+0.005*loss_align

        # ---------- 手动优化 ----------
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.parameters()),
            max_norm=1.0
        )
        opt.step()
        # ------------------------------

        self.log_dict({"loss_task": loss_task, "loss_align": loss_align, "train_loss": total_loss}, prog_bar=True)
        mae = compute_mae(gaze_dir, gt_gaze)
        self.log('train_mae', mae)

        return total_loss

    def on_train_epoch_end(self):
        # 打印当前学习率
        lr = self.optimizers().param_groups[0]['lr']
        print(f"[LR] epoch={self.current_epoch} | lr={lr:.8f}")



    def validation_step(self, batch, batch_idx):
        image, head_mask, body_dv, gaze_label, head_gt, body_gt, fb_label = batch
        output = self(image, head_mask, body_dv)
        gaze_dir = output['gaze_direction']

        mae = compute_mae(gaze_dir, gaze_label)
        self.validation_step_outputs.append(mae.detach())
        return mae

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) > 0:
            val_mae = torch.stack(self.validation_step_outputs).mean()
            self.log('val_mae', val_mae, prog_bar=True, logger=True)
            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        image, head_mask, body_dv, gaze_label, head_gt, body_gt, fb_label = batch

        output = self(
            image,
            head_mask,
            body_dv
        )

        prediction = output['gaze_direction']
        head_pred = output['head_direction']
        body_pred = output['body_direction']

        # Head / Body MAE
        head_mae = compute_mae(
            head_pred,
            head_gt
        )

        body_mae = compute_mae(
            body_pred,
            body_gt
        )

        mae = mae_2d = None
        front_mae = front_mae_2d = None
        back_mae = back_mae_2d = None

        if gaze_label.shape[-1] == 3:

            # ==================================================
            # 与 baseline 完全一致：
            # 一个 sequence 只根据第 0 帧的 gaze z 判断 front/back
            # ==================================================
            front_index = torch.arange(
                gaze_label.shape[0],
                device=gaze_label.device
            )[gaze_label[:, 0, -1] <= 0]

            back_index = torch.arange(
                gaze_label.shape[0],
                device=gaze_label.device
            )[gaze_label[:, 0, -1] > 0]

            # ==================================================
            # 3D MAE
            # ==================================================
            mae = compute_mae(
                prediction,
                gaze_label
            )

            if len(front_index) > 0:
                front_mae = compute_mae(
                    prediction[front_index],
                    gaze_label[front_index]
                )

            if len(back_index) > 0:
                back_mae = compute_mae(
                    prediction[back_index],
                    gaze_label[back_index]
                )

            # ==================================================
            # 2D MAE
            # ==================================================
            gaze_label_2d = (
                    gaze_label[..., :2] /
                    torch.norm(
                        gaze_label[..., :2],
                        dim=-1,
                        keepdim=True
                    )
            )

            prediction_2d = (
                    prediction[..., :2] /
                    torch.norm(
                        prediction[..., :2],
                        dim=-1,
                        keepdim=True
                    )
            )

            mae_2d = compute_mae(
                prediction_2d,
                gaze_label_2d
            )

            if len(front_index) > 0:
                front_mae_2d = compute_mae(
                    prediction_2d[front_index],
                    gaze_label_2d[front_index]
                )

            if len(back_index) > 0:
                back_mae_2d = compute_mae(
                    prediction_2d[back_index],
                    gaze_label_2d[back_index]
                )

            front_count = len(front_index)
            back_count = len(back_index)

        elif gaze_label.shape[-1] == 2:

            front_count = 0
            back_count = 0

            gaze_label_2d = (
                    gaze_label[..., :2] /
                    torch.norm(
                        gaze_label[..., :2],
                        dim=-1,
                        keepdim=True
                    )
            )

            prediction_2d = (
                    prediction[..., :2] /
                    torch.norm(
                        prediction[..., :2],
                        dim=-1,
                        keepdim=True
                    )
            )

            mae_2d = compute_mae(
                prediction_2d,
                gaze_label_2d
            )

        result = {
            'mae': mae,
            'mae_2d': mae_2d,
            'front_mae': front_mae,
            'front_mae_2d': front_mae_2d,
            'back_mae': back_mae,
            'back_mae_2d': back_mae_2d,
            'head_mae': head_mae,
            'body_mae': body_mae,
            'front_count': front_count,
            'back_count': back_count,
        }

        self.test_step_outputs.append(result)

        return result
    def on_test_epoch_end(self):

        def to_float(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.item()
            return float(x)

        def safe_mean(values):
            values = [
                to_float(x)
                for x in values
                if x is not None
            ]
            return np.nanmean(values) if len(values) > 0 else np.nan

        outputs = self.test_step_outputs

        mae = safe_mean([x['mae'] for x in outputs])
        mae_2d = safe_mean([x['mae_2d'] for x in outputs])

        front_mae = safe_mean([x['front_mae'] for x in outputs])
        front_mae_2d = safe_mean([x['front_mae_2d'] for x in outputs])

        back_mae = safe_mean([x['back_mae'] for x in outputs])
        back_mae_2d = safe_mean([x['back_mae_2d'] for x in outputs])

        head_mae = safe_mean([x['head_mae'] for x in outputs])
        body_mae = safe_mean([x['body_mae'] for x in outputs])

        total_front = sum(
            x['front_count']
            for x in outputs
        )

        total_back = sum(
            x['back_count']
            for x in outputs
        )

        print('MAE (3D front): ', front_mae)
        print('MAE (2D front): ', front_mae_2d)
        print('MAE (3D back): ', back_mae)
        print('MAE (2D back): ', back_mae_2d)
        print('MAE (3D all): ', mae)
        print('MAE (2D all): ', mae_2d)
        print('Head MAE (3D): ', head_mae)
        print('Body MAE (3D): ', body_mae)

        print(f'Front count: {total_front}')
        print(f'Back count: {total_back}')

        # 保存最终测试结果，供 eval.py 读取
        self.final_test_results = {
            'MAE (3D front)': front_mae,
            'MAE (2D front)': front_mae_2d,
            'MAE (3D back)': back_mae,
            'MAE (2D back)': back_mae_2d,
            'MAE (3D all)': mae,
            'MAE (2D all)': mae_2d,
            'Head MAE (3D)': head_mae,
            'Body MAE (3D)': body_mae,
            'Front count': total_front,
            'Back count': total_back,
        }

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [opt]  # 返回列表