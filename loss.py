import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_basic_cos_loss(outputs, true_dir):
    """
    Compute integrated loss function of spherical regression
    """

    reg_dir = outputs['direction']

    reg_dir = reg_dir.reshape(-1, reg_dir.shape[-1])
    true_dir = true_dir.reshape(-1, true_dir.shape[-1])

    if true_dir.shape[-1] == 2:
        reg_dir = reg_dir[..., :2] / torch.norm(reg_dir[..., :2], dim=-1, keepdim=True)

    cos = torch.sum(reg_dir * true_dir, dim=-1)
    cos[cos > 1] = 1
    cos[cos < -1] = -1
    loss = 1 - cos

    return loss.mean()

def compute_mmd_loss(feat_a, feat_b, sigma=1.0):
    """
    标准高斯核最大均值差异，计算两组特征的分布距离
    Args:
        feat_a: (N, D) 第一模态特征
        feat_b: (M, D) 第二模态特征
        sigma: 高斯核带宽
    Returns:
        mmd: 标量损失值，非负
    """
    # 1. 先做L2归一化，压缩数值范围，从根源提升计算稳定性
    feat_a = F.normalize(feat_a.reshape(-1, feat_a.shape[-1]), dim=-1)
    feat_b = F.normalize(feat_b.reshape(-1, feat_b.shape[-1]), dim=-1)

    # 2. 计算内积矩阵
    aa = torch.mm(feat_a, feat_a.t())
    bb = torch.mm(feat_b, feat_b.t())
    ab = torch.mm(feat_a, feat_b.t())

    # 3. 正确计算各距离平方矩阵
    a_norm = aa.diag().unsqueeze(1)  # (N, 1) 每个a样本的模长平方
    b_norm = bb.diag().unsqueeze(0)  # (1, M) 每个b样本的模长平方

    dist_aa = a_norm + a_norm.t() - 2 * aa
    dist_bb = b_norm + b_norm.t() - 2 * bb
    dist_ab = a_norm + b_norm - 2 * ab  # 交叉项必须用a、b各自的模长

    # 4. 高斯核计算
    def gaussian_kernel(dist):
        return torch.exp(-dist / (2 * sigma ** 2))

    k_aa = gaussian_kernel(dist_aa)
    k_bb = gaussian_kernel(dist_bb)
    k_ab = gaussian_kernel(dist_ab)

    # 5. 最终MMD，加非负截断屏蔽数值误差
    mmd = k_aa.mean() + k_bb.mean() - 2 * k_ab.mean()
    return torch.clamp(mmd, min=0.0)

class TemporalConsistency3DLoss(nn.Module):
    def __init__(self, tau=0.05, beta=20.0):
        super().__init__()
        self.tau = tau
        self.beta

    def forward(self, pred_mid, gt_prev, gt_next, head_center_mid, head_center_prev, head_center_next):
        sim_prev = (pred_mid * gt_prev).sum(dim=-1)
        sim_next = (pred_mid * gt_next).sum(dim=-1)
        loss_prev = 1 - sim_prev
        loss_next = 1 - sim_next

        delta_prev = torch.norm(head_center_mid - head_center_prev, dim=-1)
        delta_next = torch.norm(head_center_next - head_center_next, dim=-1)

        w_prev = torch.sigmoid(-self.beta * (delta_prev - self.tau))
        w_next = torch.sigmoid(-self.beta * (delta_next - self.tau))

        # 已修复：w_next 乘以 loss_next
        return (w_prev * loss_prev + w_next * loss_next).mean()

