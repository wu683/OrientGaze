import torch
from torch import nn
# ---------- Mamba 导入 ----------
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("mamba_ssm not found, please install it")

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