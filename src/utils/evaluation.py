"""Evaluation tools"""
import torch
from torch import nn

class JSD(nn.Module):
    """Jensen-Shannon Divergence"""

    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p) + self.kl(m, q))
        