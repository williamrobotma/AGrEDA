"""Losses"""
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


def coral_loss(source, target):
    """Compute CORAL loss.

    Args:
        source (torch.tensor): Source domain features.
        target (torch.tensor): Target domain features.

    Returns:
        torch.tensor: CORAL loss.

    """
    c_diff = torch.cov(source.T) - torch.cov(target.T)
    loss = torch.sum(torch.mul(c_diff, c_diff))
    return loss / (4 * source.size(1) ** 2)
