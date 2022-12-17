"""Evaluation tools"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
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


def format_iters(nested_list, startpoint=False, endpoint=True):
    """Generates x and y values, given a nested list of iterations by epoch.


    x will be evenly spaced by epoch, and y will be the flattened values in the
    nested list.

    Args:
        nested_list (list): List of lists.
        startpoint (bool): Include startpoint of iteration. Defaults to False.
        endpoint (bool): Include endpoint of iteration. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y values.


    """

    x = []
    if startpoint:
        for i, l in enumerate(nested_list):
            if endpoint and i == len(nested_list) - 1:
                x_i = np.linspace(i - 1, i, len(l), endpoint=True, dtype=np.float32)
            else:
                x_i = np.linspace(i - 1, i, len(l), endpoint=False, dtype=np.float32)
            x.append(x_i)
    else:
        for i, l in enumerate(nested_list):
            if not endpoint and i == len(nested_list) - 1:
                x_i = np.linspace(
                    i, i - 1, len(l + 1), endpoint=False, dtype=np.float32
                )
                x_i = x_i[1:]
            else:
                x_i = np.linspace(i, i - 1, len(l), endpoint=False, dtype=np.float32)

            # Flip to not include startpoint i.e. shift to end of iteration
            x_i = np.flip(x_i)
            x.append(x_i)

    x = np.asarray(list(itertools.chain(*x)))
    y = np.asarray(list(itertools.chain(*nested_list)))

    return x, y
