"""DANN model."""
import warnings

from torch import nn, tensor
from torch.autograd import Function

import torch.nn.functional as F

from .utils import set_requires_grad


class RevGrad(Function):
    """Gradient Reversal layer."""

    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output / alpha_
        return grad_input, None


def grad_reverse(x, alpha_):
    alpha_ = tensor(alpha_, requires_grad=False)
    return RevGrad.apply(x, alpha_)


class DANN(nn.Module):
    """ADDA model for spatial transcriptomics.

    Args:
        inp_dim (int): Number of gene expression features.
        emb_dim (int): Embedding size.
        ncls_source (int): Number of cell types.

    Attributes:
        is_encoder_source (bool): Whether source encoder is used for forward
            pass; else use target encoder

    """

    def __init__(self, inp_dim, emb_dim, ncls_source, alpha_=1.0):
        super().__init__()

        self.encoder = DannMLPEncoder(inp_dim, emb_dim)
        self.source_encoder = self.target_encoder = self.encoder
        self.dis = DannDiscriminator(emb_dim)
        self.clf = DannPredictor(emb_dim, ncls_source)
        self.alpha_ = alpha_

        self.is_pretraining = False

    def set_lambda(self, alpha_):
        self.alpha_ = alpha_

    def forward(self, x, dis=True, clf=True):
        x = self.encoder(x)

        if clf or self.is_pretraining:
            x_clf = self.clf(x)
        else:
            x_clf = None
        if dis or not self.is_pretraining:
            x_dis = grad_reverse(x, self.alpha_)
            x_dis = self.dis(x_dis)

        return x_clf, x_dis

    def pretraining(self):
        """Enable pretraining mode to train model on source domain."""
        set_requires_grad(self.encoder, True)
        set_requires_grad(self.clf, True)
        set_requires_grad(self.dis, False)

        self.is_pretraining = True

    def advtraining(self):
        """Enable adversarial training mode to train."""
        set_requires_grad(self.encoder, True)
        set_requires_grad(self.clf, True)
        set_requires_grad(self.dis, True)

        self.is_pretraining = False

    def target_inference(self):
        """Enable target inference mode."""
        self.pretraining()

    def set_encoder(self, encoder="source"):
        pass


class DannMLPEncoder(nn.Module):
    """MLP embedding encoder for gene expression data.

    Args:
        inp_dim (int): Number of gene expression features.
        emb_dim (int): Embedding size.

    """

    def __init__(self, inp_dim, emb_dim, dropout=0.5):
        super().__init__()

        self.encoder = nn.Sequential(
            # nn.BatchNorm1d(inp_dim, eps=0.001, momentum=0.99),
            # nn.Dropout(0.5),
            nn.Linear(inp_dim, 1024),
            # nn.BatchNorm1d(1024, eps=0.001, momentum=0.99),
            nn.LeakyReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(1024, 512),
            # nn.BatchNorm1d(512, eps=0.001, momentum=0.99),
            # nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, emb_dim),
            # nn.BatchNorm1d(emb_dim, eps=0.001, momentum=0.99),
            nn.ELU(),
        )

    def forward(self, x):
        return self.encoder(x)


class DannPredictor(nn.Module):
    """Predicts cell type proportions from embeddings.

    Args:
        emb_dim (int): Embedding size.
        ncls_source (int): Number of cell types.

    """

    def __init__(self, emb_dim, ncls_source):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            # nn.BatchNorm1d(32, eps=0.001, momentum=0.99),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, ncls_source),
            # nn.LogSoftmax(dim=1),
            # F.nor
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.head(x)
        # x = (x - torch.min(x, dim=1, keepdim=True)[0]) / (
        #     torch.max(x, dim=1)[0] - torch.min(x, dim=1)[0]
        # ).view(x.shape[0], -1)
        # print(x)
        # x = torch.log(x)
        x = F.log_softmax(x, dim=1)
        return x


class DannDiscriminator(nn.Module):
    """Classifies domain of embedding.

    Args:
        emb_dim (int): Embedding size.

    """

    def __init__(self, emb_dim):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            # nn.BatchNorm1d(512, eps=0.001, momentum=0.99),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            # nn.Linear(512, 1024),
            # nn.BatchNorm1d(1024, eps=0.001, momentum=0.99),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.head(x)
