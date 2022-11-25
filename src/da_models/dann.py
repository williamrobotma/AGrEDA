"""DANN model."""
import warnings

from torch import nn, tensor
from torch.autograd import Function

from .components import (
    MLPEncoder,
    Predictor,
    Discriminator,
    ADDAMLPEncoder,
    AddaDiscriminator,
    AddaPredictor,
)
from .utils import set_requires_grad


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, lambd):
        ctx.save_for_backward(input_, lambd)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, lambd = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output / lambd
        return grad_input, None


def grad_reverse(x, lambd):
    lambd = tensor(lambd, requires_grad=False)
    return RevGrad.apply(x, lambd)


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

    def __init__(self, inp_dim, emb_dim, ncls_source, lambd=1.0):
        super().__init__()

        self.encoder = ADDAMLPEncoder(inp_dim, emb_dim)
        self.dis = AddaDiscriminator(emb_dim)
        self.clf = AddaPredictor(emb_dim, ncls_source)
        self.lambd = lambd

        self.is_pretraining = False

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dis=True, clf=True):
        x = self.encoder(x)

        if clf or self.is_pretraining:
            x_clf = self.clf(x)
        else:
            x_clf = None
        if dis or not self.is_pretraining:
            x_dis = grad_reverse(x, self.lambd)
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
