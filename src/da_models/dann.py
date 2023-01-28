"""DANN model."""
import warnings

from torch import nn, tensor
from torch.autograd import Function

import torch.nn.functional as F

from .utils import set_requires_grad
from .components import MLP

ENC_HIDDEN_LAYER_SIZES = (
    1024,
    512,
)

PREDICTOR_HIDDEN_LAYER_SIZES = (512,)
DIS_HIDDEN_LAYER_SIZES = (512,)


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
        alpha_ (float): Value to divide encoder gradient after gradient
            reversal. Default: 1.0
        dropout (float): Dropout rate.
        batchnorm (bool): Whether to use batch normalization. Default: False.
        bn_momentum (float): Batch normalization momentum. Default: 0.1

    Attributes:
        is_encoder_source (bool): Whether source encoder is used for forward
            pass; else use target encoder

    """

    def __init__(
        self,
        inp_dim,
        emb_dim,
        ncls_source,
        enc_hidden_layer_sizes=ENC_HIDDEN_LAYER_SIZES,
        enc_out_act="elu",
        predictor_hidden_layer_sizes=PREDICTOR_HIDDEN_LAYER_SIZES,
        dis_hidden_layer_sizes=DIS_HIDDEN_LAYER_SIZES,
        dropout=0.5,
        dis_dropout_factor=1,
        batchnorm=False,
        bn_momentum=0.1,
        alpha_=1.0,
        **kwargs
    ):
        super().__init__()

        # self.encoder = DannMLPEncoder(inp_dim, emb_dim, **kwargs)
        # self.source_encoder = self.target_encoder = self.encoder
        # self.dis = DannDiscriminator(emb_dim, **kwargs)
        # self.clf = DannPredictor(emb_dim, ncls_source, **kwargs)
        common_kwargs = dict(
            batchnorm=batchnorm, bn_kwargs={"momentum": bn_momentum}, **kwargs
        )
        self.encoder = MLP(
            inp_dim,
            emb_dim,
            hidden_layer_sizes=enc_hidden_layer_sizes,
            dropout=dropout,
            output_act=enc_out_act,
            **common_kwargs
        )
        self.source_encoder = self.target_encoder = self.encoder
        self.dis = MLP(
            emb_dim,
            1,
            hidden_layer_sizes=dis_hidden_layer_sizes,
            dropout=dropout * dis_dropout_factor,
            **common_kwargs
        )
        self.clf = MLP(
            emb_dim,
            ncls_source,
            hidden_layer_sizes=predictor_hidden_layer_sizes,
            dropout=dropout,
            output_act=nn.LogSoftmax(dim=1),
            **common_kwargs
        )

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


def get_act_from_str(act):
    """Get activation function module from string.

    If act is not a string, or doesn't match, return act.

    Args:
        act (str or nn.Module): Activation function.

    Returns:
        nn.Module: Activation function module or act if not a string.

    """
    if isinstance(act, str):
        act = act.lower()
        if act == "elu":
            return nn.ELU()
        if act == "leakyrelu":
            return nn.LeakyReLU()
        if act == "relu":
            return nn.ReLU()
        if act == "tanh":
            return nn.Tanh()
        if act == "sigmoid":
            return nn.Sigmoid()

    return act


class DannMLPEncoder(nn.Module):
    """MLP embedding encoder for gene expression data.

    Args:
        inp_dim (int): Number of gene expression features.
        emb_dim (int): Embedding size.
        dropout (float): Dropout rate.
        batchnorm (bool): Whether to use batch normalization. Default: False.
        bn_momentum (float): Batch normalization momentum. Default: 0.1

    """

    def __init__(
        self,
        inp_dim,
        emb_dim,
        enc_hidden_layer_sizes=ENC_HIDDEN_LAYER_SIZES,
        dropout=0.5,
        batchnorm=False,
        bn_momentum=0.1,
        enc_out_act="elu",
        **kwargs
    ):
        super().__init__()

        layers = []
        for i, h in enumerate(enc_hidden_layer_sizes):
            layers.append(
                nn.Linear(inp_dim if i == 0 else enc_hidden_layer_sizes[i - 1], h)
            )
            if batchnorm:
                layers.append(nn.BatchNorm1d(h, momentum=bn_momentum))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(enc_hidden_layer_sizes[-1], emb_dim))

        if enc_out_act:
            enc_out_act = get_act_from_str(enc_out_act)

            layers.append(enc_out_act)

        # layers.append(nn.ELU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class DannPredictor(nn.Module):
    """Predicts cell type proportions from embeddings.

    Args:
        emb_dim (int): Embedding size.
        ncls_source (int): Number of cell types.
        dropout (float): Dropout rate.
        batchnorm (bool): Whether to use batch normalization. Default: False.
        bn_momentum (float): Batch normalization momentum. Default: 0.1

    """

    def __init__(
        self,
        emb_dim,
        ncls_source,
        predictor_hidden_layer_sizes=PREDICTOR_HIDDEN_LAYER_SIZES,
        dropout=0.5,
        batchnorm=False,
        bn_momentum=0.1,
        **kwargs
    ):
        super().__init__()

        layers = []
        for i, h in enumerate(predictor_hidden_layer_sizes):
            layers.append(
                nn.Linear(emb_dim if i == 0 else predictor_hidden_layer_sizes[i - 1], h)
            )
            if batchnorm:
                layers.append(nn.BatchNorm1d(h, momentum=bn_momentum))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(predictor_hidden_layer_sizes[-1], ncls_source))

        self.head = nn.Sequential(*layers)

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

    def __init__(
        self,
        emb_dim,
        dis_hidden_layer_sizes=DIS_HIDDEN_LAYER_SIZES,
        dropout=0.5,
        dis_dropout_factor=1,
        batchnorm=False,
        bn_momentum=0.1,
        **kwargs
    ):
        super().__init__()

        layers = []
        for i, h in enumerate(dis_hidden_layer_sizes):
            layers.append(
                nn.Linear(emb_dim if i == 0 else dis_hidden_layer_sizes[i - 1], h)
            )
            if batchnorm:
                layers.append(nn.BatchNorm1d(h, momentum=bn_momentum))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout * dis_dropout_factor))

        layers.append(nn.Linear(dis_hidden_layer_sizes[-1], 1))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)
