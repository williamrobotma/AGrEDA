"""Building blocks for models."""
from collections.abc import Iterable
from copy import deepcopy

from torch import nn
import torch
import torch.nn.functional as F

ADDA_ENC_HIDDEN_LAYER_SIZES = (
    1024,
    512,
)


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


def iterify_act(act):
    if isinstance(act, Iterable) and not isinstance(act, str):
        for a in act:
            yield get_act_from_str(a)
    else:
        base_act = get_act_from_str(act)
        while True:
            yield deepcopy(base_act)


class MLP(nn.Module):
    """MLP embedding encoder for gene expression data.

    Args:
        inp_dim (int): Number of gene expression features.
        emb_dim (int): Embedding size.

    """

    def __init__(
        self,
        inp_dim,
        out_dim,
        hidden_layer_sizes=None,
        dropout=None,
        batchnorm=True,
        batchnorm_output=False,
        bn_kwargs=None,
        hidden_act="leakyrelu",
        output_act=None,
    ):
        super().__init__()

        if bn_kwargs is None:
            bn_kwargs = {}

        layers = []
        if not hidden_layer_sizes:
            layers.append(nn.Linear(inp_dim, out_dim))
        else:
            act_gen = iterify_act(hidden_act)
            for i, h in enumerate(hidden_layer_sizes):
                layers.append(
                    nn.Linear(inp_dim if i == 0 else hidden_layer_sizes[i - 1], h)
                )
                if batchnorm:
                    layers.append(nn.BatchNorm1d(h, **bn_kwargs))
                next_act = next(act_gen)
                if next_act:
                    layers.append(next_act)
                if dropout or dropout == 0:
                    layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_layer_sizes[-1], out_dim))

        if batchnorm_output:
            layers.append(nn.BatchNorm1d(out_dim, **bn_kwargs))
        if output_act:
            output_act = get_act_from_str(output_act)
            layers.append(output_act)

        # layers.append(nn.ELU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class MLPEncoder(nn.Module):
    """MLP embedding encoder for gene expression data.

    Args:
        inp_dim (int): Number of gene expression features.
        emb_dim (int): Embedding size.

    """

    def __init__(self, inp_dim, emb_dim, bn_momentum=0.99):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, 1024),
            nn.BatchNorm1d(1024, eps=0.001, momentum=bn_momentum),
            nn.ELU(),
            nn.Linear(1024, emb_dim),
            nn.BatchNorm1d(emb_dim, eps=0.001, momentum=bn_momentum),
            nn.ELU(),
        )

    def forward(self, x):
        return self.encoder(x)


class ADDAMLPEncoder(nn.Module):
    """MLP embedding encoder for gene expression data.

    Args:
        inp_dim (int): Number of gene expression features.
        emb_dim (int): Embedding size.
        dropout (float): Dropout rate.
        enc_out_act (nn.Module): Activation function for encoder output.
            Default: nn.ELU()

    """

    def __init__(
        self,
        inp_dim,
        emb_dim,
        hidden_layer_sizes=ADDA_ENC_HIDDEN_LAYER_SIZES,
        dropout=0.5,
        enc_out_act="elu",
        bn_momentum=0.99,
    ):
        super().__init__()

        layers = []
        for i, h in enumerate(hidden_layer_sizes):
            layers.append(
                nn.Linear(inp_dim if i == 0 else hidden_layer_sizes[i - 1], h)
            )
            layers.append(nn.BatchNorm1d(h, eps=0.001, momentum=bn_momentum))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_layer_sizes[-1], emb_dim))

        if enc_out_act:
            if enc_out_act == "elu":
                enc_out_act = nn.ELU()
            elif enc_out_act == "relu":
                enc_out_act = nn.ReLU()
            elif enc_out_act == "tanh":
                enc_out_act = nn.Tanh()
            elif enc_out_act == "sigmoid":
                enc_out_act = nn.Sigmoid()

            layers.append(enc_out_act)
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class ADDAMLPDecoder(nn.Module):
    """MLP embedding decoder for gene expression data.

    Args:
        emb_dim (int): Embedding size.
        inp_dim (int): Number of gene expression features.
        dropout (float): Dropout rate.
        dec_out_act (nn.Module): Activation function for decoder output.
            Default: nn.Sigmoid()

    """

    def __init__(
        self,
        inp_dim,
        emb_dim,
        hidden_layer_sizes=ADDA_ENC_HIDDEN_LAYER_SIZES,
        dropout=0.5,
        dec_out_act="sigmoid",
        bn_momentum=0.99,
    ):
        super().__init__()

        layers = []
        for i, h in enumerate(reversed(hidden_layer_sizes)):
            layers.append(nn.Linear(emb_dim if i == 0 else hidden_layer_sizes[-i], h))
            layers.append(nn.BatchNorm1d(h, eps=0.001, momentum=bn_momentum))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_layer_sizes[0], inp_dim))

        if dec_out_act:
            if dec_out_act == "elu":
                dec_out_act = nn.ELU()
            elif dec_out_act == "relu":
                dec_out_act = nn.ReLU()
            elif dec_out_act == "tanh":
                dec_out_act = nn.Tanh()
            elif dec_out_act == "sigmoid":
                dec_out_act = nn.Sigmoid()
            layers.append(dec_out_act)

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Predictor(nn.Module):
    """Predicts cell type proportions from embeddings.

    Args:
        emb_dim (int): Embedding size.
        ncls_source (int): Number of cell types.

    """

    def __init__(self, emb_dim, ncls_source):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(emb_dim, ncls_source),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.head(x)


class AddaPredictor(nn.Module):
    """Predicts cell type proportions from embeddings.

    Args:
        emb_dim (int): Embedding size.
        ncls_source (int): Number of cell types.

    """

    def __init__(self, emb_dim, ncls_source):
        super().__init__()

        self.head = nn.Sequential(
            # nn.Linear(emb_dim, 32),
            # nn.BatchNorm1d(32, eps=0.001, momentum=0.99),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(emb_dim, ncls_source),
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


class Discriminator(nn.Module):
    """Classifies domain of embedding.

    Args:
        emb_dim (int): Embedding size.

    """

    def __init__(self, emb_dim, bn_momentum=0.99):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.BatchNorm1d(32, eps=0.001, momentum=bn_momentum),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.head(x)


class AddaDiscriminator(nn.Module):
    """Classifies domain of embedding.

    Args:
        emb_dim (int): Embedding size.

    """

    def __init__(self, emb_dim, bn_momentum=0.99):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.BatchNorm1d(512, eps=0.001, momentum=bn_momentum),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, eps=0.001, momentum=bn_momentum),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.head(x)
