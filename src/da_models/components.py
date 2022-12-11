"""Building blocks for models."""
from torch import nn
import torch
import torch.nn.functional as F


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
        self, inp_dim, emb_dim, dropout=0.5, enc_out_act=nn.ELU(), bn_momentum=0.99
    ):
        super().__init__()

        layers = [
            # nn.BatchNorm1d(inp_dim, eps=0.001, momentum=bn_momentum),
            # nn.Dropout(0.5),
            nn.Linear(inp_dim, 1024),
            nn.BatchNorm1d(1024, eps=0.001, momentum=bn_momentum),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, eps=0.001, momentum=bn_momentum),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, emb_dim),
            # nn.BatchNorm1d(emb_dim, eps=0.001, momentum=bn_momentum),
        ]
        if enc_out_act:
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
        self, inp_dim, emb_dim, dropout=0.5, dec_out_act=nn.Sigmoid(), bn_momentum=0.99
    ):
        super().__init__()

        layers = [
            nn.Linear(emb_dim, 512),
            nn.BatchNorm1d(512, eps=0.001, momentum=bn_momentum),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, eps=0.001, momentum=bn_momentum),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, inp_dim),
        ]
        if dec_out_act:
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
