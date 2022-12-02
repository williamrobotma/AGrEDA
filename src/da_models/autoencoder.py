"""Autoencoder."""
from torch import nn

from .components import ADDAMLPEncoder, ADDAMLPDecoder

class AutoEncoder(nn.Module):
    """Autoencoder model for pseudo-spots.
    Args:
        inp_dim (int): Number of gene expression features.
        emb_dim (int): Embedding size.

    """

    def __init__(self, *encoder_args, **encoder_kwargs):
        super().__init__()

        self.encoder = ADDAMLPEncoder(*encoder_args, **encoder_kwargs)
        self.decoder = ADDAMLPDecoder(*encoder_args, **encoder_kwargs)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
