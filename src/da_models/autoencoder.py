"""Autoencoder."""
from copy import copy
from torch import nn

from .components import ADDAMLPEncoder, ADDAMLPDecoder

class AutoEncoder(nn.Module):
    """Autoencoder model for pseudo-spots.
    Args:
        inp_dim (int): Number of gene expression features.
        emb_dim (int): Embedding size.
        dropout (float): Dropout rate.
        enc_out_act (nn.Module): Activation function for encoder output.
            Default: nn.ELU()
        dec_out_act (nn.Module): Activation function for decoder output.
            Default: nn.Sigmoid()

    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        encoder_kwargs = copy(kwargs)
        decoder_kwargs = copy(kwargs)

        encoder_kwargs.pop("dec_out_act", None)
        decoder_kwargs.pop("enc_out_act", None)

        self.encoder = ADDAMLPEncoder(*args, **encoder_kwargs)
        self.decoder = ADDAMLPDecoder(*args, **decoder_kwargs)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
