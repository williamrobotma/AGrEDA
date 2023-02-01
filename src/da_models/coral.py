"""CORAL model."""

from torch import nn

from .components import MLP

ENC_HIDDEN_LAYER_SIZES = (
    1024,
    512,
)

PREDICTOR_HIDDEN_LAYER_SIZES = None


class CORAL(nn.Module):
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
        # emb_dim,
        ncls_source,
        enc_hidden_layer_sizes=ENC_HIDDEN_LAYER_SIZES,
        # enc_out_act="elu",
        # predictor_hidden_layer_sizes=PREDICTOR_HIDDEN_LAYER_SIZES,
        **kwargs
    ):
        super().__init__()
        self.encoder = MLP(
            inp_dim,
            ncls_source,
            hidden_layer_sizes=enc_hidden_layer_sizes,
            output_act=None,
            **kwargs
        )
        self.source_encoder = self.target_encoder = self.encoder

        # self.clf = MLP(
        #     emb_dim,
        #     ncls_source,
        #     hidden_layer_sizes=predictor_hidden_layer_sizes,
        #     output_act=None,
        #     **kwargs
        # )

        self.output_act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.clf(x)

        output = self.output_act(x)

        return output, x

    def pretraining(self):
        pass

    def advtraining(self):
        pass

    def target_inference(self):
        pass

    def set_encoder(self, encoder="source"):
        pass
