"""CORAL model."""

from collections import OrderedDict
from torch import nn

from src.da_models.components import MLP, get_act_from_str

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
        emb_dim,
        ncls_source,
        enc_hidden_layer_sizes=ENC_HIDDEN_LAYER_SIZES,
        enc_out_act=False,
        use_predictor=False,
        predictor_hidden_layer_sizes=PREDICTOR_HIDDEN_LAYER_SIZES,
        bn_momentum=0.1,
        **kwargs
    ):
        super().__init__()

        common_kwargs = dict(bn_kwargs={"momentum": bn_momentum}, **kwargs)

        if not emb_dim:
            emb_dim = ncls_source
        _encoder = MLP(
            inp_dim,
            emb_dim,
            hidden_layer_sizes=enc_hidden_layer_sizes,
            output_act=None,
            **common_kwargs
        )


        if enc_out_act:
            _encoder_output_act = get_act_from_str(kwargs.get("hidden_act"))
        else:
            _encoder_output_act = nn.Identity()

        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ("encoder_bare", _encoder),
                    ("output_act", _encoder_output_act),
                ]
            )
        )
        self.source_encoder = self.target_encoder = self.encoder

        if use_predictor:
            self.clf = MLP(
                emb_dim,
                ncls_source,
                hidden_layer_sizes=predictor_hidden_layer_sizes,
                output_act=None,
                **kwargs
            )
        else:
            self.clf = nn.Identity()

        self.output_act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        xl = []

        xl.append(self.encoder.encoder_bare(x))
        x = self.encoder.output_act(xl[-1])

        xl.append(self.clf(x))
        x = self.output_act(xl[-1])

        return x, xl

    def pretraining(self):
        pass

    def advtraining(self):
        pass

    def target_inference(self):
        pass

    def set_encoder(self, encoder="source"):
        pass
