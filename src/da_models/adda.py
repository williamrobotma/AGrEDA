"""ADDA model."""
import warnings

from torch import nn

from .components import (
    MLP,
    MLPEncoder,
    Predictor,
    Discriminator,
    ADDAMLPEncoder,
    AddaDiscriminator,
    AddaPredictor,
)
from .utils import set_requires_grad

ADDA_ENC_HIDDEN_LAYER_SIZES = (
    1024,
    512,
)


class ADDAST(nn.Module):
    """ADDA model for spatial transcriptomics.

    Args:
        inp_dim (int): Number of gene expression features.
        emb_dim (int): Embedding size.
        ncls_source (int): Number of cell types.

    Attributes:
        is_encoder_source (bool): Whether source encoder is used for forward
            pass; else use target encoder

    """

    def __init__(self, inp_dim, emb_dim, ncls_source, bn_momentum=0.99, is_adda=False):
        super().__init__()

        clf_args = (emb_dim, ncls_source)
        clf_kwargs = dict(hidden_layer_sizes=None, output_act=nn.LogSoftmax(dim=1))
        if is_adda:
            enc_args = (inp_dim, emb_dim)
            enc_kwargs = dict(
                hidden_layer_sizes=ADDA_ENC_HIDDEN_LAYER_SIZES,
                dropout=0.5,
                bn_kwargs={"eps": 0.001, "momentum": bn_momentum},
                output_act="elu",
            )
            self.enc_args_kwargs = [enc_args, enc_kwargs]
            self.source_encoder = MLP(*enc_args, **enc_kwargs)
            self.clf = MLP(*clf_args, **clf_kwargs)
        else:
            enc_args = (inp_dim, emb_dim)
            enc_kwargs = dict(
                hidden_layer_sizes=(1024,),
                dropout=None,
                batchnorm_output=True,
                bn_kwargs={"eps": 0.001, "momentum": bn_momentum},
                hidden_act="elu",
                output_act="elu",
            )
            self.source_encoder = MLP(*enc_args, **enc_kwargs)
            self.target_encoder = MLP(*enc_args, **enc_kwargs)
            self.dis = MLP(
                inp_dim,
                2,
                hidden_layer_sizes=(32,),
                dropout=0.5,
                bn_kwargs={"eps": 0.001, "momentum": bn_momentum},
                hidden_act="elu",
            )

            self.clf = MLP(*clf_args, **clf_kwargs)

        self.is_encoder_source = True
        # self.inp_dim = inp_dim
        # self.emb_dim = emb_dim
        # self.ncls_source = ncls_source
        # self.bn_momentum = bn_momentum

    def forward(self, x):
        if self.is_encoder_source:
            x = self.source_encoder(x)
        else:
            x = self.target_encoder(x)

        x = self.clf(x)

        return x

    def init_adv(self):
        self.target_encoder = MLP(
            *self.enc_args_kwargs[0],
            dropout=0.0,
            **{k: v for k, v in self.enc_args_kwargs[1].items() if k != "dropout"},
        )
        self.target_encoder.load_state_dict(self.source_encoder.state_dict())
        self.dis = MLP(
            self.emb_dim,
            1,
            hidden_layer_sizes=(512, 1024),
            dropout=0.5,
            bn_kwargs=self.enc_args_kwargs[1].get("bn_kwargs"),
        )

    def pretraining(self):
        """Enable pretraining mode to train model on source domain."""
        self.is_encoder_source = True
        set_requires_grad(self.source_encoder, True)
        set_requires_grad(self.clf, True)

    def advtraining(self, train_dis=True):
        """Enable adversarial training mode to train target encoder.

        Args:
        train_dis (bool): Whether to train discriminator first; else train
            target encoder

        """
        set_requires_grad(self.source_encoder, False)

        if train_dis:
            self.train_discriminator()
        else:
            self.train_target_encoder()

    def target_inference(self):
        """Inference mode for target. Does not change grad or eval mode."""
        self.set_encoder("target")

    def train_discriminator(self):
        set_requires_grad(self.target_encoder, False)
        set_requires_grad(self.dis, True)

    def train_target_encoder(self):
        set_requires_grad(self.target_encoder, True)
        set_requires_grad(self.dis, False)

    def set_encoder(self, encoder="source"):
        if encoder == "source":
            self.is_encoder_source = True
        elif encoder == "target":
            self.is_encoder_source = False
        else:
            current_encoder_str = "'source'" if self.is_encoder_source else "'target'"
            warnings.warn(
                (
                    "encoder parameter should be 'source' or 'target', got",
                    f" {encoder}; encoder is currently {current_encoder_str}",
                ),
                RuntimeWarning,
            )
