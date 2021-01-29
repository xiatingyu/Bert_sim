# -*- encoding:utf-8 -*-
import torch
from uer_sim.layers.embeddings import BertEmbedding, WordEmbedding
from uer_sim.encoders.bert_encoder import BertEncoder
from uer_sim.encoders.rnn_encoder import LstmEncoder, GruEncoder
from uer_sim.encoders.birnn_encoder import BilstmEncoder
from uer_sim.encoders.cnn_encoder import CnnEncoder, GatedcnnEncoder
from uer_sim.encoders.attn_encoder import AttnEncoder
from uer_sim.encoders.gpt_encoder import GptEncoder
from uer_sim.encoders.mixed_encoder import RcnnEncoder, CrnnEncoder
from uer_sim.encoders.synt_encoder import SyntEncoder
from uer_sim.targets.bert_target import BertTarget
from uer_sim.targets.lm_target import LmTarget
from uer_sim.targets.cls_target import ClsTarget
from uer_sim.targets.mlm_target import MlmTarget
from uer_sim.targets.nsp_target import NspTarget
from uer_sim.targets.s2s_target import S2sTarget
from uer_sim.targets.bilm_target import BilmTarget
from uer_sim.subencoders.avg_subencoder import AvgSubencoder
from uer_sim.subencoders.rnn_subencoder import LstmSubencoder
from uer_sim.subencoders.cnn_subencoder import CnnSubencoder
from uer_sim.models.model import Model


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder, 
    and target layers yield pretrained models of different 
    properties. 
    We could select suitable one for downstream tasks.
    """

    if args.subword_type != "none":
        subencoder = globals()[args.subencoder.capitalize() + "Subencoder"](args, len(args.sub_vocab))
    else:
        subencoder = None

    embedding = globals()[args.embedding.capitalize() + "Embedding"](args, len(args.vocab))
    encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
    target = globals()[args.target.capitalize() + "Target"](args, len(args.vocab))
    model = Model(args, embedding, encoder, target, subencoder)

    return model
