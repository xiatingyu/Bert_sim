# -*- encoding:utf-8 -*-
import torch.nn as nn
from uer_sim.layers.layer_norm import LayerNorm
from uer_sim.layers.position_ffn import PositionwiseFeedForward
from uer_sim.layers.multi_headed_attn import MultiHeadedAttention
from uer_sim.layers.transformer import TransformerLayer


class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([
            TransformerLayer(args) for _ in range(self.layers_num)
        ])
        
    def forward(self, emb, seg, similarity):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)

        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        #print(similarity.shape)
        similarity = similarity.unsqueeze(1)
        similarity = similarity.float()

        #print('%.7f' % similarity[0][0][1][4])
        hidden = emb
        for i in range(self.layers_num):
            if i == 0:
                hidden = self.transformer[i](hidden, mask, similarity)
            else:
                hidden = self.transformer[i](hidden, mask)
        return hidden
