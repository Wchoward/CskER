"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import dgl.function as fn
# from dgl.nn import GATConv
from torch.nn.modules.activation import ReLU, Tanh
from .func_utils import gather_nd


class StaticGraphAttentionLayer(nn.Module):
    def __init__(self, args):
        super(StaticGraphAttentionLayer, self).__init__()
        self.args = args
        self.num_trans_units = args.num_trans_units
        # 将预训练好的entity和relation的transE表示concat到一起，并加载为entity_trans Embedding
        self.entity_trans = nn.Embedding.from_pretrained(
            torch.from_numpy(
                np.concatenate(
                    (
                        np.loadtxt(args.entity_trans_path, dtype=np.float32),
                        np.loadtxt(args.relation_trans_path, dtype=np.float32),
                    ),
                    axis=0,
                )
            )
        )
        # self.relation_embed = nn.Embedding.from_pretrained(torch.from_numpy())
        self.fc_trans_transform = nn.Sequential(nn.Linear(self.num_trans_units, self.num_trans_units), nn.Tanh())
        self.fc_head_tail_transform = nn.Sequential(
            nn.Linear(self.num_trans_units * 2, self.num_trans_units), nn.Tanh()
        )
        self.fc_relation_transform = nn.Sequential(
            nn.Linear(self.num_trans_units, self.num_trans_units),
        )

    def forward(self, sents, triples, sent_triples):
        # triple_num =?= sent_seq_len ?
        # sents_triple: bsz * sent_seq_len * 1
        bsz = sents.shape[0]
        sent_seq_len = sents.shape[1]
        # triples: bsz * triple_num * triple_len * 3
        triple_num = triples.shape[1]
        triple_len = triples.shape[2]
        # triples_embedding: bsz * triple_num * triple_len * (3*num_trans_units)
        triples_embedding = self.fc_trans_transform(self.entity_trans(triples)).reshape(
            -1, triple_num, triple_len, 3 * self.num_trans_units
        )
        # head: bsz * triple_num * triple_len * num_trans_units
        head, relation, tail = triples_embedding.split([self.num_trans_units] * 3, dim=3)
        # head_tail: bsz * triple_num * triple_len * (2*num_trans_units)
        head_tail = torch.cat((head, tail), dim=3)
        # head_tail_transformed: bsz * triple_num * triple_len * (num_trans_units)
        head_tail_transformed = self.fc_head_tail_transform(head_tail)
        # relation_transformed: bsz * triple_num * triple_len * (num_trans_units)
        relation_transformed = self.fc_relation_transform(relation)
        # e_weight: bsz * triple_num * triple_len
        e_weight = torch.sum(relation_transformed * head_tail_transformed, dim=3)
        # alpha_weight: bsz * triple_num * triple_len
        alpha_weight = F.softmax(e_weight, dim=-1)
        # graph_embed: bsz * triple_num * (2*num_trans_units)
        graph_embed = torch.sum(torch.unsqueeze(alpha_weight, dim=3) * head_tail, dim=2)
        # indices: bsz * sent_seq_len * 2
        graph_embed_indices = torch.cat(
            (
                torch.arange(bsz, dtype=torch.int32)
                .reshape(-1, 1, 1)
                .repeat([1, sent_seq_len, 1])
                .to(self.args.device),
                sent_triples,
            ),
            dim=2,
        )
        # graph_embed_input: bsz * sent_seq_len * (2*num_trans_units)
        graph_embed_input = gather_nd(graph_embed, graph_embed_indices)
        return graph_embed_input


# class GAT(nn.Module):
# def __init__(
#     self,
#     g,
#     num_layers,
#     in_dim,
#     num_hidden,
#     num_classes,
#     heads,
#     activation,
#     feat_drop,
#     attn_drop,
#     negative_slope,
#     residual,
# ):
#     super(GAT, self).__init__()
#     self.g = g
#     self.num_layers = num_layers
#     self.gat_layers = nn.ModuleList()
#     self.activation = activation
#     # input projection (no residual)
#     self.gat_layers.append(
#         GATConv(in_dim, num_hidden, heads[0], feat_drop, attn_drop, negative_slope, False, self.activation)
#     )
#     # hidden layers
#     for l in range(1, num_layers):
#         # due to multi-head, the in_dim = num_hidden * num_heads
#         self.gat_layers.append(
#             GATConv(
#                 num_hidden * heads[l - 1],
#                 num_hidden,
#                 heads[l],
#                 feat_drop,
#                 attn_drop,
#                 negative_slope,
#                 residual,
#                 self.activation,
#             )
#         )
#     # output projection
#     self.gat_layers.append(
#         GATConv(
#             num_hidden * heads[-2], num_classes, heads[-1], feat_drop, attn_drop, negative_slope, residual, None
#         )
#     )

# def forward(self, inputs):
#     h = inputs
#     for l in range(self.num_layers):
#         h = self.gat_layers[l](self.g, h).flatten(1)
#     # output projection
#     logits = self.gat_layers[-1](self.g, h).mean(1)
#     return logits
