import torch
import torch.nn as nn
from rff.layers import GaussianEncoding

import alf


@alf.configurable
class ActorGraph(nn.Module):
    def __init__(
        self,
        layer_layout,
        d_node=32,
        d_edge=32,
        num_eval_samples=64,
        d_in=1,
        d_edge_in=1,
        rev_edge_features=False,
        zero_out_bias=False,
        zero_out_weights=False,
        inp_factor=1,
        input_layers=1,
        sin_emb=False,
        sin_emb_dim=128,
        use_pos_embed=True,
        # stats=None,
    ):
        super().__init__()
        self.rev_edge_features = rev_edge_features
        self.zero_out_bias = zero_out_bias
        self.zero_out_weights = zero_out_weights
        self.use_pos_embed = use_pos_embed
        # self.stats = stats if stats is not None else {}
        self._d_node = d_node
        self._d_edge = d_edge

        self.pos_embed_layout = (
            [1] * layer_layout[0] + layer_layout[1:-1] + [1] * layer_layout[-1]
        )
        self.pos_embed = nn.Parameter(torch.randn(len(self.pos_embed_layout), d_node))

        if not self.zero_out_weights:
            proj_weight = []
            if sin_emb:
                proj_weight.append(
                    GaussianEncoding(
                        sigma=inp_factor,
                        input_size=d_edge_in
                        + (2 * d_edge_in if rev_edge_features else 0),
                        encoded_size=sin_emb_dim,
                    )
                )
                proj_weight.append(nn.Linear(2 * sin_emb_dim, d_edge))
            else:
                proj_weight.append(
                    nn.Linear(
                        d_edge_in + (2 * d_edge_in if rev_edge_features else 0), d_edge
                    )
                )

            for i in range(input_layers - 1):
                proj_weight.append(nn.SiLU())
                proj_weight.append(nn.Linear(d_edge, d_edge))

            self.proj_weight = nn.Sequential(*proj_weight)
        if not self.zero_out_bias:
            proj_bias = []
            if sin_emb:
                proj_bias.append(
                    GaussianEncoding(
                        sigma=inp_factor,
                        input_size=d_in,
                        encoded_size=sin_emb_dim,
                    )
                )
                proj_bias.append(nn.Linear(2 * sin_emb_dim, d_node))
            else:
                proj_bias.append(nn.Linear(d_in, d_node))

            for i in range(input_layers - 1):
                proj_bias.append(nn.SiLU())
                proj_bias.append(nn.Linear(d_node, d_node))

            self.proj_bias = nn.Sequential(*proj_bias)

        self.proj_node_in = nn.Linear(d_node, d_node)
        self.proj_edge_in = nn.Linear(d_edge, d_edge)
        self.proj_eval_out = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_eval_samples, d_node),
                    nn.LayerNorm(d_node),
                )
                for _ in range(len(layer_layout))
            ]
        )

    def forward(self, inputs):
        node_features, edge_features, eval_out = inputs
        eval_out = [
            proj(eval_out[i].permute(1, 2, 0)) for i, proj in enumerate(
                self.proj_eval_out)]
        probe_features = torch.cat(eval_out, dim=1)

        node_features = node_features.unsqueeze(-1)
        edge_features = edge_features.unsqueeze(-1)
        mask = edge_features.sum(dim=-1, keepdim=True) != 0
        if self.rev_edge_features:
            rev_edge_features = edge_features.transpose(-2, -3)
            edge_features = torch.cat(
                [edge_features, rev_edge_features, edge_features + rev_edge_features],
                dim=-1,
            )
            mask = mask | mask.transpose(-3, -2)

        if self.zero_out_weights:
            edge_features = torch.zeros(
                (*edge_features.shape[:-1], self._d_edge),
                device=edge_features.device,
                dtype=edge_features.dtype,
            )
        else:
            edge_features = self.proj_weight(edge_features)
        if self.zero_out_bias:
            # only zero out bias, not param_eval_graph
            node_features = torch.zeros(
                (*node_features.shape[:-1], self._d_node),
                device=node_features.device,
                dtype=node_features.dtype,
            )
        else:
            node_features = self.proj_bias(node_features)

        if probe_features is not None:
            node_features = node_features + probe_features

        node_features = self.proj_node_in(node_features)
        edge_features = self.proj_edge_in(edge_features)

        if self.use_pos_embed:
            pos_embed = torch.cat(
                [
                    # each pos_embed[i]: [d] -> [1 n d] 
                    self.pos_embed[i].unsqueeze(0).expand(1, n, -1)
                    for i, n in enumerate(self.pos_embed_layout)
                ],
                dim=1,
            )
            node_features = node_features + pos_embed
        return node_features, edge_features, mask
