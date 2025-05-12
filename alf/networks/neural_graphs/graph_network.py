import torch
import torch.nn as nn

import alf
from .pooling import HomogeneousAggregator
from .relational_transformer import RTLayer 


@alf.configurable
class GraphNetwork(nn.Module):
    def __init__(
        self,
        layer_layout,
        d_out,
        d_node=64,
        d_edge=32,
        d_attn_hid=128,
        d_node_hid=128,
        d_edge_hid=64,
        d_out_hid=256,
        n_layers=4,
        n_heads=8,
        dropout=0.0,
        node_update_type="rt",
        disable_edge_updates=False,
        pooling_method="cat",
        pooling_layer_idx="last",
        modulate_v=True,
        use_ln=True,
        tfixit_init=False,
    ):
        super().__init__()
        assert pooling_method is not "cls_token", (
            "Using cls_token is not supported.")
        self.pooling_method = pooling_method
        self.pooling_layer_idx = pooling_layer_idx

        self.layers = nn.ModuleList(
            [
                torch.jit.script(
                    RTLayer(
                        d_node,
                        d_edge,
                        d_attn_hid,
                        d_node_hid,
                        d_edge_hid,
                        n_heads,
                        dropout,
                        node_update_type=node_update_type,
                        disable_edge_updates=(
                            (disable_edge_updates or (i == n_layers - 1))
                            and pooling_method != "mean_edge"
                            and pooling_layer_idx != "all"
                        ),
                        modulate_v=modulate_v,
                        use_ln=use_ln,
                        tfixit_init=tfixit_init,
                        n_layers=n_layers,
                    )
                )
                for i in range(n_layers)
            ]
        )

        if pooling_method != "cls_token":
            self.pool = HomogeneousAggregator(
                pooling_method,
                pooling_layer_idx,
                layer_layout,
            )

        num_graph_features = (
            layer_layout[-1] * d_node
            if pooling_method == "cat" and pooling_layer_idx == "last"
            else d_edge if pooling_method in ("mean_edge", "max_edge") else d_node
        )
        self.proj_out = nn.Sequential(
            nn.Linear(num_graph_features, d_out_hid),
            nn.ReLU(),
            # nn.Linear(d_out_hid, d_out_hid),
            # nn.ReLU(),
            nn.Linear(d_out_hid, d_out),
        )

    def forward(self, inputs):
        node_features, edge_features, mask = inputs

        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, mask)

        if self.pooling_method == "cls_token":
            graph_features = node_features[:, 0]
        else:
            graph_features = self.pool(node_features, edge_features)

        return self.proj_out(graph_features)
