import torch
import torch.nn as nn
from torch_geometric.nn.aggr import (
    AttentionalAggregation,
    GraphMultisetTransformer,
    MaxAggregation,
    MeanAggregation,
    SetTransformerAggregation,
)


class CatAggregation(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(1, 2)

    def forward(self, x, index=None):
        return self.flatten(x)


class HeterogeneousAggregator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        pooling_method,
        pooling_layer_idx,
        input_channels,
        num_classes,
    ):
        super().__init__()
        self.pooling_method = pooling_method
        self.pooling_layer_idx = pooling_layer_idx
        self.input_channels = input_channels
        self.num_classes = num_classes

        if pooling_layer_idx == "all":
            self._pool_layer_idx_fn = self.get_all_layer_indices
        elif pooling_layer_idx == "last":
            self._pool_layer_idx_fn = self.get_last_layer_indices
        elif isinstance(pooling_layer_idx, int):
            self._pool_layer_idx_fn = self.get_nth_layer_indices
        else:
            raise ValueError(f"Unknown pooling layer index {pooling_layer_idx}")

        if pooling_method == "mean":
            self.pool = MeanAggregation()
        elif pooling_method == "max":
            self.pool = MaxAggregation()
        elif pooling_method == "cat":
            self.pool = CatAggregation()
        elif pooling_method == "attentional_aggregation":
            self.pool = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                ),
                nn=nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, output_dim),
                ),
            )
        elif pooling_method == "set_transformer":
            self.pool = SetTransformerAggregation(
                input_dim, heads=8, num_encoder_blocks=4, num_decoder_blocks=4
            )
        elif pooling_method == "graph_multiset_transformer":
            self.pool = GraphMultisetTransformer(input_dim, k=8, heads=8)
        else:
            raise ValueError(f"Unknown pooling method {pooling_method}")

    def get_last_layer_indices(
        self, x, layer_layouts, node_mask=None, return_dense=False
    ):
        batch_size = x.shape[0]
        device = x.device

        # NOTE: node_mask needs to exist in the heterogeneous case only
        if node_mask is None:
            node_mask = torch.ones_like(x[..., 0], dtype=torch.bool, device=device)

        valid_layer_indices = (
            torch.arange(node_mask.shape[1], device=device)[None, :] * node_mask
        )
        last_layer_indices = valid_layer_indices.topk(
            k=self.num_classes, dim=1
        ).values.fliplr()

        if return_dense:
            return torch.arange(batch_size, device=device)[:, None], last_layer_indices

        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(
            self.num_classes
        )
        return batch_indices, last_layer_indices.flatten()

    def get_nth_layer_indices(
        self, x, layer_layouts, node_mask=None, return_dense=False
    ):
        batch_size = x.shape[0]
        device = x.device

        cum_layer_layout = [
            torch.cumsum(torch.tensor([0] + layer_layout), dim=0)
            for layer_layout in layer_layouts
        ]

        layer_sizes = torch.tensor(
            [layer_layout[self.pooling_layer_idx] for layer_layout in layer_layouts],
            dtype=torch.long,
            device=device,
        )
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(
            layer_sizes
        )
        layer_indices = torch.cat(
            [
                torch.arange(
                    layout[self.pooling_layer_idx],
                    layout[self.pooling_layer_idx + 1],
                    device=device,
                )
                for layout in cum_layer_layout
            ]
        )
        return batch_indices, layer_indices

    def get_all_layer_indices(
        self, x, layer_layouts, node_mask=None, return_dense=False
    ):
        """Imitate flattening with indexing"""
        batch_size, num_nodes = x.shape[:2]
        device = x.device
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(
            num_nodes
        )
        layer_indices = torch.arange(num_nodes, device=device).repeat(batch_size)
        return batch_indices, layer_indices

    def forward(self, x, layer_layouts, node_mask=None):
        # NOTE: `cat` only works with `pooling_layer_idx == "last"`
        return_dense = self.pooling_method == "cat" and self.pooling_layer_idx == "last"
        batch_indices, layer_indices = self._pool_layer_idx_fn(
            x, layer_layouts, node_mask=node_mask, return_dense=return_dense
        )

        flat_x = x[batch_indices, layer_indices]
        return self.pool(flat_x, index=batch_indices)


class HomogeneousAggregator(nn.Module):
    def __init__(
        self,
        pooling_method,
        pooling_layer_idx,
        layer_layout,
    ):
        super().__init__()
        self.pooling_method = pooling_method
        self.pooling_layer_idx = pooling_layer_idx
        self.layer_layout = layer_layout

    def forward(self, node_features, edge_features):
        if self.pooling_method == "mean" and self.pooling_layer_idx == "all":
            graph_features = node_features.mean(dim=1)
        elif self.pooling_method == "max" and self.pooling_layer_idx == "all":
            graph_features = node_features.max(dim=1).values
        elif self.pooling_method == "mean" and self.pooling_layer_idx == "last":
            graph_features = node_features[:, -self.layer_layout[-1] :].mean(dim=1)
        elif self.pooling_method == "cat" and self.pooling_layer_idx == "last":
            graph_features = node_features[:, -self.layer_layout[-1] :].flatten(1, 2)
        elif self.pooling_method == "mean" and isinstance(self.pooling_layer_idx, int):
            graph_features = node_features[
                :,
                self.layer_idx[self.pooling_layer_idx] : self.layer_idx[
                    self.pooling_layer_idx + 1
                ],
            ].mean(dim=1)
        elif self.pooling_method == "cat_mean" and self.pooling_layer_idx == "all":
            graph_features = torch.cat(
                [
                    node_features[:, self.layer_idx[i] : self.layer_idx[i + 1]].mean(
                        dim=1
                    )
                    for i in range(len(self.layer_layout))
                ],
                dim=1,
            )
        elif self.pooling_method == "mean_edge" and self.pooling_layer_idx == "all":
            graph_features = edge_features.mean(dim=(1, 2))
        elif self.pooling_method == "max_edge" and self.pooling_layer_idx == "all":
            graph_features = edge_features.flatten(1, 2).max(dim=1).values
        elif self.pooling_method == "mean_edge" and self.pooling_layer_idx == "last":
            graph_features = edge_features[:, :, -self.layer_layout[-1] :].mean(
                dim=(1, 2)
            )
        return graph_features
