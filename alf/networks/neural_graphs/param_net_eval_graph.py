import hydra
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

import alf
from alf.networks.param_networks import ActorParamNetwork


@alf.configurable
class ParamNetEvalGraph(nn.Module):
    def __init__(self, 
                 param_net_kwargs, 
                 param_net_ctor=None,
                 num_samples=64, 
                 sample_init=None, 
                 proj_dim=None):
        param_net_input_spec = param_net_kwargs.get("input_spec", None)
        assert param_net_input_spec is not None, (
            "param_net_kwargs needs to have input_spec")
        input_dim = param_net_input_spec.shape[0]

        if param_net_ctor is None:
            param_net_ctor = ActorParamNetwork
        self.param_net = param_net_ctor(**param_net_kwargs) 

        samples = (
            sample_init
            if sample_init is not None
            else 2 * torch.rand(1, num_samples, input_dim) - 1
        )
        self.samples = nn.Parameter(samples, requires_grad=sample_init is None)

        # self.reshape_weights = Rearrange("b i o 1 -> b (o i)")
        # self.reshape_biases = Rearrange("b o 1 -> b o")

        self.proj_dim = proj_dim
        if proj_dim is not None:
            self.proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(num_samples, proj_dim),
                        nn.LayerNorm(proj_dim),
                    )
                    for _ in range(inr.num_layers + 1)
                ]
            )

    def forward(self, weights, biases):

        # # weights = [self.reshape_weights(w) for w in weights]
        # # biases = [self.reshape_biases(b) for b in biases]
        # params_flat = torch.cat(
        #     [w_or_b for p in zip(weights, biases) for w_or_b in p], dim=-1
        # )
        # out = self.sirens(params_flat, self.inputs.expand(params_flat.shape[0], -1, -1))

        self.param_net.update_parameters(weights, biases)
        out = self.param_net(self.samples, full_neurons=True)

        if self.proj_dim is not None:
            out = [proj(out[i].permute(0, 2, 1)) for i, proj in enumerate(self.proj)]
            out = torch.cat(out, dim=1)
            return out
        else:
            out = torch.cat(out, dim=-1)
            return out.permute(0, 2, 1)
