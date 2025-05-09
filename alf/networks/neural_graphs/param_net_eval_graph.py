import torch
import torch.nn as nn

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
        super().__init__()
        param_net_input_spec = param_net_kwargs.get("input_tensor_spec", None)
        assert param_net_input_spec is not None, (
            "param_net_kwargs needs to have input_tensor_spec")
        input_dim = param_net_input_spec.shape[0]

        if param_net_ctor is None:
            param_net_ctor = ActorParamNetwork
        num_layers = len(param_net_kwargs.get("fc_layer_params", 0)) + 1
        self.param_net = param_net_ctor(**param_net_kwargs) 

        samples = (
            sample_init
            if sample_init is not None
            else 2 * torch.rand(num_samples, input_dim) - 1
        )
        self.samples = nn.Parameter(samples, requires_grad=sample_init is None)

        self.proj_dim = proj_dim
        if proj_dim is not None:
            self.proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(num_samples, proj_dim),
                        nn.LayerNorm(proj_dim),
                    )
                    for _ in range(num_layers + 1)
                ]
            )

    def forward(self, weights, biases):
        self.param_net.update_parameters(weights, biases, update_n_groups=True)
        out = self.param_net(self.samples, full_neurons=True)[0]

        if self.proj_dim is not None:
            out = [proj(out[i].permute(1, 2, 0)) for i, proj in enumerate(self.proj)]
            out = torch.cat(out, dim=1)
            return out
        else:
            out = torch.cat(out, dim=-1)
            return out.permute(0, 2, 1)
