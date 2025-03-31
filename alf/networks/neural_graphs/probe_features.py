import hydra
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from nn.inr import make_functional, params_to_tensor, wrap_func


class GraphProbeFeatures(nn.Module):
    # TODO: neads rewrite with callable inr_model
    def __init__(self, d_in, num_inputs, inr_model, input_init=None, proj_dim=None):
        super().__init__()
        inr = hydra.utils.instantiate(inr_model)
        fmodel, params = make_functional(inr)

        vparams, vshapes = params_to_tensor(params)
        self.sirens = torch.vmap(wrap_func(fmodel, vshapes))

        inputs = (
            input_init
            if input_init is not None
            else 2 * torch.rand(1, num_inputs, d_in) - 1
        )
        self.inputs = nn.Parameter(inputs, requires_grad=input_init is None)

        # self.reshape_weights = Rearrange("b i o 1 -> b (o i)")
        # self.reshape_biases = Rearrange("b o 1 -> b o")

        self.proj_dim = proj_dim
        if proj_dim is not None:
            self.proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(num_inputs, proj_dim),
                        nn.LayerNorm(proj_dim),
                    )
                    for _ in range(inr.num_layers + 1)
                ]
            )

    def forward(self, weights, biases):
        # weights = [self.reshape_weights(w) for w in weights]
        # biases = [self.reshape_biases(b) for b in biases]
        params_flat = torch.cat(
            [w_or_b for p in zip(weights, biases) for w_or_b in p], dim=-1
        )

        out = self.sirens(params_flat, self.inputs.expand(params_flat.shape[0], -1, -1))
        if self.proj_dim is not None:
            out = [proj(out[i].permute(0, 2, 1)) for i, proj in enumerate(self.proj)]
            out = torch.cat(out, dim=1)
            return out
        else:
            out = torch.cat(out, dim=-1)
            return out.permute(0, 2, 1)
