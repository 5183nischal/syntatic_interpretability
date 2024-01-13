import torch
from transformer_lens import HookedTransformer, FactoredMatrix # type: ignore
from jaxtyping import Float
from dataclasses import dataclass

@dataclass
class AttnEffectiveDimension:
    qk_eff_dims: Float[torch.Tensor, "n_layers n_heads"]
    ov_eff_dims: Float[torch.Tensor, "n_layers n_heads"]

def circuit_effective_dimension(circuit: FactoredMatrix) -> Float[torch.Tensor, "n_layers n_heads"]:
    # circuit: Float[torch.Tensor, "n_layers n_heads d_model d_model"]
    singular_values: Float[torch.Tensor, "n_layers n_heads d_head"] = circuit.svd()[1]
    return singular_values.sum(dim=-1)**2 / (singular_values**2).sum(dim=-1)

def attn_circuits_effective_dimimension(model: HookedTransformer) -> AttnEffectiveDimension:
    return AttnEffectiveDimension(
        qk_eff_dims = circuit_effective_dimension(model.QK),
        ov_eff_dims = circuit_effective_dimension(model.OV)
    )

@dataclass
class AttnEffectiveRank:
    qk_eff_ranks: Float[torch.Tensor, "n_layers n_heads"]
    ov_eff_ranks: Float[torch.Tensor, "n_layers n_heads"]

def circuit_effective_rank(circuit: FactoredMatrix) -> Float[torch.Tensor, "n_layers n_heads"]:
    # circuit: Float[torch.Tensor, "n_layers n_heads d_model d_model"]
    singular_values: Float[torch.Tensor, "n_layers n_heads d_head"] = circuit.svd()[1]
    singular_values_normalized: Float[torch.Tensor, "n_layers n_heads d_head"] = singular_values / singular_values.sum(dim=-1, keepdim=True)
    singular_value_entropy: Float[torch.Tensor, "n_layers n_heads"] = (-singular_values_normalized * torch.log(singular_values_normalized)).sum(dim=-1)
    return torch.exp(singular_value_entropy) # TODO: Does this need to add an eps??

def attn_circuits_effective_rank(model: HookedTransformer) -> AttnEffectiveRank:
    return AttnEffectiveRank(
        qk_eff_ranks = circuit_effective_rank(model.QK),
        ov_eff_ranks = circuit_effective_rank(model.OV)
    )