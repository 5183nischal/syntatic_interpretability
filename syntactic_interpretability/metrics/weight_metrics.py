import torch
from transformer_lens import HookedTransformer, FactoredMatrix # type: ignore
from jaxtyping import Float
from dataclasses import dataclass
from typing import Literal

# TODO: Test for Memory Leakages
# TODO: Change dataclass into pydantic??
# TODO: Collapse all metrics into AttnCircuitMeasurement properties & Make them optional & create a config for running the sweep
# TODO: Create a save to disk function
# TODO: Add checkpoint_index to attn_circuit_measurements

Metric = Literal['effective_dimension', 'effective_rank']

def _circuit_effective_dimension(circuit: FactoredMatrix) -> Float[torch.Tensor, "n_layers n_heads"]:
    # circuit: Float[torch.Tensor, "n_layers n_heads d_model d_model"]
    with torch.no_grad():
        singular_values: Float[torch.Tensor, "n_layers n_heads d_head"] = circuit.svd()[1]
        return singular_values.sum(dim=-1)**2 / (singular_values**2).sum(dim=-1)

def _circuit_effective_rank(circuit: FactoredMatrix) -> Float[torch.Tensor, "n_layers n_heads"]:
    # circuit: Float[torch.Tensor, "n_layers n_heads d_model d_model"]
    with torch.no_grad():
        singular_values: Float[torch.Tensor, "n_layers n_heads d_head"] = circuit.svd()[1]
        singular_values_normalized: Float[torch.Tensor, "n_layers n_heads d_head"] = singular_values / singular_values.sum(dim=-1, keepdim=True)
        singular_value_entropy: Float[torch.Tensor, "n_layers n_heads"] = (-singular_values_normalized * torch.log(singular_values_normalized)).sum(dim=-1)  # TODO: Do we need a + EPS here? 
        return torch.exp(singular_value_entropy)

METRIC_REGISTRIY = {
    'effective_dimension': _circuit_effective_dimension,
    'effective_rank': _circuit_effective_rank
}

@dataclass
class AttnCircuitMeasurements:
    model_name: str
    num_tokens_seen: int
    metric: Metric
    qk: Float[torch.Tensor, "n_layers n_heads"]
    ov: Float[torch.Tensor, "n_layers n_heads"]

    @property
    def qk_reduced(self) -> float:
        return torch.mean(self.qk).item()
    
    @property
    def ov_reduced(self) -> float:
        return torch.mean(self.qk).item()

def measure_attn_circuits(model: HookedTransformer, metric: Metric) -> AttnCircuitMeasurements:
    return AttnCircuitMeasurements(
        model_name=model.cfg.model_name,
        num_tokens_seen=model.cfg.checkpoint_value,
        metric=metric,
        qk = METRIC_REGISTRIY[metric](model.QK),
        ov = METRIC_REGISTRIY[metric](model.OV)
    )

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("pythia-160m", checkpoint_index=-1, device='cpu')
    foo = measure_attn_circuits(model, 'effective_dimension')
    bar = measure_attn_circuits(model, 'effective_rank')