import torch
from transformer_lens import HookedTransformer, FactoredMatrix
from typing import Literal, Tuple

# TODO: Tensortyping

def circuit_effective_dimension(circuit: FactoredMatrix) -> float:
    svd = circuit.svd()[1].flatten()
    return float(svd.sum()**2 / (svd**2).sum())

# TODO: Make a dataclass
def attn_circuits_effective_dim(model: HookedTransformer) -> Tuple[float, float]:
    ov_eff_dim = circuit_effective_dimension(model.OV)
    qk_eff_dim = circuit_effective_dimension(model.QK)
    return qk_eff_dim, ov_eff_dim

# TODO: I'm not sure what the motivation behind this is
# TODO: Typing
def weight_dimensions_unroll(model: HookedTransformer):
    q_ov = model.OV.svd()[1]
    q_qk = model.QK.svd()[1]
    eff_dim_ov = (q_ov).sum(axis=-1)**2 / (q_ov**2).sum(axis=-1)
    eff_dim_qk = (q_qk).sum(axis=-1)**2 / (q_qk**2).sum(axis=-1)
    dims = [eff_dim_ov.mean(axis=-1), eff_dim_qk.mean(axis=-1)]
    return dims

# TODO: Test for memory consistency
# TODO: Typing
def effective_rank(model: HookedTransformer):
  q_ov = model.OV.svd()[1]
  q_qk = model.QK.svd()[1]
  q_ov = q_ov/q_ov.sum(dim=-1, keepdim=True)
  q_qk = q_qk/q_qk.sum(dim=-1, keepdim=True)
  q_ov = torch.sum(q_ov * torch.log(q_ov), dim=-1)
  q_qk = torch.sum(q_qk * torch.log(q_qk), dim=-1)
  dims = [torch.exp(-q_ov).mean(axis=-1).cpu().detach().numpy(), torch.exp(-q_qk).mean(axis=-1).cpu().detach().numpy()]
  del q_ov, q_qk
  return dims

if __name__ == "__main__":
    device = 'cpu'
    model_name = 'pythia-160m'
    index = 120
    model = HookedTransformer.from_pretrained(model_name, checkpoint_index=index, device=device)
    attn_circuits_effective_dim(model)