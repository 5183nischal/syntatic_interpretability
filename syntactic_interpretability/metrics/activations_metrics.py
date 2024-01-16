import torch
import numpy as np
from jaxtyping import Float
from transformer_lens.ActivationCache import ActivationCache
import re
from typing import Literal


def extract_modules_from_activation_cache(module: Literal['attn', 'mlp'], cache: ActivationCache) -> Float[torch.Tensor, "n_layers seq_pos d_model"]:
  out = []
  for module_name, module_tensor in cache.items():
     if not re.match(rf"^blocks\.\d+\.hook_{module}_out$", module_name):
        continue
     out.append(module_tensor)
  return torch.cat(out, dim=0)


def attn_eff_dim(n_layers: int, cache: ActivationCache):
    """
    Computes 
    """
    # TODO: Double check whether language is correct in the write up.
    with torch.no_grad():
      eff_dim = []
      for i in range(n_layers):
        layer: Float[torch.Tensor, "seq_pos d_model"] = cache[f"blocks.{i}.hook_attn_out"]
        eigenvalues: Float[torch.Tensor, "seq_pos"] = torch.linalg.eigvalsh(layer @ layer.T)
        layer_eff_dim: Float[torch.Tensor, ""] = eigenvalues.sum()**2 / (eigenvalues**2).sum()
        eff_dim.append(layer_eff_dim)
      return torch.cat(eff_dim)

if __name__ == "__main__":
  from transformer_lens import HookedTransformer

  model = HookedTransformer.from_pretrained('pythia-160m', checkpoint_index=-1, device='cpu')
  _, cache = model.run_with_cache('Hello World', return_type="loss", loss_per_token=True, remove_batch_dim=True)
  attn_eff_dim(12, cache)