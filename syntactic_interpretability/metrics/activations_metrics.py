import torch
import numpy as np

# TODO: TensorTyping
# TODO: See if the n_layers is needed or if we can just derive it from the cache itself
def attn_eff_dim(n_layers, cache):
    """
    Computes the effective dimension of the attention layer

    TODO: Write out the mathematical description of the effective dimension
    """
    eff_dim = []
    for i in range(n_layers):
      neuron = cache[f"blocks.{i}.hook_attn_out"].T
      q = torch.linalg.svdvals(neuron @ neuron.T)
      temp_eff_dim = (q).sum()**2 / (q**2).sum()
      eff_dim.append(temp_eff_dim.item())
    return np.asarray(eff_dim)