from typing import List,  Literal, Optional, Dict
import torch
from transformer_lens import HookedTransformer, evals
from jaxtyping import Float
from dataclasses import dataclass

@dataclass
class Extracted:
    model_name: str
    num_tokens_seen: int
    data: str
    logits: Optional[Float[torch.Tensor, "seq_pos vocab_len"]] = None
    activations: Optional[Dict[str, Float[torch.Tensor, "seq_pos d_model"]]] = None

Module = Literal['attn_out', 'mlp_out']

def extractor(
    model_name: str, 
    checkpoint_idx: int, 
    data: List[str], 
    device: str, 
    extract_logits: bool, 
    extract_activations: List[Module]
) -> List[Extracted]:
    
    def _extract_activations_and_logits(x: str, model: HookedTransformer, extract_logits: bool, extract_activations: List[Module]) -> Extracted:
        if not extract_activations:
            logits = model(x, return_type="logits")
            return Extracted(
                model_name=model_name, 
                num_tokens_seen=model.cfg.checkpoint_value, 
                data=x, 
                logits=logits
            )
        
        logits, cache = model.run_with_cache(x, return_type="logits", loss_per_token=True, remove_batch_dim=True)
        activations = {key: activation for key, activation in cache.items() if any([x in key for x in extract_activations])}
        return Extracted(
            model_name=model_name, 
            num_tokens_seen=model.cfg.checkpoint_value, 
            data=x, 
            logits=logits if extract_logits else None,
            activations= activations
        )
    
    model = HookedTransformer.from_pretrained(model_name, checkpoint_index=checkpoint_idx, device=device)
    return [_extract_activations_and_logits(x, model, extract_logits, extract_activations) for x in data]

if __name__ == "__main__":
    extractor('pythia-160m', -1, ["Hello World", "My name is Johny Smith"], "cpu", extract_logits=True, extract_activations=['attn_out', 'mlp_out'])