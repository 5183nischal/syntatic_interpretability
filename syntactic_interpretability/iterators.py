from typing import List, Any, Tuple, Dict
import torch
from transformer_lens import HookedTransformer, evals
from transformers import AutoTokenizer


class TrainingDynamicsConfig:
    model_names: List[str] # TODO: Add a verifier that interfaces with the correct external libraries
    checkpoint_indices: List[int]
    dataset: Any
    num_tokens_predicted: int
    num_devices: int # TODO Add a verifier which asserts that this value is greater than 0, and that there are the correct num of devices
    tokenizer: str
    data_loader_batch_size: int
    # TODO: metric ???

# TODO: Add tensortyping 
def gather_logits_at_num_tokens_seen(
        model_name: str, 
        checkpoint_index: int, 
        device:torch.device, 
        input_tokens: Any
    ) -> Tuple[int, Any]:    
    tokens = input_tokens.to(device)
    model = HookedTransformer.from_pretrained(model_name, checkpoint_index=checkpoint_index, device=device)
    num_tokens_seen = model.cfg.checkpoint_value
    logits = model(tokens, return_type="logits") # TODO: Do I have to take this into it's own device??
    del model
    return num_tokens_seen, logits

LogitMetric = Callable[[str, int, torch.device, Any], Tuple[int, Any]]
WeightMetric = Callable[[str, int, torch.device], Tuple[int, Any]]
ActivationMetric = Callable[[str, int, torch.device, Any], Tuple[int, Any]]

from typing import Callable, Union

# TODO: Parallelize this
# TODO: Generalize this to take in a class for what kind of data you are extracting
def iterate_models_and_checkpoints(config: TrainingDynamicsConfig, metric: Union[LogitMetric, WeightMetric, ActivationMetric]) -> Dict[str, Dict[int, Any]]:
    if config.num_devices == 0:
        device = 'cpu'
    elif config.num_devices == 1:
        device = 'cuda'
    else:
        raise NotImplementedError("We have yet to implement parallelization for this")

    # TODO: load dataloader
    # TODO: I'm not crazy about this whole evals thing for the dataloader....
    # TODO: Also, not crazy about this whole loading in the model to use the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    pile_dataloader = evals.make_pile_data_loader(tokenizer=tokenizer, batch_size=config.data_loader_batch_size) # TODO: Hmm, this may actually be a bug in the experiment. Like, if we're just doing this this is bad.
    tokens = next(iter(pile_dataloader))['tokens'][:,:config.num_tokens_predicted]

    model_name_checkpoint_idx_tuples = [(model_name, checkpoint_idx) for model_name in config.model_names for checkpoint_idx in config.checkpoint_indices]
    out = {model_name: dict() for model_name in config.model_names}
    for model_name, index in model_name_checkpoint_idx_tuples:
        num_tokens_seen, logits = gather_logits_at_num_tokens_seen(model_name, index, device, tokens)
        out[model_name][num_tokens_seen] = logits
    return out

# Then I need to reshape things (For reasons which I don't quite understand)
# Then I need to inpca the model predictions

# And then I need to create the visualization