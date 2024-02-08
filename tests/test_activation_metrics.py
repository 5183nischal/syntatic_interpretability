from datasets import load_dataset
from pydantic import BaseModel
from transformer_lens import HookedTransformer
from typing import List, Literal

from syntactic_interpretability.metrics.activations_metrics import Extracted, activation_effective_dimension

# TODO: Sample rather than iterate in prep_dataset

def prep_dataset(dataset_name: str, dataset_split: str, num_samples: int, max_token_len: int, model_name: str):
    dataset = load_dataset(dataset_name)
    _tokenizer = HookedTransformer.from_pretrained(model_name).tokenizer
    
    out = []
    for i in range(num_samples):
        text = dataset[dataset_split][i]['text']
        tokens = _tokenizer(text)['input_ids']
        out_str = _tokenizer.decode(tokens[:max_token_len])
        out.append(out_str)
    return out

Module = Literal['attn_out', 'mlp_out']

def save_activations_as_jsonl(foo: List[Extracted]):
    raise NotImplementedError

if __name__ == "__main__":

    class ActivationMetricsConfig(BaseModel):
        # Model
        model_name: str

        # Dataset
        dataset_name: str
        dataset_split: str
        num_samples: int
        max_token_len: int



    dataset = load_dataset("NeelNanda/pile-10k")
    _tokenizer = HookedTransformer.from_pretrained('pythia-160m').tokenizer
    
    out = []
    for i in range(2):
        text = dataset['train'][i]['text']
        tokens = _tokenizer(text)['input_ids']
        out_str = _tokenizer.decode(tokens[:20])
        out.append(out_str)
    
    del _tokenizer
    
    model = HookedTransformer.from_pretrained('pythia-160m')
    extracted = [Extracted.from_model(data, model, True, ['attn_out', 'mlp_out']) for data in out]
    measured = [activation_effective_dimension(x.activations) for x in extracted]
    # Then we make plots w/ Line
    print('based')