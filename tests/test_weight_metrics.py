from transformer_lens import HookedTransformer

from syntactic_interpretability.metrics.weight_metrics import AttnCircuitsMeasurements
from syntactic_interpretability.visualizations import GraphLine

# TODO: Test all combinations of metrics
# TODO: Test intermittent persistence

if __name__ == "__main__":
    from pydantic import BaseModel
    from typing import List, Dict
    import yaml

    from syntactic_interpretability.metrics.weight_metrics import AttnCircuit, Metric
    from syntactic_interpretability.visualizations import plot_attn_circuit_measurements

    class AttnCircuitConfig(BaseModel):
        model_name_color_dict: Dict[str, str]
        checkpoint_index: List[int]
        attn_circuit: AttnCircuit
        metric: Metric

    model_name_color_dict = {'gpt2': 'blue', 'pythia-160m': 'red'}

    out = dict(red=[])
    for model_name in ['pythia-160m']:
        for i in [0, -1]:
            model = HookedTransformer.from_pretrained(model_name, checkpoint_index=i, device='cpu')
            out[model_name_color_dict.get(model_name)].append(AttnCircuitsMeasurements.from_model(model, ['ov']))
            del model

    plot_attn_circuit_measurements(
        out, 
        'ov', 
        'effective_dimension',
        'Test',
        'checkpoint_value',
        'ef_dim'
    )
