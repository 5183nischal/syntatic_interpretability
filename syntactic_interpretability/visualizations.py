import plotly.graph_objects as go
import webcolors
from typing import List, Literal
from dataclasses import dataclass
import torch

from syntactic_interpretability.metrics.weight_metrics import AttnCircuitMeasurements

# TODO: Move the cicuit type upper & lower bound logic into the metric class
# TODO: Once metrics are optional, Depending on circuit, validate all metrics are present
# TODO: plot_attn_circuit_measurements :: Dict[str, List[AttnCircuitMeasurements]] -> Literal['qk', 'ov'] -> go.Figure

@dataclass
class Line:
    name: str
    color: str
    x: List[float]
    y: List[float]
    upper_bound: List[float]
    lower_bound: List[float]

    def __post_init__(self):
        # Check that the color is valid and re-init it
        self.color = webcolors.name_to_rgb(self.color)
        
        # Check that all lenghts are the same
        lengths = [len(self.x), len(self.y), len(self.upper_bound), len(self.lower_bound)]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All lists in Line must have the same length")
    
    @classmethod
    def from_attn_circuit_measurement(circuits: List[AttnCircuitMeasurements], color: str, circuit_type: Literal['qk', 'ov']) -> "Line":
        assert all([circuit.model_name == circuits[0].model_name for circuit in circuits]), "Circuit measurements must come from the same model"

        def _tensor_min(tensor: torch.Tensor) -> float:
          flattened_tensor = torch.flatten(tensor)
          return torch.min(flattened_tensor).item()
            
        def _tensor_max(tensor: torch.Tensor) -> float:
          flattened_tensor = torch.flatten(tensor)
          return torch.max(flattened_tensor).item()

        if circuit_type == 'qk':
          ys = [circuit.qk_reduced for circuit in circuits]
          lower_bound = [_tensor_min(circuit.qk) for circuit in circuits]
          upper_bound = [_tensor_max(circuit.qk) for circuit in circuits]
        elif circuit_type == 'ov':
          ys = [circuit.ov_reduced for circuit in circuits]
          lower_bound = [_tensor_min(circuit.ov) for circuit in circuits]
          upper_bound = [_tensor_max(circuit.ov) for circuit in circuits]
        else:
           raise ValueError(f"circuit type must be either 'qk' or 'ov': {circuit_type}")
        
        return Line(
            name=circuits[0].model_name,
            color=color,
            x=[circuit.num_tokens_seen for circuit in circuits],
            y=ys,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )
    

def continuous_error_boundary_plot(
    lines: List[Line], 
    title: str,
    x_axis_title: str,
    y_axis_title: str
) -> go.Figure:
    fig = go.Figure()
    for line in lines:
        # Add the main line
        fig.add_trace(go.Scatter(x=line.x, y=line.y, mode='lines', line=dict(color=f'rgb({line.color[0]}, {line.color[1]}, {line.color[2]})'), name=line.name))

        # Add the upper boundary
        fig.add_trace(go.Scatter(x=line.x, y=line.upper_bound, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))

        # Add the lower boundary
        fig.add_trace(go.Scatter(x=line.x, y=line.lower_bound, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))

        # Add the shaded region between upper and lower boundaries
        fig.add_trace(go.Scatter(x=np.concatenate([line.x, line.x[::-1]]),
                                 y=np.concatenate([line.upper_bound, line.lower_bound[::-1]]),
                                 fill='toself', fillcolor=f'rgba({line.color[0]}, {line.color[1]}, {line.color[2]}, 0.2)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 showlegend=False))
    # Update layout
    fig.update_layout(title=title, xaxis_title=x_axis_title, yaxis_title=y_axis_title)
    
    return fig

if __name__ == "__main__":
  import numpy as np
  lines = [
      Line(name='Sin Line', color='red', x=np.linspace(0, 10, 100), y=np.sin(np.linspace(0, 10, 100)),
          upper_bound=np.sin(np.linspace(0, 10, 100)) + 0.2, lower_bound=np.sin(np.linspace(0, 10, 100)) - 0.2),
      Line(name='Cos Line', color='blue', x=np.linspace(0, 10, 100), y=np.cos(np.linspace(0, 10, 100)),
          upper_bound=np.cos(np.linspace(0, 10, 100)) + 0.1, lower_bound=np.cos(np.linspace(0, 10, 100)) - 0.1),
  ]

  fig = continuous_error_boundary_plot(lines, 'the graphy graph', 'foo', 'bar')
  fig.show()
