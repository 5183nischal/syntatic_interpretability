import plotly.graph_objects as go
import webcolors
from typing import List
from dataclasses import dataclass

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
