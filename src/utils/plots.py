"""Plotting utilities."""

from typing import Dict
import matplotlib.pyplot as plt
import plotly.io as pio
import persim
from IPython.display import HTML
import plotly.graph_objects as go
from keras.src.models.sequential import Sequential


def plot_side_by_side(fig0, fig1):
    """Wrapper function to plot two plotly figures side by side."""
    html0 = pio.to_html(fig0, full_html=False, include_plotlyjs="cdn")
    html1 = pio.to_html(fig1, full_html=False, include_plotlyjs=False)

    html = f"""
        <div style="display: flex; flex-direction: row;">
            <div style="width: 50%">{html0}</div>
            <div style="width: 50%">{html1}</div>
        </div>
    """
    return HTML(html)


def plot_training_data(dcls: Dict) -> None:
    fig = go.Figure()

    for cs in dcls:
        fig.add_trace(
            go.Scatter(
                x=dcls[cs]["td"][:,0],
                y=dcls[cs]["td"][:,1],
                mode='markers',
                name=f"class_{cs}",
                marker_color=dcls[cs]["col"],
                marker_size=1.5)
            )

    fig.update_layout(
        title="Training data",
        width=400,
        height=400,
        xaxis_range=[-4.5,4.5],
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        template="plotly_dark"
    )

    fig.show()


def plot_data_shape_evolution(dcls: Dict, cs: str, model: Sequential) -> None:
    """Plot how data cloud evolves layer by layer under the neural network action."""
    num_layers = len(model.layers)-1

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dcls[cs]["td"][:,0],
            y=dcls[cs]["td"][:,1],
            mode='markers',
            marker_color=dcls[cs]["col"],
            marker_size=1.5,
            visible=True
        )
    )

    for i in range(num_layers):
        fig.add_trace(
            go.Scatter(
                x=dcls[cs]["pttd"][i,:,0],
                y=dcls[cs]["pttd"][i,:,1],
                mode='markers',
                marker_color=dcls[cs]["col"],
                marker_size=1.5,
                visible=False
            )
        )

    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],  # layout attribute
            label="Training data" if i==0 else str(i)
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Layer: "},
        pad={"t": 50},
        steps=steps,
        tickwidth=1,
        tickcolor="white"
    )]

    fig.update_layout(
        width=400,
        height=450,
        title=f"Evolution of True class {cs}",
        template="plotly_dark",
        sliders=sliders
    )

    return fig


def plot_classification(dcls: Dict, cs: str) -> None:
    """Colour the binary classification that the sigmoid final layer applies."""
    fig = go.Figure()

    # Prediction A
    subarray_A = dcls[cs]["pttd"][-1][dcls[cs]["out"]==0]
    fig.add_trace(
        go.Scatter(
            x=subarray_A[:,0],
            y=subarray_A[:,1],
            mode='markers',
            marker_color="red",
            marker_size=1.5,
            name="Predicted A"
        )
    )

    # Predicted B
    subarray_B = dcls[cs]["pttd"][-1][dcls[cs]["out"]==1]
    fig.add_trace(
        go.Scatter(
            x=subarray_B[:,0],
            y=subarray_B[:,1],
            mode='markers',
            marker_color="green",
            marker_size=1.5,
            name="Predicted B"
        )
    )

    fig.update_layout(
        width=400,
        height=400,
        title=f"Classification of True class {cs}",
        template="plotly_dark"
    )

    return fig


def plot_class(dcls: Dict, cs: str) -> None:
    """Plot the training data cloud for a given class."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dcls[cs]["td"][:,0],
            y=dcls[cs]["td"][:,1],
            mode='markers',
            name=f"class_{cs}",
            marker_color=dcls[cs]["col"],
            marker_size=1.5)
        )

    fig.update_layout(
        title=f"Class {cs} training data",
        width=400,
        height=400,
        xaxis_range=[-4.5,4.5],
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        template="plotly_dark"
    )

    return fig

    

def plot_increasing_disks(dcls: Dict, cs: str) -> None:
    """Plot evolution of data cloud as we increase disk size around each point."""
    fig = go.Figure()

    for i in range(11):
        fig.add_trace(
            go.Scatter(
                x=dcls[cs]["td"][:,0],
                y=dcls[cs]["td"][:,1],
                mode='markers',
                marker_color=dcls[cs]["col"],
                marker_size=2 + i,
                visible=i==0  # Make only the first trace visible
            )
        )

    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],  # layout attribute
            label=1+0.5*i
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Disk size: "},
        pad={"t": 50},
        ticklen=0,
        steps=steps
    )]

    fig.update_layout(
        width=400,
        height=450,
        title=f"Class {cs} shape persistance",
        sliders=sliders,
        template="plotly_dark"
    )

    return fig


def plot_persistent_homology(dcls: Dict, cs: str) -> None:
    td = dcls[cs]["td"]
    td_diagrams = dcls[cs]["td_diagrams"]
    pttd = dcls[cs]["pttd"][-1]
    pttd_diagrams = dcls[cs]["last_pttd_diagrams"]

    _, axes = plt.subplots(nrows=2, ncols=2)

    # Row 0: Training data
    axes[0, 0].set_title("Training data")
    axes[0, 1].set_title("Persistent homology (training)")
    axes[0, 0].scatter(td[:,0], td[:,1], marker=".", s=10, alpha=0.5)
    persim.plot_diagrams(td_diagrams, ax=axes[0, 1])

    # Row 1: Transformed data
    axes[1, 0].set_title("Transformed data")
    axes[1, 1].set_title("Persistent homology (transformed)")
    axes[1, 0].scatter(pttd[:,0], pttd[:,1], marker=".", s=10, alpha=0.5)
    persim.plot_diagrams(pttd_diagrams, ax=axes[1, 1])

    plt.tight_layout()





