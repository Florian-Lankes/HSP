"""Small plotting helpers for HSP training notebooks."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class LossPlotter:
    def plot(self, losses: Sequence[float]) -> tuple[Figure, Axes]:
        figure = Figure()
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        axes.plot(losses)
        axes.set_xlabel("Training step")
        axes.set_ylabel("VAE loss")
        axes.set_title("Training loss")
        axes.grid()
        return figure, axes
