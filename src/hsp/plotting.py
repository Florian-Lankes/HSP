"""Small plotting helpers for HSP training notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class LossPlotter:
    def plot(self, losses: Mapping[str, Sequence[float]]) -> tuple[Figure, list[Axes]]:
        figure, axes = plt.subplots(1, len(losses), figsize=(6 * len(losses), 4))
        axes_list = list(axes) if len(losses) > 1 else [axes]
        for axis, (label, values) in zip(axes_list, losses.items(), strict=True):
            self._plot_loss(axis, values, title=f"{label} loss")
        figure.tight_layout()
        plt.close(figure)
        return figure, axes_list

    @staticmethod
    def _plot_loss(
        axis: Axes,
        losses: Sequence[float],
        *,
        title: str,
        ylabel: str = "Loss",
    ) -> None:
        axis.plot(losses)
        axis.set_xlabel("Training step")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid()