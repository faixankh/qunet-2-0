from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

def plot_training_curves(history: dict, path: str | Path):
    path = Path(path)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    for key, values in history.items():
        ax.plot(values, label=key)
    ax.legend()
    ax.set_title("Training Curves")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
