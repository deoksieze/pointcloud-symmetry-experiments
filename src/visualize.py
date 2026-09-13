"""Visualize synthetic point clouds: one example per generator.

Run:
    python -m src.visualize
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.data import (
    ALL_GENERATORS, generate_sample, center_points, normalize_points
)

OUTPUT_DIR = "outputs/figures"


def plot_single_cloud(ax, points, title, label, gen_name):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=2, alpha=0.6, c="steelblue", edgecolors="none")
    ax.set_title(f"{title}\nlabel={label}, gen={gen_name}", fontsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    lim = 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])


def visualize_all_generators(num_points=256, seed=42):
    gen_names = list(ALL_GENERATORS.keys())
    n = len(gen_names)
    fig = plt.figure(figsize=(4 * n, 4))
    for i, name in enumerate(gen_names):
        pts, lbl, _ = generate_sample(num_points=num_points, seed=seed,
                                       generator_name=name)
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        plot_single_cloud(ax, pts, name, lbl, name)
    plt.tight_layout()
    return fig


def visualize_rotation_example(num_points=256, seed=100):
    """Show one shape with and without random rotation."""
    pts_canon, lbl, name = generate_sample(num_points=num_points, seed=seed,
                                           apply_rotation=False)
    pts_rot, _, _ = generate_sample(num_points=num_points, seed=seed,
                                    apply_rotation=True)
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_single_cloud(ax1, pts_canon, f"{name} (canonical)", lbl, name)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_single_cloud(ax2, pts_rot, f"{name} (rotated)", lbl, name)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Figure 1: one example per generator
    fig1 = visualize_all_generators()
    path1 = os.path.join(OUTPUT_DIR, "generators_overview.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"Saved: {path1}")

    # Figure 2: canonical vs rotated
    fig2 = visualize_rotation_example()
    path2 = os.path.join(OUTPUT_DIR, "rotation_example.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {path2}")
