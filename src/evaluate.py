"""Embedding analysis: PCA visualization and cosine-similarity of rotations.

Loads a trained checkpoint, extracts global embeddings from a held-out set,
saves a PCA scatter plot, and measures how stable embeddings are under random
rotations (cosine similarity). All results are reported as exploratory.

Run:
    python -m src.evaluate --experiment canonical --device auto
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import (
    PointCloudDataset, random_rotation_matrix, rotate_points,
)
from src.model import TinyPointNet

OUTPUTS_DIR = "outputs"

# marker shape per generator for the PCA plot
GENERATOR_MARKERS = {
    "ring": "o",
    "cylinder": "s",
    "cube": "^",
    "L_shape": "v",
    "random_clusters": "D",
}
LABEL_COLORS = {0: "tab:red", 1: "tab:blue"}


def get_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TinyPointNet(num_classes=2,
                         embedding_dim=ckpt["config"]["embedding_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["config"]


@torch.no_grad()
def extract_embeddings(model, dataset, device, batch_size=64):
    """Return (embeddings [S, D], labels [S], generator names [S])."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_emb, all_labels = [], []
    for x, _ in loader:
        emb = model(x.to(device), return_embedding=True)[1]
        all_emb.append(emb.cpu().numpy())
    emb = np.concatenate(all_emb, axis=0)
    return emb, dataset.labels, dataset.names


def pca_plot(embeddings, labels, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    xy = PCA(n_components=2, random_state=0).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in set(names):
        mask = names == name
        label = int(labels[mask][0])
        ax.scatter(xy[mask, 0], xy[mask, 1], s=14, alpha=0.7,
                   c=LABEL_COLORS[label], marker=GENERATOR_MARKERS[name],
                   label=f"{name} (class {label})")
    ax.set_title("PCA of global embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@torch.no_grad()
def rotation_cosine_analysis(model, dataset, device,
                             num_pairs=5, num_rotations=6):
    """Compare embeddings of a point cloud and several rotated copies."""
    idxs = np.linspace(0, len(dataset) - 1, num_pairs).astype(int)
    results = {}
    all_sims = []
    for i in idxs:
        pts, label = dataset[i]
        pts_np = pts.numpy()
        emb_orig = model(pts[None].to(device),
                         return_embedding=True)[1][0].cpu().numpy()
        cosines = []
        for k in range(num_rotations):
            R = random_rotation_matrix(np.random)
            pts_rot = rotate_points(pts_np.copy(), R)
            emb_rot = model(torch.from_numpy(pts_rot)[None].to(device),
                            return_embedding=True)[1][0].cpu().numpy()
            cosines.append(cosine_similarity(emb_orig, emb_rot))
        results[str(i)] = {
            "generator": str(dataset.names[i]),
            "label": int(label),
            "cosines": [round(c, 4) for c in cosines],
            "mean": round(float(np.mean(cosines)), 4),
            "min": round(float(np.min(cosines)), 4),
        }
        all_sims.extend(cosines)
    return results, all_sims


def main():
    parser = argparse.ArgumentParser(description="Embedding analysis")
    parser.add_argument("--experiment", default="canonical")
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--num-pairs", type=int, default=5)
    parser.add_argument("--num-rotations", type=int, default=6)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"device: {device} | experiment: {args.experiment}")

    ckpt_path = os.path.join(OUTPUTS_DIR, "checkpoints", f"{args.experiment}.pt")
    assert os.path.exists(ckpt_path), f"missing checkpoint: {ckpt_path}"
    model, cfg = load_model(ckpt_path, device)
    print(f"loaded checkpoint (embedding_dim={cfg['embedding_dim']})")

    # held-out canonical evaluation set (never rotated)
    ds = PointCloudDataset(num_samples=args.num_samples, num_points=256,
                           apply_rotation=False, seed=cfg["seed"] + 3000)

    embeddings, labels, names = extract_embeddings(model, ds, device)
    print(f"embeddings: {embeddings.shape}")

    os.makedirs(os.path.join(OUTPUTS_DIR, "figures"), exist_ok=True)
    pca_path = os.path.join(OUTPUTS_DIR, "figures",
                            f"embeddings_pca_{args.experiment}.png")
    pca_plot(embeddings, labels, names, pca_path)
    print(f"saved PCA plot: {pca_path}")

    rot_results, all_sims = rotation_cosine_analysis(
        model, ds, device, num_pairs=args.num_pairs,
        num_rotations=args.num_rotations,
    )
    summary = {
        "experiment": args.experiment,
        "pca_figure": pca_path,
        "per_sample": rot_results,
        "overall": {
            "num_pairs": args.num_pairs,
            "num_rotations": args.num_rotations,
            "mean_cosine": round(float(np.mean(all_sims)), 4),
            "min_cosine": round(float(np.min(all_sims)), 4),
        },
        "note": "Exploratory diagnostics only; not evidence of rotation invariance.",
    }
    os.makedirs(os.path.join(OUTPUTS_DIR, "metrics"), exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, "metrics",
                            f"cosine_similarity_{args.experiment}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved cosine results: {out_path}")

    print("\nCosine similarity of embeddings vs rotated copies:")
    for i, r in rot_results.items():
        print(f"  sample {i:>3} [{r['generator']:<15} class {r['label']}] "
              f"mean={r['mean']:.4f} min={r['min']:.4f}")
    print(f"\noverall: mean={summary['overall']['mean_cosine']} "
          f"min={summary['overall']['min_cosine']}")


if __name__ == "__main__":
    main()