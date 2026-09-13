"""Sanity checks for synthetic geometry (Phase 2)."""
import sys
import numpy as np

sys.path.insert(0, ".")
from src.data import (
    ALL_GENERATORS, LABEL_MAP, generate_dataset, generate_sample,
)

def main():
    # 1. Each generator produces [N, 3] finite points with a valid label
    for name, gen_fn in ALL_GENERATORS.items():
        rng = np.random.RandomState(0)
        pts = gen_fn(256, rng)
        assert pts.shape == (256, 3), f"{name}: bad shape {pts.shape}"
        assert np.all(np.isfinite(pts)), f"{name}: non-finite points"
        assert LABEL_MAP[name] in (0, 1), f"{name}: bad label map"
        print(f"generator {name:<15} shape={pts.shape}  label={LABEL_MAP[name]}")

    # 2. generate_sample returns float32 [N,3] centered and max-norm ~1
    pts, lbl, name = generate_sample(seed=0)
    assert pts.shape == (256, 3), pts.shape
    assert pts.dtype == np.float32, pts.dtype
    norm = np.linalg.norm(pts, axis=1)
    print(f"\nsample: generator={name}, label={lbl}, dtype={pts.dtype}, "
          f"max_norm={norm.max():.4f}")
    assert abs(norm.max() - 1.0) < 0.05, f"normalization broke, max_norm={norm.max()}"
    assert abs(pts.mean(axis=0)).max() < 0.05, "points not centered"

    # 3. generate_dataset: balanced classes, correct shapes
    points, labels, gen_names = generate_dataset(num_samples=40, num_points=256)
    print(f"\ndataset: points={points.shape} dtype={points.dtype}, "
          f"labels={labels.shape} dtype={labels.dtype}")
    assert points.shape == (40, 256, 3)
    assert labels.dtype == np.int64
    assert points.dtype == np.float32
    dist = np.bincount(labels)
    print(f"class distribution: 0 (asymmetric)={dist[0]}, 1 (symmetric)={dist[1]}")
    assert len(dist) == 2 and dist[0] == dist[1], "classes not balanced"

    print("\nALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()