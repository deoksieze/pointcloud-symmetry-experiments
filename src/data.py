"""Synthetic point-cloud generation, normalization, rotation, and dataset.

Shapes:
    - ring: points sampled on a circle in a random plane
    - cylinder: points on the lateral surface of a cylinder
    - cube: points on the surface of a unit cube
    - L-shape: union of two axis-aligned boxes (asymmetric)
    - random_clusters: several random Gaussian blobs (asymmetric)

Each sample is returned as float32 [N, 3] with an integer label.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

GLOBAL_SEED = 42


def set_seed(seed: int = GLOBAL_SEED):
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def random_rotation_matrix(rng: np.random.RandomState) -> np.ndarray:
    """Random 3x3 rotation matrix via QR decomposition of a random Gaussian matrix."""
    Z = rng.randn(3, 3)
    Q, R = np.linalg.qr(Z)
    D = np.sign(np.diag(R))
    D[D == 0] = 1.0
    R = Q * D
    return R.astype(np.float32)


def rotate_points(points: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply rotation matrix R [3,3] to points [N,3]."""
    return (R @ points.T).T


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def center_points(points: np.ndarray) -> np.ndarray:
    """Translate centroid to the origin."""
    return points - points.mean(axis=0)


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Scale so that the maximum point norm is 1."""
    norms = np.linalg.norm(points, axis=1)
    max_norm = norms.max()
    if max_norm > 0:
        return points / max_norm
    return points


def add_jitter(points: np.ndarray, sigma: float = 0.01,
               rng: np.random.RandomState = None) -> np.ndarray:
    """Add isotropic Gaussian noise."""
    if rng is None:
        rng = np.random.RandomState()
    return points + rng.randn(*points.shape).astype(np.float32) * sigma


# ---------------------------------------------------------------------------
# Shape generators — each returns [N, 3]
# ---------------------------------------------------------------------------

def generate_ring(num_points: int, rng: np.random.RandomState) -> np.ndarray:
    """Points on a circle of radius 1, lying in a random plane."""
    theta = rng.uniform(0, 2 * np.pi, num_points)
    ring = np.stack([np.cos(theta), np.sin(theta), np.zeros(num_points)], axis=1)
    R = random_rotation_matrix(rng)
    return rotate_points(ring, R)


def generate_cylinder(num_points: int, rng: np.random.RandomState) -> np.ndarray:
    """Points on the lateral surface of a cylinder (radius 0.5, height 1)."""
    theta = rng.uniform(0, 2 * np.pi, num_points)
    z = rng.uniform(-0.5, 0.5, num_points)
    r = 0.5
    cylinder = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    return cylinder


def generate_cube(num_points: int, rng: np.random.RandomState) -> np.ndarray:
    """Points uniformly sampled on the surface of a unit cube [-0.5, 0.5]^3."""
    # Each point lands on one of 6 faces
    faces = rng.randint(0, 6, num_points)
    pts = np.zeros((num_points, 3), dtype=np.float64)
    for i in range(num_points):
        face = faces[i]
        u, v = rng.uniform(-0.5, 0.5, 2)
        if face == 0:    # +x
            pts[i] = [0.5, u, v]
        elif face == 1:  # -x
            pts[i] = [-0.5, u, v]
        elif face == 2:  # +y
            pts[i] = [u, 0.5, v]
        elif face == 3:  # -y
            pts[i] = [u, -0.5, v]
        elif face == 4:  # +z
            pts[i] = [u, v, 0.5]
        else:             # -z
            pts[i] = [u, v, -0.5]
    return pts


def generate_L_shape(num_points: int, rng: np.random.RandomState) -> np.ndarray:
    """Union of two boxes forming an L (asymmetric)."""
    half_n = num_points // 2
    rem = num_points - half_n
    # vertical bar
    bar = np.stack([
        rng.uniform(-0.1, 0.1, half_n),
        rng.uniform(-0.5, 0.5, half_n),
        rng.uniform(0.0, 1.0, half_n),
    ], axis=1)
    # horizontal foot
    foot = np.stack([
        rng.uniform(-0.1, 0.1, rem),
        rng.uniform(-0.5, 0.5, rem),
        rng.uniform(-0.5, 0.0, rem),
    ], axis=1)
    return np.concatenate([bar, foot], axis=0)


def generate_random_clusters(num_points: int, rng: np.random.RandomState,
                             n_clusters: int = 4) -> np.ndarray:
    """Several Gaussian blobs at random centres (asymmetric baseline)."""
    centres = rng.randn(n_clusters, 3) * 1.5
    labels_per_cluster = np.full(num_points, 0, dtype=int)
    counts = np.diff(np.round(np.linspace(0, num_points, n_clusters + 1)).astype(int))
    parts = []
    for i, c in enumerate(centres):
        pts = rng.randn(counts[i], 3) * 0.15 + c
        parts.append(pts)
    remaining = num_points - sum(p.shape[0] for p in parts)
    if remaining > 0:
        parts.append(rng.randn(remaining, 3) * 0.15 + centres[0])
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Mapping from label → generator (symmetric: 1, asymmetric: 0)
# ---------------------------------------------------------------------------

SYMMETRIC_GENERATORS = {
    "ring": generate_ring,
    "cylinder": generate_cylinder,
    "cube": generate_cube,
}

ASYMMETRIC_GENERATORS = {
    "L_shape": generate_L_shape,
    "random_clusters": generate_random_clusters,
}

ALL_GENERATORS = {**SYMMETRIC_GENERATORS, **ASYMMETRIC_GENERATORS}
LABEL_MAP = {name: (1 if name in SYMMETRIC_GENERATORS else 0)
             for name in ALL_GENERATORS}


# ---------------------------------------------------------------------------
# High-level sample generator
# ---------------------------------------------------------------------------

def generate_sample(
    num_points: int = 256,
    apply_rotation: bool = False,
    jitter_sigma: float = 0.01,
    seed: int = None,
    generator_name: str = None,
) -> Tuple[np.ndarray, int, str]:
    """Generate one point cloud.

    Returns:
        points : float32 [N, 3]
        label  : int  (0 = asymmetric, 1 = rotationally symmetric)
        name   : str  generator name (for tracking)
    """
    rng = np.random.RandomState(seed)

    if generator_name is None:
        generator_name = rng.choice(list(ALL_GENERATORS.keys()))
    gen_fn = ALL_GENERATORS[generator_name]
    points = gen_fn(num_points, rng)

    points = center_points(points)
    points = normalize_points(points)

    if apply_rotation:
        R = random_rotation_matrix(rng)
        points = rotate_points(points, R)

    points = add_jitter(points, sigma=jitter_sigma, rng=rng)
    points = points.astype(np.float32)

    label = LABEL_MAP[generator_name]
    return points, label, generator_name


def generate_dataset(
    num_samples: int = 1000,
    num_points: int = 256,
    apply_rotation: bool = False,
    jitter_sigma: float = 0.01,
    seed: int = GLOBAL_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a full dataset.

    Returns:
        points      : float32 [N_samples, N_points, 3]
        labels      : int64   [N_samples]
        generator_names : str array [N_samples]
    """
    rng = np.random.RandomState(seed)
    sym_names = list(SYMMETRIC_GENERATORS.keys())
    asym_names = list(ASYMMETRIC_GENERATORS.keys())
    all_points = []
    all_labels = []
    all_gen_names = []

    for i in range(num_samples):
        # Alternate labels to keep classes perfectly balanced.
        label = i % 2
        pool = sym_names if label == 1 else asym_names
        name = pool[rng.randint(0, len(pool))]
        s = rng.randint(0, 1_000_000)
        pts, lbl, _ = generate_sample(
            num_points=num_points,
            apply_rotation=apply_rotation,
            jitter_sigma=jitter_sigma,
            seed=s,
            generator_name=name,
        )
        all_points.append(pts)
        all_labels.append(lbl)
        all_gen_names.append(name)

    return (
        np.stack(all_points),
        np.array(all_labels, dtype=np.int64),
        np.array(all_gen_names),
    )


class PointCloudDataset(Dataset):
    """PyTorch Dataset over pre-generated synthetic point clouds.

    Base point clouds are generated once in __init__ (canonical orientation).
    If apply_rotation is True, each call to __getitem__ applies a fresh random
    rotation, which acts as augmentation during training.
    """

    def __init__(self, num_samples: int = 1000, num_points: int = 256,
                 apply_rotation: bool = False, jitter_sigma: float = 0.01,
                 seed: int = GLOBAL_SEED):
        self.points, self.labels, self.names = generate_dataset(
            num_samples=num_samples,
            num_points=num_points,
            apply_rotation=False,  # base cloud is canonical
            jitter_sigma=jitter_sigma,
            seed=seed,
        )
        self.apply_rotation = apply_rotation

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pts = self.points[idx].copy()  # copy: we must not mutate shared array
        if self.apply_rotation:
            R = random_rotation_matrix(np.random)
            pts = rotate_points(pts, R)
        return torch.from_numpy(pts), torch.tensor(int(self.labels[idx]))


if __name__ == "__main__":
    pts, lbl, name = generate_sample(seed=0)
    print(f"Sample: generator={name}, label={lbl}, shape={pts.shape}, "
          f"norm range=[{np.linalg.norm(pts, axis=1).min():.3f}, "
          f"{np.linalg.norm(pts, axis=1).max():.3f}]")
