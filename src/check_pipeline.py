"""Phase 3 sanity checks: Dataset/DataLoader, forward pass, permutation invariance."""
import torch
from torch.utils.data import DataLoader

from src.data import PointCloudDataset
from src.model import TinyPointNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 1. Dataset + DataLoader produce [B, N, 3] and [B]
    ds = PointCloudDataset(num_samples=64, num_points=256, apply_rotation=False)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    x, y = next(iter(loader))
    print(f"batch x: {tuple(x.shape)}, dtype={x.dtype}")
    print(f"batch y: {tuple(y.shape)}, dtype={y.dtype}")
    assert x.shape == (8, 256, 3) and x.dtype == torch.float32
    assert y.shape == (8,) and y.dtype == torch.int64

    # 2. Forward pass shapes
    model = TinyPointNet(num_classes=2, embedding_dim=256).to(device)
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        logits, embedding = model(x, return_embedding=True)
    print(f"logits:   {tuple(logits.shape)}")
    print(f"embedding:{tuple(embedding.shape)}")
    assert logits.shape == (8, 2)
    assert embedding.shape == (8, 256)

    # 3. Permutation invariance: logits must not depend on point order
    perm = torch.randperm(x.shape[1], device=device)
    x_perm = x[:, perm, :]
    with torch.no_grad():
        logits_perm, _ = model(x_perm, return_embedding=True)
    diff = (logits - logits_perm).abs().max().item()
    print(f"max |logit difference| after permutation: {diff:.2e}")
    assert diff < 1e-4, "permutation invariance violated"
    print("\nALL PHASE 3 CHECKS PASSED")


if __name__ == "__main__":
    main()