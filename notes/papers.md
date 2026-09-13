# Reading notes

## References to follow up on

- **Mitra, N. J., Guibas, L. J., & Pauly, M. (2006).** *Partial and Approximate Symmetry Detection for 3D Geometry.* SIGGRAPH 2006.
  - Algorithm outline: find candidate transformations between locally similar point samples, cluster transformations in transformation space, verify consistent surface patches.
  - This prototype does not reproduce it; it only prepares point-cloud basics before studying partial/approximate symmetry.

- **Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017).** *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.*
  - Shared per-point MLP + symmetric aggregation (max pooling) → permutation-invariance.
  - Our `TinyPointNet` is a minimal adaptation of this idea.

- Optional later: PointNet++, equivariant networks (e.g. Vector Neurons), symmetry-aware descriptors.