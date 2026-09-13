# Project brief for OpenCode: Point-cloud symmetry experiments

## Role and working style

You are assisting with a small research-preparation project for a university laboratory interview. Act as a careful engineering collaborator and tutor.

Important working rules:

- Work incrementally: first inspect the repository, then propose a compact plan, then implement one small milestone at a time.
- Do not overengineer the project. This is a 2–3 day educational prototype, not a production ML system.
- Before making non-trivial design changes, briefly explain what will be changed and why.
- Prefer simple, readable PyTorch and NumPy code over abstractions, frameworks, or heavy dependencies.
- Keep all code, comments, variable names, CLI flags, and README text in English.
- Explanations to the user may be in Russian.
- Do not invent experimental results, accuracy values, conclusions, or implementation details. Run code before claiming it works.
- When an experiment fails, report the error or observation honestly and suggest the smallest next debugging step.
- Avoid deleting or overwriting existing work without asking first.
- Do not commit, push, install global packages, or modify remote Git configuration unless explicitly asked.
- Ask for confirmation before downloading a large dataset, adding a heavy dependency, or substantially changing the project scope.

## Project title

**Point-cloud symmetry experiments**

## Motivation

This is a small preparatory project for a research laboratory project tentatively titled:

> Symmetry spectra as universal representations of 3D objects.

The broader laboratory direction combines:

- 3D geometry and point clouds;
- approximate and partial symmetry detection;
- representation learning;
- PointNet-style networks;
- analysis of learned embeddings;
- transfer of geometric structure across object categories.

The classic geometric background is Mitra, Guibas, and Pauly, *Partial and Approximate Symmetry Detection for 3D Geometry* (SIGGRAPH 2006). Their algorithm finds candidate transformations between locally similar points, clusters these transformations, and verifies consistent surface patches.

This repository does **not** aim to reproduce the Mitra algorithm or solve realistic partial approximate symmetry detection. It is a deliberately small, controlled learning experiment to understand:

1. How point clouds are represented and processed in PyTorch.
2. How a PointNet-style architecture builds a permutation-invariant global embedding.
3. How random 3D rotations affect a simple point-cloud classifier.
4. Whether learned embeddings separate deliberately constructed synthetic shapes with different symmetry structure.

## Scope

### In scope

- Synthetic 3D point-cloud generation with NumPy.
- Point clouds represented as arrays/tensors with shape `[N, 3]`.
- PyTorch `Dataset` and `DataLoader`.
- A small PointNet-style classifier implemented from scratch.
- Shared point-wise MLP.
- Global max pooling over points.
- Classification of simple synthetic classes.
- Rotation augmentation experiments.
- Extraction of global embeddings.
- PCA visualization of embeddings.
- Cosine-similarity comparison between an object embedding and its rotated version.
- Reproducible CLI commands, fixed random seeds, concise README, and saved figures/results.

### Explicitly out of scope

- Full reproduction of Mitra et al. (2006).
- Automatic discovery of correspondences between real surface patches.
- Real meshes or ModelNet40 in the initial version.
- PointNet++ or transformer architectures.
- SE(3)/E(3)-equivariant neural networks.
- Contrastive learning in the initial version.
- Custom CUDA kernels, distributed training, Docker, CI, experiment trackers, or large infrastructure.
- Claims about universal representations, real-world symmetry detection, or scientific novelty.

## Core concepts

### Point cloud

A 3D object is represented by a set of points:

\[
X = \{p_1, \ldots, p_N\}, \qquad p_i = (x_i,y_i,z_i).
\]

Code representation:

```text
single sample: [N, 3]
batch:         [B, N, 3]
```

The order of points is arbitrary. A model should therefore be permutation-invariant:

\[
f(p_1,\ldots,p_N)=f(p_{\pi(1)},\ldots,p_{\pi(N)})
\]

for any permutation \(\pi\).

### PointNet-style model

The minimal model should follow this concept:

\[
h_i = \mathrm{MLP}(p_i), \qquad
z = \max_{i=1}^{N} h_i, \qquad
\hat y = \mathrm{Classifier}(z).
\]

- The same MLP is applied independently to every point.
- Max pooling aggregates point features into a global embedding.
- Max pooling makes the global representation independent of point order.
- The embedding can be used for classification and later analysis.

### Symmetry

A symmetry is a transformation that maps an object, or a part of it, to itself or to a corresponding part. For this prototype, labels are synthetic and simplified.

Important distinction:

- `globally symmetric`: the whole constructed shape has a clear reflectional or rotational symmetry.
- `partially symmetric`: a component of the shape repeats symmetrically, but the whole object may not be symmetric.
- `asymmetric`: intentionally lacks the constructed symmetry.

The project must describe these labels as controlled synthetic labels, not as complete mathematical symmetry detection.

## Initial classification task

Start with a binary task:

```text
class 0: asymmetric
class 1: rotationally_symmetric
```

Suggested generators:

| Shape | Label | Purpose |
|---|---:|---|
| Ring / circle in a randomly oriented plane | rotationally_symmetric | Simple rotationally symmetric object |
| Cylinder surface | rotationally_symmetric | 3D rotational symmetry |
| Cube surface | rotationally_symmetric | Reflection and discrete rotational symmetries |
| L-shaped union of boxes or point clusters | asymmetric | Clearly broken symmetry |
| Random multi-cluster cloud | asymmetric | Irregular baseline |

Do not use only one generator per class in the final dataset. Otherwise the network may learn generator-specific artifacts rather than a rough symmetry-related distinction.

### Optional extension after the binary baseline works

Add a third class:

```text
class 2: partially_symmetric
```

Suggested construction:

- Place four similar cylinder-like legs around a vertical axis.
- Add an asymmetric top component, such as a shifted block or one-sided backrest.
- The full shape is not globally rotationally symmetric.
- The legs form a repeated, partially symmetric substructure.

This extension is optional and should only be attempted after data generation, training, evaluation, visualization, and README for the binary baseline work.

## Data generation requirements

For each generated sample:

1. Generate a point cloud with `num_points` points, initially 256.
2. Center it around the origin.
3. Normalize it to unit scale, for example such that the maximum point norm is 1.
4. Optionally apply a random 3D rotation.
5. Optionally add small Gaussian jitter/noise.
6. Return points as `float32` with shape `[N, 3]` and an integer class label.

Use deterministic random number generation where feasible. Store and expose a global seed.

Implement rotation using a valid 3D rotation matrix, for example via random unit quaternions or a simple composition of rotations around x/y/z axes. Explain in README which rotation sampling approach is used and note that it is a prototype.

Avoid accidental label leakage:

- Keep point count the same in all classes.
- Apply comparable normalization to all classes.
- Use comparable noise/jitter across classes.
- Avoid giving one class a consistently different coordinate range or density unless that is an intentional ablation.

## Required experiments

### Experiment A: canonical orientation

- Train without random rotations.
- Test without random rotations.
- Purpose: confirm that the full pipeline can learn the controlled synthetic task.

### Experiment B: rotation augmentation

- Train with random rotations.
- Test with random rotations.
- Purpose: study whether rotation augmentation improves robustness to pose variation.

### Experiment C: rotation shift / negative control

- Train without random rotations.
- Test with random rotations.
- Purpose: show that permutation invariance does not automatically mean rotation invariance.

Do not assume a particular ordering of metrics before running the experiment. Report actual results only.

### Embedding analysis

For a held-out evaluation set:

1. Extract global embeddings before the final classifier.
2. Apply PCA to 2D.
3. Save a scatter plot colored by class.
4. Optionally use different marker shapes for each generator type.
5. Take several individual point clouds, create rotated copies, and compute cosine similarity of their embeddings.
6. Report the result as exploratory, not as a proof of invariance or a research conclusion.

Cosine similarity:

\[
\operatorname{cos}(z_1,z_2) = \frac{z_1^\top z_2}{\|z_1\|_2\|z_2\|_2}.
\]

## Suggested project structure

Use this structure unless the repository already has a better equivalent structure:

```text
pointcloud-symmetry-experiments/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── geometry.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── scripts/
│   ├── run_experiment_a.sh
│   ├── run_experiment_b.sh
│   └── run_experiment_c.sh
├── notebooks/
│   └── exploration.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── checkpoints/
└── notes/
    └── papers.md
```

For a very small prototype it is acceptable to begin with fewer files, for example `src/data.py`, `src/model.py`, `src/train.py`, and `src/visualize.py`. Split files only when it improves clarity.

## Technical choices

### Required dependencies

```txt
torch
numpy
matplotlib
scikit-learn
```

Optional dependency only if useful and available:

```txt
open3d
```

Do not add Open3D if Matplotlib 3D scatter plots are sufficient.

### Environment

The project will be developed in WSL2 Ubuntu on a Windows laptop with an NVIDIA RTX 4070. CUDA support may be available through PyTorch.

The code must work on CPU and optionally use CUDA:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Do not require CUDA for basic correctness. The expected dataset and model are small enough to debug on CPU.

### Training defaults

Use reasonable lightweight defaults, for example:

```text
num_points = 256
batch_size = 32
epochs = 20
learning_rate = 1e-3
optimizer = Adam
loss = CrossEntropyLoss
embedding_dim = 256
```

Expose key parameters through CLI flags where it is simple and useful:

```bash
python -m src.train \
  --experiment rotation_augmented \
  --epochs 20 \
  --num-points 256 \
  --seed 42 \
  --device auto
```

The exact CLI interface may be adjusted, but it should stay simple.

## Minimum PointNet-style architecture

A minimal valid architecture can look like this:

```python
class TinyPointNet(nn.Module):
    def __init__(self, num_classes: int = 2, embedding_dim: int = 256):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, return_embedding: bool = False):
        point_features = self.point_mlp(x)       # [B, N, embedding_dim]
        embedding = point_features.max(dim=1).values  # [B, embedding_dim]
        logits = self.classifier(embedding)      # [B, num_classes]
        if return_embedding:
            return logits, embedding
        return logits
```

This is intentionally a simplified PointNet-style baseline. It does not need T-Net, batch normalization, PointNet++, or a faithful reproduction of the original paper.

## Milestones

Implement and validate in this order.

### Milestone 0: inspect and plan

- Inspect current repository files and environment.
- State a compact implementation plan.
- Do not overwrite existing files blindly.

### Milestone 1: project skeleton

- Create the minimum directory structure.
- Add `.gitignore` for Python, virtual environments, checkpoints, and generated large outputs.
- Add `requirements.txt`.
- Add a minimal README with goal and scope.

### Milestone 2: synthetic geometry

- Implement shape generators.
- Implement centering, normalization, jitter, and rotations.
- Add a script/notebook that saves a 3D figure with examples from every class.
- Validate array shapes, finite values, labels, and normalization.

Success criteria:

- Several visually distinct examples are generated.
- Each sample has shape `[N, 3]`.
- The figure is saved under `outputs/figures/`.

### Milestone 3: data pipeline and model

- Implement `Dataset`/`DataLoader`.
- Implement a minimal PointNet-style network.
- Run a forward pass.
- Add a permutation-invariance sanity check: permuting the input point order should preserve logits in evaluation mode up to floating-point tolerance.

Success criteria:

- A batch has shape `[B, N, 3]`.
- Model logits have shape `[B, num_classes]`.
- Embeddings have shape `[B, embedding_dim]`.
- Permutation sanity check passes.

### Milestone 4: baseline training

- Train Experiment A.
- Save loss and accuracy history in a small JSON or CSV file.
- Save a checkpoint only if useful; do not commit checkpoints by default.
- Save a learning curve figure.

Success criteria:

- Training completes without errors.
- Metrics and a figure are saved.
- Results are described honestly.

### Milestone 5: rotation experiments

- Implement Experiments B and C.
- Ensure that the only intended difference is the rotation policy.
- Save a compact metrics table, preferably CSV and a Markdown table in README.

### Milestone 6: embedding analysis and README

- Extract embeddings from a held-out set.
- Save PCA plot.
- Compute cosine similarities for several original/rotated pairs.
- Update README with installation, commands, data, model, experiments, actual results, interpretation, and limitations.

## Sanity checks and common pitfalls

Actively check for the following.

### Shapes and dtypes

- Point samples: `float32`, `[N, 3]`.
- Batches: `[B, N, 3]`.
- Labels: integer tensor, `[B]`.
- Classifier output: `[B, num_classes]`.

### Training correctness

- Do not apply `softmax` before `CrossEntropyLoss`.
- Use `model.train()` during training and `model.eval()` with `torch.no_grad()` during evaluation.
- Move both inputs and labels to the selected device.
- Set seeds for Python, NumPy, and PyTorch.
- Make train/validation/test generation and rotation policies explicit.

### Experimental validity

- Prevent leakage between training and test data. Generate independent samples or use a deterministic split.
- Confirm that classes have balanced counts.
- Check whether the classifier exploits trivial artifacts: point density, scale, coordinate range, or generator-specific noise.
- Never describe PCA as proof of cluster quality or universal structure.
- Never conclude rotation invariance only from one visual embedding plot.

### PointNet-specific understanding

- Point-order invariance comes from shared point-wise processing plus symmetric pooling.
- Point-order invariance is different from rotation invariance.
- A basic PointNet-style network does not explicitly model local neighborhoods or correspondences.
- This prototype is a learned global representation baseline, not an explicit partial-symmetry detector.

## README requirements

The final README should be concise but complete and include:

1. Project goal and relation to the broader lab theme.
2. Explicit non-goals and limitations.
3. Setup instructions for WSL/Linux.
4. Commands for visualizing synthetic data and running each experiment.
5. Data-generation description and labels.
6. Brief PointNet-style architecture explanation.
7. Actual experiment table.
8. Saved figures: generated shapes, training curves, PCA embeddings.
9. Interpretation that distinguishes observation from conclusion.
10. Next steps: partial symmetry class, real point-cloud data, explicit symmetry descriptors, or stronger geometric models.

Suggested wording for limitations:

> This project does not solve partial approximate symmetry detection. It uses deliberately simplified synthetic labels to explore point-cloud representation, PointNet-style training, rotation augmentation, and embedding analysis. It does not establish that learned embeddings encode a universal symmetry spectrum.

## Expected interview explanation

The user should be able to truthfully say:

> I created a small controlled experiment with synthetic 3D point clouds. I implemented a PointNet-style model in PyTorch: a shared MLP extracts point-wise features, max pooling produces a permutation-invariant global embedding, and a classifier predicts a simplified symmetry-related label. I compared training with and without rotation augmentation, extracted embeddings, and used PCA and cosine similarity as exploratory diagnostics. The project does not detect partial approximate symmetries directly; its purpose was to understand the pipeline and its limitations before working on the laboratory problem.

## First requested action

1. Inspect the current repository.
2. Report what already exists.
3. Propose the smallest concrete plan for Milestones 1 and 2.
4. Wait for user approval before creating or modifying multiple files.