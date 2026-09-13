"""Minimal PointNet-style model for point-cloud classification.

Forward pass:
    x: [B, N, 3] -> shared MLP -> point features [B, N, D]
    -> max pooling over points -> global embedding [B, D]
    -> classifier -> logits [B, num_classes]
"""

import torch
import torch.nn as nn


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
        point_features = self.point_mlp(x)                 # [B, N, embedding_dim]
        embedding = point_features.max(dim=1).values       # [B, embedding_dim]
        logits = self.classifier(embedding)                # [B, num_classes]
        if return_embedding:
            return logits, embedding
        return logits