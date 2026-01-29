# discriminator.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DIAYNDiscriminator(nn.Module):
    """
    DIAYN discriminator p(z|trajectory).
    Uses trajectory (sequence of states) instead of single state to prevent "static pose" hacking.
    Intrinsic reward: log p(z|trajectory) - log p(z) = log p(z|trajectory) + log(num_skills).
    """
    def __init__(
        self,
        state_dim: int,
        num_skills: int,
        trajectory_length: int = 8,
        lr: float = 3e-4
    ):
        super().__init__()
        self.num_skills = num_skills
        self.state_dim = state_dim
        self.trajectory_length = trajectory_length

        # Temporal feature extractor using 1D convolutions
        # Input: (batch, trajectory_length, state_dim)
        # We'll transpose to (batch, state_dim, trajectory_length) for Conv1d
        self.temporal_net = nn.Sequential(
            nn.Conv1d(state_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Pool over time dimension -> (batch, 128, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_skills),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            trajectory: (batch, trajectory_length, state_dim)
        Returns:
            logits: (batch, num_skills)
        """
        # Transpose for Conv1d: (batch, state_dim, trajectory_length)
        x = trajectory.transpose(1, 2)
        features = self.temporal_net(x)
        logits = self.classifier(features)
        return logits

    def get_reward(self, trajectory_np: np.ndarray, skill_idx: int) -> float:
        """
        Compute intrinsic reward from a trajectory.
        Args:
            trajectory_np: (trajectory_length, state_dim) numpy array
            skill_idx: skill index
        Returns:
            intrinsic reward
        """
        self.eval()
        with torch.no_grad():
            # Validate input dimensions
            if trajectory_np.ndim != 2:
                raise ValueError(f"Expected 2D trajectory, got shape {trajectory_np.shape}")
            if trajectory_np.shape[0] != self.trajectory_length:
                raise ValueError(
                    f"Expected trajectory length {self.trajectory_length}, got {trajectory_np.shape[0]}"
                )

            # Add batch dimension: (1, trajectory_length, state_dim)
            traj = torch.as_tensor(trajectory_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.forward(traj)
            probs = torch.softmax(logits, dim=-1)
            # log p(z|trajectory) - log p(z)
            # Use eps=1e-8 for better numerical stability
            r = torch.log(probs[0, skill_idx] + 1e-8).item() + float(np.log(self.num_skills))
        return r

    def update(self, trajectories_np: np.ndarray, skill_indices_np: np.ndarray) -> float:
        """
        Train discriminator with cross-entropy on trajectory batches.
        Args:
            trajectories_np: (batch, trajectory_length, state_dim)
            skill_indices_np: (batch,)
        Returns:
            loss value
        """
        self.train()
        trajectories = torch.as_tensor(trajectories_np, dtype=torch.float32, device=self.device)
        skill_indices = torch.as_tensor(skill_indices_np, dtype=torch.long, device=self.device)

        logits = self.forward(trajectories)
        loss = self.criterion(logits, skill_indices)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
