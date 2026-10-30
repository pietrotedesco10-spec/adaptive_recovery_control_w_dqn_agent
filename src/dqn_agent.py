from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    """Compact Q-network for the three recovery actions."""

    def __init__(self, state_size: int, action_size: int):
        super(DQNNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.005,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        memory_size: int = 1024,
        batch_size: int = 64,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.memory: deque = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNNetwork(state_size, action_size).to(self.device)

        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()

    def update_target_network(self) -> None:
        """Synchronize the target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor)

        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([experience[0] for experience in batch], dtype=np.float32)
        actions = np.array([experience[1] for experience in batch], dtype=np.int64)
        rewards = np.array([experience[2] for experience in batch], dtype=np.float32)
        next_states = np.array([experience[3] for experience in batch], dtype=np.float32)
        dones = np.array([experience[4] for experience in batch], dtype=np.float32)

        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)

        current_q_values = self.model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            # Canonical DQN target: max_a Q_target(s', a).
            next_q_values = self.target_model(next_states).max(dim=1).values

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_loss = self.loss_function(current_q_values, target_q_values)

        loss = td_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )

        return loss.item()

    def save(self, path) -> None:
        torch.save(
            {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        saved_state_size = checkpoint.get(
            "state_size",
            checkpoint["model_state_dict"]["network.0.weight"].shape[1],
        )
        saved_action_size = checkpoint.get(
            "action_size",
            checkpoint["model_state_dict"]["network.4.weight"].shape[0],
        )

        if saved_state_size != self.state_size:
            raise ValueError(
                "Model state size does not match the current configuration: "
                f"saved={saved_state_size}, current={self.state_size}. "
                "Rebuild the dataset and retrain the DQN."
            )

        if saved_action_size != self.action_size:
            raise ValueError(
                "Model action size does not match the current configuration: "
                f"saved={saved_action_size}, current={self.action_size}. "
                "Retrain the DQN with the current action set."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)

        self.model.eval()
        self.target_model.eval()
