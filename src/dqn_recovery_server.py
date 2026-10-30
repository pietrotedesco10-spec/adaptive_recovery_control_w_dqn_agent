"""Serve actions from the trained DQN to the AnyLogic TCP client."""

from __future__ import annotations

import math
import socket
import time
from pathlib import Path

from dqn_agent import DQNAgent
from rl_config import (
    ACTION_NAMES,
    N_ACTIONS,
    STATE_COLUMNS,
    normalize_state_values,
)


HOST = "127.0.0.1"
PORT = 5005

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "dqn_fatigue_agent.pth"
ANYLOGIC_MODEL_PATH = Path(
    r"C:\Users\pietr\Models\rl_agent_multi_op\rl_agent_multi_op.alp"
)
MODEL_DEPENDENCIES = (
    ANYLOGIC_MODEL_PATH,
    BASE_DIR / "src" / "build_training_dataset.py",
    BASE_DIR / "src" / "train_from_transitions.py",
    BASE_DIR / "src" / "dqn_agent.py",
    BASE_DIR / "src" / "rl_config.py",
)


def parse_anylogic_state(message: str) -> list[float]:
    """Validate the comma-separated state against the training schema."""
    values = [float(value.strip()) for value in message.split(",") if value.strip()]

    if not all(math.isfinite(value) for value in values):
        raise ValueError("State contains a non-finite value.")

    if len(values) != len(STATE_COLUMNS):
        raise ValueError(
            f"State has {len(values)} features; "
            f"expected {len(STATE_COLUMNS)}."
        )

    return values


def load_agent() -> DQNAgent:
    """Load a model trained after its simulation and training dependencies."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"DQN model not found: {MODEL_PATH}")
    latest_dependency_time = max(
        (
            dependency.stat().st_mtime_ns
            for dependency in MODEL_DEPENDENCIES
            if dependency.exists()
        ),
        default=0,
    )
    if MODEL_PATH.stat().st_mtime_ns < latest_dependency_time:
        raise ValueError(
            "DQN model predates the current simulation, reward, state schema, "
            "or network code. Regenerate the exploration dataset and retrain "
            "before starting mode 3."
        )

    agent = DQNAgent(
        state_size=len(STATE_COLUMNS),
        action_size=N_ACTIONS,
        epsilon=0.0,
    )
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0

    print(f"DQN model loaded from: {MODEL_PATH}")
    print(f"State features: {STATE_COLUMNS}")
    return agent


def select_action(agent: DQNAgent, message: str) -> int:
    """Apply training-time normalization before DQN inference."""
    raw_state = parse_anylogic_state(message)
    normalized_state = normalize_state_values(*raw_state)
    action = agent.select_action(normalized_state)

    if action not in range(N_ACTIONS):
        raise ValueError(f"Model returned invalid action: {action}")
    return action


def start_server() -> None:
    agent = load_agent()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)

        print(f"=== DQN SERVER LISTENING on {HOST}:{PORT} ===")

        try:
            while True:
                conn, _ = server_socket.accept()

                with conn:
                    try:
                        # Avoid blocking the server on a broken client connection.
                        conn.settimeout(5.0)
                        message = conn.recv(4096).decode("utf-8").strip()

                        if not message:
                            raise ValueError("AnyLogic sent an empty state.")

                        start_time = time.perf_counter()
                        action = select_action(agent, message)
                        conn.sendall(f"{action}\n".encode("utf-8"))
                        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                        print(
                            f"State: {message} | "
                            f"Action: {action} "
                            f"({ACTION_NAMES.get(action, 'unknown')}) | "
                            f"Time: {elapsed_ms:.2f} ms"
                        )

                    except Exception as exc:
                        print(f"Inference error: {exc}")
                        try:
                            conn.sendall(b"ERROR\n")
                        except OSError:
                            pass

        except KeyboardInterrupt:
            print("\nShutting down DQN server...")


if __name__ == "__main__":
    start_server()
