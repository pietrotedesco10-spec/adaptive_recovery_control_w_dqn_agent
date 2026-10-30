from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dqn_agent import DQNAgent
from rl_config import (
    ACTION_NAMES,
    N_ACTIONS,
    STATE_COLUMNS,
    TRAINING_CONFIG,
    TRAINING_PROFILE,
    normalize_state_values,
)


BASE_DIR = Path(__file__).resolve().parents[1]
TRANSITIONS_PATH = BASE_DIR / "data" / "agent_log" / "training_transitions.csv"
MODEL_PATH = BASE_DIR / "models" / "dqn_fatigue_agent.pth"
RESULTS_DIR = BASE_DIR / "results" / "csv"
ANYLOGIC_MODEL_PATH = Path(
    r"C:\Users\pietr\Models\rl_agent_multi_op\rl_agent_multi_op.alp"
)
TRANSITION_DEPENDENCIES = (
    ANYLOGIC_MODEL_PATH,
    BASE_DIR / "src" / "build_training_dataset.py",
    BASE_DIR / "src" / "rl_config.py",
)


def load_transitions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Transitions file not found: {path}")
    latest_dependency_time = max(
        (
            dependency.stat().st_mtime_ns
            for dependency in TRANSITION_DEPENDENCIES
            if dependency.exists()
        ),
        default=0,
    )
    if path.stat().st_mtime_ns < latest_dependency_time:
        raise ValueError(
            "Training transitions predate the current AnyLogic model, "
            "reward, or state configuration. "
            "Rerun exploration and build_training_dataset.py before training."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required_columns = (
        ["action", "reward", "done"]
        + [f"state_{column}" for column in STATE_COLUMNS]
        + [f"next_{column}" for column in STATE_COLUMNS]
    )
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing transition columns: {missing}")

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns).reset_index(drop=True)
    df = df[df["action"].isin(range(N_ACTIONS))].copy()

    if df.empty:
        raise ValueError("No valid transitions found.")

    return df


def row_to_state(row: pd.Series, prefix: str) -> np.ndarray:
    values = [row[f"{prefix}_{column}"] for column in STATE_COLUMNS]
    return normalize_state_values(*values)


def dataframe_to_experiences(df: pd.DataFrame) -> list[tuple]:
    experiences = []

    for _, row in df.iterrows():
        experiences.append(
            (
                row_to_state(row, "state"),
                int(row["action"]),
                float(row.get("trainingReward", row["reward"])),
                row_to_state(row, "next"),
                bool(int(row["done"])),
            )
        )

    return experiences


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_agent(
    agent: DQNAgent,
    experiences: list[tuple],
) -> list[float]:
    for experience in experiences:
        agent.remember(*experience)

    losses: list[float] = []
    print("\nTraining DQN from offline transitions...")

    training_steps = int(TRAINING_CONFIG["training_steps"])
    target_update_frequency = int(TRAINING_CONFIG["target_update_frequency"])
    report_interval = int(TRAINING_CONFIG["report_interval"])

    for step in range(1, training_steps + 1):
        loss = agent.train_step()

        if loss is not None:
            losses.append(loss)

        if step % target_update_frequency == 0:
            agent.update_target_network()

        if step % report_interval == 0:
            average_loss = np.mean(losses[-report_interval:]) if losses else 0.0
            print(
                f"Step {step}/{training_steps} | "
                f"Average loss: {average_loss:.6f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return losses


def save_loss_history(losses: list[float]) -> None:
    if not losses:
        return

    loss_path = RESULTS_DIR / "transition_training_loss.csv"
    pd.DataFrame(
        {
            "step": np.arange(1, len(losses) + 1),
            "loss": losses,
        }
    ).to_csv(loss_path, index=False)

    print(f"Loss history saved to: {loss_path}")


def save_policy_check(
    agent: DQNAgent,
    df: pd.DataFrame,
) -> None:
    agent.epsilon = 0.0

    state_keys = [
        column
        for column in ["sourceRun", "decisionTime", "scheduleId", "operator"]
        if column in df.columns
    ]
    unique_mask = (
        ~df.duplicated(subset=state_keys)
        if state_keys
        else pd.Series(True, index=df.index)
    )

    output = df.loc[unique_mask].copy()
    output["predictedActionAfterTraining"] = [
        agent.select_action(row_to_state(row, "state"))
        for _, row in output.iterrows()
    ]
    output["predictedActionNameAfterTraining"] = output[
        "predictedActionAfterTraining"
    ].map(ACTION_NAMES).fillna("unknown")

    print("\nPredicted action distribution after training:")
    for action, count in (
        output["predictedActionAfterTraining"]
        .value_counts()
        .sort_index()
        .items()
    ):
        print(f"{action} - {ACTION_NAMES.get(action, 'unknown')}: {count}")

    output_path = RESULTS_DIR / "transition_policy_check.csv"
    output.to_csv(output_path, index=False)
    print(f"Policy check saved to: {output_path}")

    if "operator" in output.columns:
        print("\nPredicted actions by operator:")
        print(
            pd.crosstab(
                output["operator"],
                output["predictedActionNameAfterTraining"],
            )
        )

    if "region" in output.columns:
        print("\nPredicted actions by region:")
        print(
            pd.crosstab(
                output["region"],
                output["predictedActionNameAfterTraining"],
            )
        )

def main() -> None:
    set_random_seed(int(TRAINING_CONFIG["random_seed"]))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training profile: {TRAINING_PROFILE}")

    print("Loading transitions...")
    df = load_transitions(TRANSITIONS_PATH)
    print(f"Transitions loaded: {len(df)}")

    if "operator" in df.columns:
        print("\nTransitions by operator:")
        print(df["operator"].value_counts())

    print("\nObserved exploration action distribution:")
    print(df["action"].value_counts().sort_index())

    print("\nReward summary:")
    print(df["reward"].describe())

    training_df = df.copy()

    if training_df.empty:
        raise ValueError("No sequential transitions are eligible for training.")

    print(f"Sequential transitions used for training: {len(training_df)}")
    print("\nActual explored action distribution used for training:")
    print(training_df["action"].value_counts().sort_index())

    reward_scale = max(
        1.0,
        float(training_df["reward"].abs().quantile(0.95)),
    )
    training_df["trainingReward"] = training_df["reward"] / reward_scale
    print(f"Reward scale used for training: {reward_scale:.6f}")

    experiences = dataframe_to_experiences(training_df)
    batch_size = min(int(TRAINING_CONFIG["batch_size"]), len(experiences))
    memory_size = max(
        int(TRAINING_CONFIG["minimum_memory_size"]),
        len(experiences) * int(TRAINING_CONFIG["memory_multiplier"]),
    )

    agent = DQNAgent(
        state_size=len(STATE_COLUMNS),
        action_size=N_ACTIONS,
        learning_rate=float(TRAINING_CONFIG["learning_rate"]),
        gamma=float(TRAINING_CONFIG["gamma"]),
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        memory_size=memory_size,
        batch_size=batch_size,
    )

    losses = train_agent(agent, experiences)
    agent.save(MODEL_PATH)

    print("\nTraining completed.")
    print(f"Model saved to: {MODEL_PATH}")

    save_loss_history(losses)
    save_policy_check(agent, df)


if __name__ == "__main__":
    main()
