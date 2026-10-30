"""Collect factual exploration logs from AnyLogic for selected seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from anylogic_batch_runner import (
    DATA_DIR,
    normalized_seeds,
    run_simulation,
    validate_environment,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AnyLogic exploration for the selected seeds."
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def exploration_outputs(seed: int) -> tuple[Path, Path]:
    return (
        DATA_DIR / f"rl_event_log_exploration_s{seed}.csv",
        DATA_DIR / f"rl_decision_log_exploration_s{seed}.csv",
    )


def main() -> None:
    args = parse_arguments()
    seeds = normalized_seeds(args.seeds)
    validate_environment()

    print(f"Exploration runs: {len(seeds)}", flush=True)
    print(f"Seeds: {seeds}", flush=True)

    for seed in seeds:
        run_simulation(
            mode=2,
            baseline=1,
            seed=seed,
            output_paths=exploration_outputs(seed),
            label="exploration",
            timeout=args.timeout,
            startup_timeout=args.startup_timeout,
            retries=args.retries,
            overwrite=args.overwrite,
        )

    print("\nExploration batch completed.", flush=True)
    print("Next command: py src/build_training_dataset.py", flush=True)


if __name__ == "__main__":
    main()
