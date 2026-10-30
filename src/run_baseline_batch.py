"""Run the fixed-schedule baselines for the requested AnyLogic seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from anylogic_batch_runner import (
    BASE_DIR,
    DATA_DIR,
    normalized_seeds,
    run_simulation,
    validate_environment,
)


BASELINES = {
    1: ("no_recovery", None),
    2: ("threshold_recovery", None),
    3: ("work60_recovery40", 0.40),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three AnyLogic baselines for the selected seeds."
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def baseline_outputs(stem: str, seed: int) -> tuple[Path, Path]:
    return (
        DATA_DIR / f"rl_event_log_{stem}_s{seed}.csv",
        DATA_DIR / f"rl_decision_log_{stem}_s{seed}.csv",
    )


def generate_ga_plans() -> None:
    """Reuse the GA plan unless its schedule or generator has changed."""
    generator = BASE_DIR / "src" / "generate_static_ga_baseline.py"
    schedule = BASE_DIR / "schedule" / "simulation" / "schedule_v1_01.csv"

    if not generator.exists():
        raise FileNotFoundError(f"GA generator not found: {generator}")
    if not schedule.exists():
        raise FileNotFoundError(f"Simulation schedule not found: {schedule}")

    for baseline, (_, recovery_ratio) in BASELINES.items():
        if recovery_ratio is None:
            continue

        suffix = "r40"
        output = (
            BASE_DIR
            / "schedule"
            / "simulation"
            / f"ga_recovery_plan_{schedule.stem}_{suffix}.csv"
        )

        newest_input_time = max(
            schedule.stat().st_mtime_ns,
            generator.stat().st_mtime_ns,
        )
        if output.exists() and output.stat().st_mtime_ns >= newest_input_time:
            print(f"Using existing GA plan: {output.name}", flush=True)
            continue

        print(
            f"Preparing baseline {baseline}: "
            f"work={1.0 - recovery_ratio:.0%}, recovery={recovery_ratio:.0%}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(generator),
                "--schedule",
                str(schedule),
                "--rest-ratio",
                str(recovery_ratio),
                "--output",
                str(output),
            ],
            cwd=BASE_DIR,
            check=True,
        )


def main() -> None:
    args = parse_arguments()
    seeds = normalized_seeds(args.seeds)
    validate_environment()
    generate_ga_plans()

    total_runs = len(seeds) * len(BASELINES)
    print(f"\nBaseline runs: {total_runs}", flush=True)
    print(f"Seeds: {seeds}", flush=True)

    for seed in seeds:
        print(f"\nSeed {seed}", flush=True)
        for baseline, (stem, _) in BASELINES.items():
            run_simulation(
                mode=1,
                baseline=baseline,
                seed=seed,
                output_paths=baseline_outputs(stem, seed),
                label=f"baseline{baseline}_{stem}",
                timeout=args.timeout,
                startup_timeout=args.startup_timeout,
                retries=args.retries,
                overwrite=args.overwrite,
            )

    print("\nBaseline batch completed.", flush=True)
    print("KPI command: py src/analyze_experiments.py", flush=True)


if __name__ == "__main__":
    main()
