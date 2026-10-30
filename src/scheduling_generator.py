from __future__ import annotations

import argparse
import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pyjobshop import Model


BASE_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = BASE_DIR / "results" / "schedule"

OPERATORS = ("OperatorA", "OperatorB", "OperatorC")
OPERATOR_ASSIGNMENT_WEIGHTS = (0.45, 0.33, 0.22)
SOLVER_TIME_LIMIT_SECONDS = 2.0
MIN_JOBS = 6
MAX_JOBS = 8
MIN_TASKS_PER_JOB = 5
MAX_TASKS_PER_JOB = 8
DIFFICULTY_SETTINGS = {
    # Calibrated so a nominal mixed sequence approaches Hy after about
    # three consecutive tasks, while preserving distinct task severities.
    "light": {"weight": 0.40, "duration": (10, 15), "fatigue": 0.30},
    "medium": {"weight": 0.35, "duration": (16, 22), "fatigue": 0.42},
    "heavy": {"weight": 0.25, "duration": (23, 30), "fatigue": 0.55},
}

TRAINING_DIFFICULTY_WEIGHTS = {
    "nominal": (0.40, 0.35, 0.25),
    "stress": (0.30, 0.35, 0.35),
    "extreme": (0.25, 0.35, 0.40),
}

PROFILE_TARGETS = {
    "OperatorA": {"idle_share": 0.01, "max_run": 14},
    "OperatorB": {"idle_share": 0.12, "max_run": 7},
    "OperatorC": {"idle_share": 0.28, "max_run": 4},
}


@dataclass(frozen=True)
class TaskRecord:
    job_id: str
    source_job_id: str
    difficulty: str
    target_fatigue: float
    task_lambda: float
    nominal_duration_minutes: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random PyJobShop schedules for AnyLogic."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of random schedules to generate (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible schedules.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="random_schedule",
        help="Unique batch name for generated CSV and PNG files.",
    )
    parser.add_argument(
        "--merged-name",
        type=str,
        default=None,
        help=(
            "Optional filename for the merged exploration CSV, without .csv. "
            "Default: <prefix>_merged."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of files belonging to the selected batch name.",
    )
    parser.add_argument(
        "--output-set",
        choices=("exploration", "simulation"),
        default="exploration",
        help="Destination schedule set (default: exploration).",
    )
    parser.add_argument(
        "--profile-candidates",
        type=int,
        default=3,
        help=(
            "Candidate schedules solved for each output; the closest A/B/C "
            "workload profile is retained (default: 3)."
        ),
    )
    arguments = parser.parse_args()

    if arguments.count < 1:
        parser.error("--count must be at least 1.")
    if arguments.profile_candidates < 1:
        parser.error("--profile-candidates must be at least 1.")

    return arguments


def calculate_task_lambda(
    duration_minutes: float,
    difficulty: str,
) -> tuple[float, float]:
    """Derive Glock lambda from the selected cycle-fatigue intensity."""
    target_fatigue = float(DIFFICULTY_SETTINGS[difficulty]["fatigue"])
    duration_seconds = max(duration_minutes * 60.0, 1.0)
    task_lambda = -math.log1p(-target_fatigue) / duration_seconds
    return task_lambda, target_fatigue


def select_generation_profile(
    schedule_number: int,
    schedule_count: int,
    output_set: str,
) -> tuple[str, tuple[float, float, float]]:
    """Use broad workload mixes for exploration and nominal mixes for tests."""
    if output_set == "simulation":
        return "nominal", TRAINING_DIFFICULTY_WEIGHTS["nominal"]

    position = schedule_number / max(schedule_count, 1)
    if position <= 0.70:
        profile = "nominal"
    elif position <= 0.90:
        profile = "stress"
    else:
        profile = "extreme"
    return profile, TRAINING_DIFFICULTY_WEIGHTS[profile]


def build_random_model(
    rng: random.Random,
    difficulty_weights: tuple[float, float, float],
) -> tuple[Model, list[TaskRecord]]:
    """Build one random job-shop instance with serial job routes."""
    model = Model()
    resources = {
        name: model.add_machine(name=name)
        for name in OPERATORS
    }
    task_records: list[TaskRecord] = []

    job_count = rng.randint(MIN_JOBS, MAX_JOBS)

    for job_number in range(1, job_count + 1):
        job = model.add_job(name=f"Job{job_number}", release_date=0)
        previous_task = None
        task_count = rng.randint(MIN_TASKS_PER_JOB, MAX_TASKS_PER_JOB)

        for task_number in range(1, task_count + 1):
            difficulty = rng.choices(
                population=("light", "medium", "heavy"),
                weights=difficulty_weights,
                k=1,
            )[0]
            duration_limits = DIFFICULTY_SETTINGS[difficulty]["duration"]
            duration_minutes = rng.randint(
                int(duration_limits[0]),
                int(duration_limits[1]),
            )
            operator_name = rng.choices(
                population=OPERATORS,
                weights=OPERATOR_ASSIGNMENT_WEIGHTS,
                k=1,
            )[0]
            task_id = f"job_{job_number}_task_{task_number}"
            task_lambda, target_fatigue = calculate_task_lambda(
                duration_minutes,
                difficulty,
            )

            task = model.add_task(job, name=task_id)
            model.add_mode(task, resources[operator_name], duration=duration_minutes)

            if previous_task is not None:
                model.add_end_before_start(previous_task, task)
            previous_task = task

            task_records.append(
                TaskRecord(
                    job_id=task_id,
                    source_job_id=f"job_{job_number}",
                    difficulty=difficulty,
                    target_fatigue=target_fatigue,
                    task_lambda=task_lambda,
                    nominal_duration_minutes=duration_minutes,
                )
            )

    return model, task_records


def export_anylogic_schedule(
    model: Model,
    task_records: list[TaskRecord],
    result,
    output_path: Path | None,
    schedule_id: str,
) -> list[dict[str, object]]:
    """Export solved operations and natural machine gaps in AnyLogic CSV format."""
    if result.best is None:
        raise RuntimeError("PyJobShop did not return a feasible schedule.")

    data = model.data()
    machine_ids = set(data.machine_idcs)
    solved_operations: dict[str, list[dict[str, object]]] = {
        operator_name: []
        for operator_name in OPERATORS
    }

    for task_index, scheduled_task in enumerate(result.best.tasks):
        if not scheduled_task.present:
            continue

        record = task_records[task_index]
        machine_id = next(
            (
                resource_id
                for resource_id in scheduled_task.resources
                if resource_id in machine_ids
            ),
            None,
        )
        if machine_id is None:
            raise RuntimeError(f"No operator resource found for {record.job_id}.")

        operator_name = data.resources[machine_id].name
        solved_operations[operator_name].append(
            {
                "record": record,
                "start_minutes": float(scheduled_task.start),
                "end_minutes": float(scheduled_task.end),
            }
        )

    rows: list[dict[str, object]] = []

    for operator_name in OPERATORS:
        operations = solved_operations[operator_name]
        operations.sort(
            key=lambda operation: (
                operation["start_minutes"],
                operation["end_minutes"],
            )
        )
        cursor_minutes = 0.0
        sequence = 1

        for operation in operations:
            start_minutes = float(operation["start_minutes"])
            end_minutes = float(operation["end_minutes"])
            record: TaskRecord = operation["record"]

            if start_minutes > cursor_minutes + 1.0e-9:
                rows.append(
                    {
                        "operator": operator_name,
                        "sequence": sequence,
                        "activity_type": "idle",
                        "job_id": "idle",
                        "processing_time_seconds": (
                            start_minutes - cursor_minutes
                        ) * 60.0,
                        "lambda_per_second": 0.0,
                        "difficulty": "idle",
                        "target_fatigue_from_zero": 0.0,
                        "source_job_id": "",
                        "scheduled_start_seconds": cursor_minutes * 60.0,
                        "scheduled_end_seconds": start_minutes * 60.0,
                        "schedule_id": schedule_id,
                    }
                )
                sequence += 1

            rows.append(
                {
                    "operator": operator_name,
                    "sequence": sequence,
                    "activity_type": "task",
                    "job_id": record.job_id,
                    "processing_time_seconds": record.nominal_duration_minutes * 60.0,
                    "lambda_per_second": record.task_lambda,
                    "difficulty": record.difficulty,
                    "target_fatigue_from_zero": record.target_fatigue,
                    "source_job_id": record.source_job_id,
                    "scheduled_start_seconds": start_minutes * 60.0,
                    "scheduled_end_seconds": end_minutes * 60.0,
                    "schedule_id": schedule_id,
                }
            )
            sequence += 1
            cursor_minutes = end_minutes

    rows.sort(
        key=lambda row: (
            str(row["operator"]),
            float(row["scheduled_start_seconds"]),
            int(row["sequence"]),
        )
    )

    if output_path is not None:
        write_schedule_csv(rows, output_path)

    return rows


def write_schedule_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write a single AnyLogic-compatible schedule CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _compact_task_label(job_id: str) -> str:
    """Convert job_4_task_3 to the compact presentation label J4 T3."""
    match = re.fullmatch(r"job_(\d+)_task_(\d+)", job_id.strip().lower())
    if match is None:
        return job_id
    return f"J{match.group(1)} T{match.group(2)}"


def save_gantt(
    rows: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    """Save a clear operator Gantt with one distinct color per job."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_rows = [
        row for row in rows
        if str(row.get("activity_type", "")).strip().lower() == "task"
    ]
    job_names = sorted(
        {str(row.get("source_job_id", "")).strip() for row in task_rows},
        key=lambda value: int(re.search(r"\d+", value).group())
        if re.search(r"\d+", value) else value,
    )
    palette = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # green
        "#D55E00",  # vermillion
        "#CC79A7",  # magenta
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#6A3D9A",  # purple
    ]
    job_colors = {
        job_name: palette[index % len(palette)]
        for index, job_name in enumerate(job_names)
    }

    operator_order = [
        operator for operator in OPERATORS
        if any(str(row.get("operator", "")) == operator for row in rows)
    ]
    y_positions = {
        operator: len(operator_order) - 1 - index
        for index, operator in enumerate(operator_order)
    }

    figure, axis = plt.subplots(figsize=(20, 6.8))
    bar_height = 0.62

    for row in rows:
        operator = str(row.get("operator", "")).strip()
        if operator not in y_positions:
            continue

        start = float(row.get("scheduled_start_seconds", 0.0)) / 60.0
        end = float(row.get("scheduled_end_seconds", 0.0)) / 60.0
        duration = max(0.0, end - start)
        if duration <= 0.0:
            continue

        activity_type = str(row.get("activity_type", "")).strip().lower()
        source_job = str(row.get("source_job_id", "")).strip()
        face_color = "#E5E7EB" if activity_type == "idle" else job_colors[source_job]
        edge_color = "#9CA3AF" if activity_type == "idle" else "#263238"
        y = y_positions[operator]

        axis.broken_barh(
            [(start, duration)],
            (y - bar_height / 2.0, bar_height),
            facecolors=face_color,
            edgecolors=edge_color,
            linewidth=0.8,
        )

        if activity_type == "task":
            label = _compact_task_label(str(row.get("job_id", "")))
            text_color = "#111827" if face_color in {"#E69F00", "#56B4E9", "#F0E442"} else "white"
            axis.text(
                start + duration / 2.0,
                y,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="semibold",
                color=text_color,
                clip_on=True,
            )

    axis.set_title(title, pad=14, fontsize=17)
    axis.set_xlabel("Time (minutes)")
    axis.set_ylabel("Operator")
    axis.set_yticks(
        [y_positions[operator] for operator in operator_order],
        [operator.replace("Operator", "Operator ") for operator in operator_order],
    )
    schedule_end = max(
        float(row.get("scheduled_end_seconds", 0.0)) / 60.0
        for row in rows
    )
    axis.set_xlim(0.0, schedule_end * 1.01)
    axis.set_ylim(-0.65, len(operator_order) - 0.35)
    axis.grid(axis="x", alpha=0.22)
    axis.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=job_colors[job], edgecolor="#263238", label=job.replace("_", " ").title())
        for job in job_names
    ]
    if any(str(row.get("activity_type", "")).lower() == "idle" for row in rows):
        legend_handles.append(
            Patch(facecolor="#E5E7EB", edgecolor="#9CA3AF", label="Idle")
        )
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(5, len(legend_handles)),
        frameon=False,
    )

    figure.tight_layout(pad=1.4)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def ensure_output_is_available(output_path: Path, overwrite: bool) -> None:
    """Prevent accidental replacement of a previously generated schedule."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Choose a different --prefix/--merged-name or use --overwrite intentionally."
        )


def workload_profile(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Summarize natural idle and uninterrupted task runs per operator."""
    result = {}
    for operator in OPERATORS:
        operator_rows = [row for row in rows if row["operator"] == operator]
        task_minutes = 0.0
        idle_minutes = 0.0
        current_run = 0
        max_run = 0

        for row in operator_rows:
            duration = float(row["processing_time_seconds"]) / 60.0
            if row["activity_type"] == "idle":
                idle_minutes += duration
                max_run = max(max_run, current_run)
                current_run = 0
            else:
                task_minutes += duration
                current_run += 1
        max_run = max(max_run, current_run)
        result[operator] = {
            "idle_share": idle_minutes / max(task_minutes + idle_minutes, 1.0e-9),
            "max_run": float(max_run),
        }
    return result


def workload_profile_score(rows: list[dict[str, object]]) -> float:
    """Prefer a busy A, moderately interrupted B, and idle-rich C."""
    profile = workload_profile(rows)
    score = 0.0
    for operator, target in PROFILE_TARGETS.items():
        actual = profile[operator]
        score += 16.0 * abs(actual["idle_share"] - target["idle_share"])
        score += 0.30 * abs(actual["max_run"] - target["max_run"])

    # Long uninterrupted runs on C defeat the purpose of its natural idle.
    score += 1.5 * max(0.0, profile["OperatorC"]["max_run"] - 5.0)
    return score


def generate_schedule(
    schedule_number: int,
    rng: random.Random,
    prefix: str,
    overwrite: bool,
    output_set: str,
    schedule_count: int,
    profile_candidates: int,
) -> tuple[Path, Path, float, int, list[dict[str, object]], str]:
    generation_profile, difficulty_weights = select_generation_profile(
        schedule_number,
        schedule_count,
        output_set,
    )

    filename = f"{prefix}_{schedule_number:02d}"
    schedule_dir = BASE_DIR / "schedule" / output_set
    schedule_path = schedule_dir / f"{filename}.csv"
    figure_path = FIGURE_DIR / f"{filename}_gantt.png"
    if output_set == "simulation":
        ensure_output_is_available(schedule_path, overwrite)
    ensure_output_is_available(figure_path, overwrite)

    best_candidate = None
    best_score = float("inf")

    for _ in range(profile_candidates):
        model, task_records = build_random_model(rng, difficulty_weights)
        model.set_objective(weight_makespan=1)
        result = model.solve(
            display=False,
            num_workers=1,
            time_limit=SOLVER_TIME_LIMIT_SECONDS,
        )
        if result.best is None:
            continue
        rows = export_anylogic_schedule(
            model=model,
            task_records=task_records,
            result=result,
            output_path=None,
            schedule_id=filename,
        )
        score = workload_profile_score(rows)
        if score < best_score:
            best_score = score
            best_candidate = (model, task_records, result, rows)

    if best_candidate is None:
        raise RuntimeError(f"Schedule {schedule_number} is infeasible.")

    model, task_records, result, rows = best_candidate
    if output_set == "simulation":
        write_schedule_csv(rows, schedule_path)
    save_gantt(
        rows=rows,
        output_path=figure_path,
        title=f"Random Production Schedule {schedule_number}",
    )

    job_count = len({record.source_job_id for record in task_records})
    return (
        schedule_path,
        figure_path,
        float(result.best.makespan),
        job_count,
        rows,
        generation_profile,
    )


def main() -> None:
    arguments = parse_arguments()
    rng = random.Random(arguments.seed)

    print(f"Generating {arguments.count} random schedule(s)...")
    if arguments.seed is not None:
        print(f"Random seed: {arguments.seed}")

    merged_rows: list[dict[str, object]] = []

    for schedule_number in range(1, arguments.count + 1):
        (
            schedule_path,
            figure_path,
            makespan,
            job_count,
            rows,
            generation_profile,
        ) = generate_schedule(
            schedule_number=schedule_number,
            rng=rng,
            prefix=arguments.prefix,
            overwrite=arguments.overwrite,
            output_set=arguments.output_set,
            schedule_count=arguments.count,
            profile_candidates=arguments.profile_candidates,
        )
        merged_rows.extend(rows)
        print(f"\nSchedule {schedule_number}")
        print(f"  Jobs: {job_count} (random range {MIN_JOBS}-{MAX_JOBS})")
        print(
            "  Tasks per job: random range "
            f"{MIN_TASKS_PER_JOB}-{MAX_TASKS_PER_JOB}"
        )
        print(
            "  Difficulty mix: "
            f"{generation_profile} {TRAINING_DIFFICULTY_WEIGHTS[generation_profile]}"
        )
        print(
            "  Fatigue intensities (light/medium/heavy): "
            f"{tuple(DIFFICULTY_SETTINGS[name]['fatigue'] for name in ('light', 'medium', 'heavy'))}"
        )
        print(f"  Workload profile: {workload_profile(rows)}")
        print(f"  Makespan: {makespan:.2f} min")
        print(f"  CSV rows including idle periods: {len(rows)}")
        if arguments.output_set == "simulation":
            print(f"  AnyLogic CSV: {schedule_path}")
        print(f"  Gantt: {figure_path}")

    if arguments.output_set == "exploration":
        merged_name = arguments.merged_name or f"{arguments.prefix}_merged"
        if not merged_name.endswith(".csv"):
            merged_name += ".csv"
        merged_path = BASE_DIR / "schedule" / "exploration" / merged_name
        ensure_output_is_available(merged_path, arguments.overwrite)
        write_schedule_csv(merged_rows, merged_path)
        print("\nMerged exploration schedule set:")
        print(f"  AnyLogic CSV: {merged_path}")
        print(f"  Scenarios: {arguments.count}")
        print(f"  Total rows including idle periods: {len(merged_rows)}")


if __name__ == "__main__":
    main()
