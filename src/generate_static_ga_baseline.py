from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SCHEDULE_DIR = BASE_DIR / "schedule" / "simulation"
RESULTS_DIR = BASE_DIR / "results" / "csv"

HY = 0.80
# Must match Operator.mu in AnyLogic and the DQN reward model.
MU_PER_SECOND = 0.0017
SHIFT_SECONDS = 4.0 * 3600.0
SHIFT_GROWTH = 0.05

ACTION_CONTINUE = 0
ACTION_SHORT_BREAK = 1
ACTION_LONG_RECOVERY = 2

ACTION_NAMES = {
    ACTION_CONTINUE: "continue_working",
    ACTION_SHORT_BREAK: "short_break",
    ACTION_LONG_RECOVERY: "long_break",
}

RESIDUAL_FATIGUE = {
    # Action 1 reduces fatigue by 40%; action 2 reduces it by 60%.
    ACTION_SHORT_BREAK: 0.60,
    ACTION_LONG_RECOVERY: 0.40,
}

OPERATOR_MULTIPLIERS = {
    "OperatorA": 0.95,
    "OperatorB": 1.05,
    "OperatorC": 0.98,
}


@dataclass(frozen=True)
class Activity:
    operator: str
    sequence: int
    activity_type: str
    job_id: str
    duration_seconds: float
    lambda_per_second: float
    schedule_id: str


@dataclass(frozen=True)
class Metrics:
    makespan_seconds: float
    average_fatigue: float
    maximum_fatigue: float
    time_above_hy_seconds: float
    recovery_time_seconds: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a static work-rest plan with a genetic algorithm. "
            "The plan is optimized offline and does not react during simulation."
        )
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=SCHEDULE_DIR / "schedule_v1_01.csv",
        help="Input schedule CSV.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument(
        "--rest-ratio",
        type=float,
        default=0.30,
        help="Share of eligible task boundaries containing a recovery.",
    )
    parser.add_argument(
        "--makespan-weight",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--fatigue-weight",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plan CSV. By default it is named after the input schedule.",
    )
    return parser.parse_args()


def load_schedule(path: Path) -> list[Activity]:
    if not path.exists():
        raise FileNotFoundError(f"Schedule not found: {path}")

    activities: list[Activity] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "operator",
            "sequence",
            "activity_type",
            "job_id",
            "processing_time_seconds",
            "lambda_per_second",
            "schedule_id",
        }
        missing = required.difference(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing schedule columns: {sorted(missing)}"
            )

        for row in reader:
            activity_type = row["activity_type"].strip().lower()
            job_id = row["job_id"].strip()

            if activity_type not in {"task", "idle"}:
                continue

            activities.append(
                Activity(
                    operator=row["operator"].strip(),
                    sequence=int(row["sequence"]),
                    activity_type=activity_type,
                    job_id=job_id if job_id else "idle",
                    duration_seconds=float(row["processing_time_seconds"]),
                    lambda_per_second=float(row["lambda_per_second"] or 0.0),
                    schedule_id=row["schedule_id"].strip(),
                )
            )

    if not activities:
        raise ValueError(f"No valid activities found in: {path}")

    schedule_ids = {activity.schedule_id for activity in activities}
    if len(schedule_ids) != 1:
        raise ValueError(
            "The static GA baseline requires exactly one schedule_id."
        )

    activities.sort(key=lambda item: (item.operator, item.sequence))
    return activities


def fatigue_integral_work(
    initial_fatigue: float,
    rate: float,
    duration_seconds: float,
) -> tuple[float, float]:
    if rate <= 0.0:
        return initial_fatigue * duration_seconds, initial_fatigue

    decay = math.exp(-rate * duration_seconds)
    final_fatigue = 1.0 - (1.0 - initial_fatigue) * decay
    area = (
        duration_seconds
        - (1.0 - initial_fatigue) * (1.0 - decay) / rate
    )
    return area, final_fatigue


def fatigue_integral_rest(
    initial_fatigue: float,
    duration_seconds: float,
) -> tuple[float, float]:
    if MU_PER_SECOND <= 0.0:
        return initial_fatigue * duration_seconds, initial_fatigue

    residual = math.exp(-MU_PER_SECOND * duration_seconds)
    final_fatigue = initial_fatigue * residual
    area = initial_fatigue * (1.0 - residual) / MU_PER_SECOND
    return area, final_fatigue


def time_above_hy_work(
    initial_fatigue: float,
    final_fatigue: float,
    rate: float,
    duration_seconds: float,
) -> float:
    if initial_fatigue >= HY:
        return duration_seconds
    if final_fatigue <= HY or rate <= 0.0:
        return 0.0

    crossing_time = -math.log(
        (1.0 - HY) / (1.0 - initial_fatigue)
    ) / rate
    return max(0.0, duration_seconds - crossing_time)


def time_above_hy_rest(
    initial_fatigue: float,
    duration_seconds: float,
) -> float:
    if initial_fatigue <= HY or MU_PER_SECOND <= 0.0:
        return 0.0

    crossing_time = math.log(initial_fatigue / HY) / MU_PER_SECOND
    return min(duration_seconds, max(0.0, crossing_time))


def recovery_duration_seconds(action: int) -> float:
    residual = RESIDUAL_FATIGUE[action]
    return -math.log(residual) / MU_PER_SECOND


def eligible_boundaries(
    activities: list[Activity],
) -> list[tuple[str, int]]:
    final_sequence = {
        operator: max(
            activity.sequence
            for activity in activities
            if activity.operator == operator
        )
        for operator in {activity.operator for activity in activities}
    }

    return [
        (activity.operator, activity.sequence)
        for activity in activities
        if activity.activity_type == "task"
        and activity.sequence < final_sequence[activity.operator]
    ]


def simulate_plan(
    activities: list[Activity],
    loci: list[tuple[str, int]],
    chromosome: list[int],
) -> Metrics:
    action_by_boundary = dict(zip(loci, chromosome, strict=True))
    activities_by_operator: dict[str, list[Activity]] = {}

    for activity in activities:
        activities_by_operator.setdefault(activity.operator, []).append(
            activity
        )

    total_area = 0.0
    total_observed_time = 0.0
    maximum_fatigue = 0.0
    total_time_above_hy = 0.0
    total_recovery_time = 0.0
    operator_end_times: list[float] = []

    for operator, operator_activities in activities_by_operator.items():
        fatigue = 0.0
        elapsed = 0.0
        operator_area = 0.0
        operator_time_above_hy = 0.0
        multiplier = OPERATOR_MULTIPLIERS.get(operator, 1.0)

        for activity in operator_activities:
            duration = max(0.0, activity.duration_seconds)

            if activity.activity_type == "idle":
                area, final_fatigue = fatigue_integral_rest(
                    fatigue,
                    duration,
                )
                operator_time_above_hy += time_above_hy_rest(
                    fatigue,
                    duration,
                )
            else:
                fatigue_coefficient = (
                    1.0 if fatigue <= 0.25 else 0.75 + fatigue
                )
                duration *= fatigue_coefficient
                shift_progress = min(1.0, elapsed / SHIFT_SECONDS)
                effective_lambda = (
                    activity.lambda_per_second
                    * multiplier
                    * (1.0 + SHIFT_GROWTH * shift_progress)
                )
                area, final_fatigue = fatigue_integral_work(
                    fatigue,
                    effective_lambda,
                    duration,
                )
                operator_time_above_hy += time_above_hy_work(
                    fatigue,
                    final_fatigue,
                    effective_lambda,
                    duration,
                )

            operator_area += area
            elapsed += duration
            fatigue = final_fatigue
            maximum_fatigue = max(maximum_fatigue, fatigue)

            action = action_by_boundary.get(
                (operator, activity.sequence),
                ACTION_CONTINUE,
            )

            if action in RESIDUAL_FATIGUE:
                recovery_duration = recovery_duration_seconds(action)
                area, final_fatigue = fatigue_integral_rest(
                    fatigue,
                    recovery_duration,
                )
                operator_time_above_hy += time_above_hy_rest(
                    fatigue,
                    recovery_duration,
                )
                operator_area += area
                elapsed += recovery_duration
                total_recovery_time += recovery_duration
                fatigue = final_fatigue

        total_area += operator_area
        total_observed_time += elapsed
        total_time_above_hy += operator_time_above_hy
        operator_end_times.append(elapsed)

    average_fatigue = (
        total_area / total_observed_time
        if total_observed_time > 0.0
        else 0.0
    )

    return Metrics(
        makespan_seconds=max(operator_end_times, default=0.0),
        average_fatigue=average_fatigue,
        maximum_fatigue=maximum_fatigue,
        time_above_hy_seconds=total_time_above_hy,
        recovery_time_seconds=total_recovery_time,
    )


def repair_chromosome(
    chromosome: list[int],
    loci: list[tuple[str, int]],
    rest_counts: dict[str, int],
    rng: random.Random,
) -> list[int]:
    repaired = chromosome.copy()

    for operator, required_count in rest_counts.items():
        operator_indices = [
            index
            for index, locus in enumerate(loci)
            if locus[0] == operator
        ]
        active = [
            index
            for index in operator_indices
            if repaired[index] != ACTION_CONTINUE
        ]
        inactive = [
            index
            for index in operator_indices
            if repaired[index] == ACTION_CONTINUE
        ]

        while len(active) > required_count:
            index = rng.choice(active)
            repaired[index] = ACTION_CONTINUE
            active.remove(index)
            inactive.append(index)

        while len(active) < required_count:
            index = rng.choice(inactive)
            repaired[index] = rng.choice(
                [ACTION_SHORT_BREAK, ACTION_LONG_RECOVERY]
            )
            inactive.remove(index)
            active.append(index)

    return repaired


def random_chromosome(
    loci: list[tuple[str, int]],
    rest_counts: dict[str, int],
    rng: random.Random,
) -> list[int]:
    chromosome = [ACTION_CONTINUE] * len(loci)

    for operator, required_count in rest_counts.items():
        operator_indices = [
            index
            for index, locus in enumerate(loci)
            if locus[0] == operator
        ]

        for index in rng.sample(operator_indices, required_count):
            chromosome[index] = rng.choice(
                [ACTION_SHORT_BREAK, ACTION_LONG_RECOVERY]
            )

    return chromosome


def mutate(
    chromosome: list[int],
    loci: list[tuple[str, int]],
    rest_counts: dict[str, int],
    rng: random.Random,
    mutation_rate: float = 0.20,
) -> list[int]:
    mutated = chromosome.copy()

    if rng.random() < mutation_rate:
        operator = rng.choice(list(rest_counts))
        active = [
            index
            for index, action in enumerate(mutated)
            if loci[index][0] == operator
            and action != ACTION_CONTINUE
        ]
        inactive = [
            index
            for index, action in enumerate(mutated)
            if loci[index][0] == operator
            and action == ACTION_CONTINUE
        ]

        if active and inactive:
            old_index = rng.choice(active)
            new_index = rng.choice(inactive)
            action = mutated[old_index]
            mutated[old_index] = ACTION_CONTINUE
            mutated[new_index] = action

    if rng.random() < mutation_rate:
        active = [
            index for index, action in enumerate(mutated) if action != 0
        ]
        if active:
            index = rng.choice(active)
            mutated[index] = (
                ACTION_LONG_RECOVERY
                if mutated[index] == ACTION_SHORT_BREAK
                else ACTION_SHORT_BREAK
            )

    return repair_chromosome(mutated, loci, rest_counts, rng)


def normalized_costs(
    metrics: list[Metrics],
    makespan_weight: float,
    fatigue_weight: float,
) -> list[float]:
    makespans = [item.makespan_seconds for item in metrics]
    fatigues = [item.average_fatigue for item in metrics]
    min_makespan, max_makespan = min(makespans), max(makespans)
    min_fatigue, max_fatigue = min(fatigues), max(fatigues)

    makespan_span = max(max_makespan - min_makespan, 1e-12)
    fatigue_span = max(max_fatigue - min_fatigue, 1e-12)

    return [
        makespan_weight
        * (item.makespan_seconds - min_makespan)
        / makespan_span
        + fatigue_weight
        * (item.average_fatigue - min_fatigue)
        / fatigue_span
        for item in metrics
    ]


def optimize_plan(
    activities: list[Activity],
    loci: list[tuple[str, int]],
    population_size: int,
    generations: int,
    rest_ratio: float,
    makespan_weight: float,
    fatigue_weight: float,
    seed: int,
) -> tuple[list[int], Metrics, list[dict[str, float]]]:
    if not loci:
        raise ValueError("No eligible task boundaries found.")
    if population_size < 4:
        raise ValueError("Population size must be at least 4.")
    if generations < 1:
        raise ValueError("Generations must be at least 1.")
    if not 0.0 < rest_ratio <= 1.0:
        raise ValueError("rest_ratio must be in (0, 1].")

    weight_sum = makespan_weight + fatigue_weight
    if weight_sum <= 0.0:
        raise ValueError("At least one objective weight must be positive.")

    makespan_weight /= weight_sum
    fatigue_weight /= weight_sum

    rng = random.Random(seed)
    rest_counts: dict[str, int] = {}

    for operator in sorted({locus[0] for locus in loci}):
        operator_locus_count = sum(
            locus[0] == operator for locus in loci
        )
        rest_counts[operator] = max(
            1,
            min(
                operator_locus_count,
                round(operator_locus_count * rest_ratio),
            ),
        )

    population = [
        random_chromosome(loci, rest_counts, rng)
        for _ in range(population_size)
    ]
    history: list[dict[str, float]] = []

    for generation in range(1, generations + 1):
        metrics = [
            simulate_plan(activities, loci, chromosome)
            for chromosome in population
        ]
        costs = normalized_costs(
            metrics,
            makespan_weight,
            fatigue_weight,
        )
        ranked = sorted(range(len(population)), key=costs.__getitem__)
        best_index = ranked[0]
        best_metrics = metrics[best_index]

        history.append(
            {
                "generation": generation,
                "normalized_cost": costs[best_index],
                "makespan_minutes": (
                    best_metrics.makespan_seconds / 60.0
                ),
                "average_fatigue": best_metrics.average_fatigue,
                "maximum_fatigue": best_metrics.maximum_fatigue,
                "time_above_hy_minutes": (
                    best_metrics.time_above_hy_seconds / 60.0
                ),
                "recovery_time_minutes": (
                    best_metrics.recovery_time_seconds / 60.0
                ),
            }
        )

        elite_count = max(2, population_size // 10)
        next_population = [
            population[index].copy()
            for index in ranked[:elite_count]
        ]

        def tournament() -> list[int]:
            candidates = rng.sample(
                range(population_size),
                min(4, population_size),
            )
            winner = min(candidates, key=costs.__getitem__)
            return population[winner]

        while len(next_population) < population_size:
            parent_a = tournament()
            parent_b = tournament()
            child = [
                gene_a if rng.random() < 0.5 else gene_b
                for gene_a, gene_b in zip(
                    parent_a,
                    parent_b,
                    strict=True,
                )
            ]
            child = repair_chromosome(child, loci, rest_counts, rng)
            child = mutate(child, loci, rest_counts, rng)
            next_population.append(child)

        population = next_population

        if generation == 1 or generation % 10 == 0:
            print(
                f"Generation {generation:>3}/{generations} | "
                f"makespan={best_metrics.makespan_seconds / 60.0:.2f} min | "
                f"average fatigue={best_metrics.average_fatigue:.4f}"
            )

    final_metrics = [
        simulate_plan(activities, loci, chromosome)
        for chromosome in population
    ]
    final_costs = normalized_costs(
        final_metrics,
        makespan_weight,
        fatigue_weight,
    )
    best_index = min(range(len(population)), key=final_costs.__getitem__)
    return (
        population[best_index],
        final_metrics[best_index],
        history,
    )


def write_plan(
    path: Path,
    activities: list[Activity],
    loci: list[tuple[str, int]],
    chromosome: list[int],
) -> None:
    activity_by_key = {
        (activity.operator, activity.sequence): activity
        for activity in activities
    }
    schedule_id = activities[0].schedule_id
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "schedule_id",
                "operator",
                "after_sequence",
                "after_job_id",
                "action_id",
                "action_name",
                "expected_duration_seconds",
            ]
        )

        for key, action in zip(loci, chromosome, strict=True):
            if action == ACTION_CONTINUE:
                continue

            activity = activity_by_key[key]
            writer.writerow(
                [
                    schedule_id,
                    activity.operator,
                    activity.sequence,
                    activity.job_id,
                    action,
                    ACTION_NAMES[action],
                    f"{recovery_duration_seconds(action):.6f}",
                ]
            )


def write_history(
    path: Path,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def write_summary(
    path: Path,
    baseline_metrics: Metrics,
    optimized_metrics: Metrics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "plan",
        "makespan_minutes",
        "average_fatigue",
        "maximum_fatigue",
        "time_above_hy_minutes",
        "recovery_time_minutes",
    ]
    rows = [
        ("no_recovery_reference", baseline_metrics),
        ("static_ga_recovery", optimized_metrics),
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for name, metrics in rows:
            writer.writerow(
                {
                    "plan": name,
                    "makespan_minutes": (
                        metrics.makespan_seconds / 60.0
                    ),
                    "average_fatigue": metrics.average_fatigue,
                    "maximum_fatigue": metrics.maximum_fatigue,
                    "time_above_hy_minutes": (
                        metrics.time_above_hy_seconds / 60.0
                    ),
                    "recovery_time_minutes": (
                        metrics.recovery_time_seconds / 60.0
                    ),
                }
            )


def main() -> None:
    args = parse_arguments()
    schedule_path = args.schedule.resolve()
    activities = load_schedule(schedule_path)
    loci = eligible_boundaries(activities)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else SCHEDULE_DIR
        / f"ga_recovery_plan_{activities[0].schedule_id}.csv"
    )

    print("Static GA work-rest baseline")
    print("Schedule:", schedule_path)
    print("Activities:", len(activities))
    print("Eligible task boundaries:", len(loci))

    chromosome, metrics, history = optimize_plan(
        activities=activities,
        loci=loci,
        population_size=args.population,
        generations=args.generations,
        rest_ratio=args.rest_ratio,
        makespan_weight=args.makespan_weight,
        fatigue_weight=args.fatigue_weight,
        seed=args.seed,
    )
    no_recovery = simulate_plan(
        activities,
        loci,
        [ACTION_CONTINUE] * len(loci),
    )

    write_plan(output_path, activities, loci, chromosome)
    write_history(
        RESULTS_DIR / "static_ga_optimization_history.csv",
        history,
    )
    write_summary(
        RESULTS_DIR / "static_ga_optimization_summary.csv",
        no_recovery,
        metrics,
    )

    selected_actions = [
        action for action in chromosome if action != ACTION_CONTINUE
    ]
    print("\nOptimization completed.")
    print("Recovery points:", len(selected_actions))
    print(
        "Short / long:",
        selected_actions.count(ACTION_SHORT_BREAK),
        "/",
        selected_actions.count(ACTION_LONG_RECOVERY),
    )
    print(f"Expected makespan: {metrics.makespan_seconds / 60.0:.2f} min")
    print(f"Expected average fatigue: {metrics.average_fatigue:.4f}")
    print(f"Expected maximum fatigue: {metrics.maximum_fatigue:.4f}")
    print(
        "Expected time above Hy: "
        f"{metrics.time_above_hy_seconds / 60.0:.2f} min"
    )
    print("Plan saved to:", output_path)


if __name__ == "__main__":
    main()
