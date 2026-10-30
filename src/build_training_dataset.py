from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from rl_config import (
    ACTION_CONTINUE,
    ACTION_SHORT_BREAK,
    ACTION_LONG_RECOVERY,
    ACTION_NAMES,
    REWARD_BETA,
    STATE_COLUMNS,
    classify_region,
)


BASE_DIR = Path(__file__).resolve().parents[1]

ANYLOGIC_LOG_DIR = BASE_DIR / "data" / "anylogic_logs"
AGENT_LOG_DIR = BASE_DIR / "data" / "agent_log"
INPUT_PATTERN = "rl_decision_log_exploration_s*.csv"
LEGACY_INPUT_PATH = ANYLOGIC_LOG_DIR / "rl_decision_log_exploration.csv"
OUTPUT_PATH = AGENT_LOG_DIR / "training_transitions.csv"
EXPLORATION_SCHEDULE_PATH = (
    BASE_DIR / "schedule" / "exploration" / "schedule_training_merged.csv"
)
ANYLOGIC_MODEL_PATH = Path(
    r"C:\Users\pietr\Models\rl_agent_multi_op\rl_agent_multi_op.alp"
)


HY = 0.80
FATIGUE_WARNING = 0.75

IDLE_TASK = 0

CANDIDATE_ACTIONS = (
    ACTION_CONTINUE,
    ACTION_SHORT_BREAK,
    ACTION_LONG_RECOVERY,
)

# These values must match Operator.requestRecovery() in AnyLogic.
# Action 1 removes 40% of fatigue; action 2 removes 60%.
# Calibrated recovery rate: about 5 min for a 40% reduction and
# 9 min for a 60% reduction under R(t) = F0 * exp(-mu * t).
RECOVERY_MU_PER_SECOND = 0.0017
SHORT_RESIDUAL_FATIGUE = 0.60
LONG_RESIDUAL_FATIGUE = 0.40

# Cost weights for fatigue, makespan, and safety.
FATIGUE_AREA_WEIGHT = 8.0
FINAL_FATIGUE_WEIGHT = 4.0
WARNING_FATIGUE_COST = 15.0
SCHEDULE_DELAY_WEIGHT = 10.0
TIME_REFERENCE_SECONDS = 600.0
HY_VIOLATION_COST = 80.0
HY_EXCESS_COST = 400.0

# A recovery is redundant when natural rest is already available.
BLOCKED_TASK_RECOVERY_COST = 80.0
IDLE_RECOVERY_COST = 60.0
PRE_IDLE_RECOVERY_COST = 30.0

# Prefer the shortest action that keeps the next task below Hy.
UNNECESSARY_RECOVERY_COST = 25.0
EXCESS_RECOVERY_COST = 12.0


def clip_fatigue(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def recovery_residual_fatigue(action: int) -> float:
    if action == ACTION_SHORT_BREAK:
        return SHORT_RESIDUAL_FATIGUE
    if action == ACTION_LONG_RECOVERY:
        return LONG_RESIDUAL_FATIGUE
    return 1.0


def recovery_duration_seconds(action: int) -> float:
    residual = recovery_residual_fatigue(action)

    if residual >= 1.0:
        return 0.0

    return -math.log(residual) / RECOVERY_MU_PER_SECOND


def gotz_processing_coefficient(fatigue: float) -> float:
    """Match the fatigue-dependent processing time used by AnyLogic."""
    return 1.0 if fatigue <= 0.25 else 0.75 + fatigue


def effective_lambda_from_forecast(
    current_fatigue: float,
    predicted_fatigue: float,
    effective_duration_seconds: float,
) -> float:
    """Recover the effective task lambda from AnyLogic's next-task forecast."""
    if effective_duration_seconds <= 0.0:
        return 0.0

    remaining_capacity = max(1.0e-9, 1.0 - current_fatigue)
    fatigue_fraction = (
        predicted_fatigue - current_fatigue
    ) / remaining_capacity
    fatigue_fraction = max(0.0, min(0.999999, fatigue_fraction))

    return -math.log(1.0 - fatigue_fraction) / effective_duration_seconds


def work_fatigue_area(
    start_fatigue: float,
    intensity: float,
    duration_seconds: float,
) -> float:
    """Integrate fatigue over one productive scheduled task."""
    if duration_seconds <= 0.0:
        return 0.0

    if intensity <= 1.0e-9:
        return start_fatigue * duration_seconds

    effective_lambda = -math.log(1.0 - intensity) / duration_seconds

    return duration_seconds - (
        (1.0 - start_fatigue)
        * (1.0 - math.exp(-effective_lambda * duration_seconds))
        / effective_lambda
    )


def rest_fatigue_area(start_fatigue: float, duration_seconds: float) -> float:
    """Integrate fatigue during a recovery or scheduled idle period."""
    if duration_seconds <= 0.0:
        return 0.0

    residual = math.exp(-RECOVERY_MU_PER_SECOND * duration_seconds)
    return start_fatigue * (1.0 - residual) / RECOVERY_MU_PER_SECOND


def calculate_reward(current_row: pd.Series, next_row: pd.Series) -> dict:
    return calculate_reward_for_action(
        current_row=current_row,
        next_row=next_row,
        action=int(current_row["selectedAction"]),
        beta=REWARD_BETA,
    )


def calculate_reward_for_action(
    current_row: pd.Series,
    next_row: pd.Series,
    action: int,
    beta: float = REWARD_BETA,
) -> dict:
    """Evaluate fatigue and delay through the next scheduled activity."""
    current_fatigue = clip_fatigue(current_row["currentFatigue"])
    next_task_type = int(float(current_row["nextTaskType"]))
    second_task_type = int(float(current_row["secondTaskType"]))
    next_task_ready = int(float(current_row["nextTaskReady"]))
    next_duration_seconds = max(
        0.0,
        float(current_row["nextTaskDuration"]) * 60.0,
    )
    second_duration_seconds = max(
        0.0,
        float(current_row["secondTaskDuration"]) * 60.0,
    )

    if next_task_type == IDLE_TASK:
        baseline_activity_seconds = next_duration_seconds
        idle_residual = math.exp(
            -RECOVERY_MU_PER_SECOND * next_duration_seconds
        )
        fatigue_if_continue = clip_fatigue(current_fatigue * idle_residual)
        effective_lambda = 0.0
    else:
        fatigue_if_continue = clip_fatigue(
            current_row["predictedFatigueAfterNextTask"]
        )

        baseline_activity_seconds = (
            next_duration_seconds
            * gotz_processing_coefficient(current_fatigue)
        )
        effective_lambda = effective_lambda_from_forecast(
            current_fatigue,
            fatigue_if_continue,
            baseline_activity_seconds,
        )

    candidate_fatigues = {}
    candidate_first_duration_seconds = {}
    candidate_first_areas = {}

    for candidate_action in CANDIDATE_ACTIONS:
        candidate_recovery_seconds = recovery_duration_seconds(candidate_action)
        candidate_start = clip_fatigue(
            current_fatigue
            * recovery_residual_fatigue(candidate_action)
        )
        candidate_area = rest_fatigue_area(
            current_fatigue,
            candidate_recovery_seconds,
        )

        if next_task_type == IDLE_TASK:
            candidate_end = candidate_start * idle_residual
            first_activity_seconds = next_duration_seconds
            candidate_area += rest_fatigue_area(
                candidate_start,
                first_activity_seconds,
            )
        else:
            first_activity_seconds = (
                next_duration_seconds
                * gotz_processing_coefficient(candidate_start)
            )
            candidate_intensity = 1.0 - math.exp(
                -effective_lambda * first_activity_seconds
            )
            candidate_end = (
                candidate_start
                + (1.0 - candidate_start) * candidate_intensity
            )
            candidate_area += work_fatigue_area(
                candidate_start,
                candidate_intensity,
                first_activity_seconds,
            )

        candidate_end = clip_fatigue(candidate_end)
        candidate_fatigues[candidate_action] = candidate_end
        candidate_first_duration_seconds[candidate_action] = (
            candidate_recovery_seconds + first_activity_seconds
        )
        candidate_first_areas[candidate_action] = candidate_area

    recovery_seconds = recovery_duration_seconds(action)
    action_activity_seconds = (
        candidate_first_duration_seconds[action] - recovery_seconds
    )
    estimated_next_fatigue = candidate_fatigues[action]

    # Recover only when the next executable task would violate Hy.
    if (
        next_task_ready == 0
        or next_task_type == IDLE_TASK
        or candidate_fatigues[ACTION_CONTINUE] < HY
    ):
        minimum_sufficient_action = ACTION_CONTINUE
    elif candidate_fatigues[ACTION_SHORT_BREAK] < HY:
        minimum_sufficient_action = ACTION_SHORT_BREAK
    else:
        minimum_sufficient_action = ACTION_LONG_RECOVERY

    total_horizon_seconds = max(
        1.0,
        candidate_first_duration_seconds[action],
    )
    average_interval_fatigue = (
        candidate_first_areas[action] / total_horizon_seconds
    )

    fatigue_penalty = (
        FATIGUE_AREA_WEIGHT * (average_interval_fatigue ** 2)
        + FINAL_FATIGUE_WEIGHT * (estimated_next_fatigue ** 2)
        + WARNING_FATIGUE_COST
        * max(0.0, estimated_next_fatigue - FATIGUE_WARNING)
    )

    # Compare recovery plus the next activity against immediate continuation.
    schedule_delay_seconds = (
        candidate_first_duration_seconds[action]
        - candidate_first_duration_seconds[ACTION_CONTINUE]
    )
    schedule_delay_cost = SCHEDULE_DELAY_WEIGHT * (
        schedule_delay_seconds / TIME_REFERENCE_SECONDS
    )
    # The recovery duration is already included in schedule_delay_seconds.
    # Adding another recovery-time penalty here would double-count its effect
    # and would make long recovery artificially unattractive.
    recovery_time_cost = 0.0
    makespan_cost = schedule_delay_cost

    peak_predicted_fatigue = estimated_next_fatigue
    hy_excess = max(0.0, peak_predicted_fatigue - HY)
    safety_penalty = 0.0

    if peak_predicted_fatigue >= HY:
        safety_penalty = HY_VIOLATION_COST + HY_EXCESS_COST * hy_excess

    # Prefer the minimum recovery required by the next executable task.
    action_efficiency_penalty = 0.0
    if (
        minimum_sufficient_action == ACTION_CONTINUE
        and action != ACTION_CONTINUE
    ):
        action_efficiency_penalty = UNNECESSARY_RECOVERY_COST
    elif (
        minimum_sufficient_action == ACTION_SHORT_BREAK
        and action == ACTION_LONG_RECOVERY
    ):
        action_efficiency_penalty = EXCESS_RECOVERY_COST

    idle_recovery_penalty = 0.0
    if action != ACTION_CONTINUE and next_task_ready == 0:
        idle_recovery_penalty = BLOCKED_TASK_RECOVERY_COST
    elif action != ACTION_CONTINUE and next_task_type == IDLE_TASK:
        idle_recovery_penalty = IDLE_RECOVERY_COST
    elif (
        action != ACTION_CONTINUE
        and second_task_type == IDLE_TASK
        and candidate_fatigues[ACTION_CONTINUE] < HY
    ):
        idle_recovery_penalty = PRE_IDLE_RECOVERY_COST

    # All policies execute the same tasks. Actions are compared only through
    # fatigue, makespan contribution, and the Hy safety constraint.
    weighted_fatigue_cost = beta * fatigue_penalty
    reward = -(
        weighted_fatigue_cost
        + makespan_cost
        + safety_penalty
        + idle_recovery_penalty
        + action_efficiency_penalty
    )

    return {
        "reward": float(reward),
        "rewardBeta": float(beta),
        "fatiguePenalty": float(fatigue_penalty),
        "weightedFatigueCost": float(weighted_fatigue_cost),
        "scheduleDelayCost": float(schedule_delay_cost),
        "recoveryTimeCost": float(recovery_time_cost),
        "makespanCost": float(makespan_cost),
        "scheduleDelaySeconds": float(schedule_delay_seconds),
        "safetyPenalty": float(safety_penalty),
        "actionEfficiencyPenalty": float(action_efficiency_penalty),
        "idleRecoveryPenalty": float(idle_recovery_penalty),
        "averageIntervalFatigue": float(average_interval_fatigue),
        "recoveryDurationSeconds": float(recovery_seconds),
        "effectiveTaskDurationSeconds": float(action_activity_seconds),
        "timeDelta": float(
            max(
                1.0,
                float(next_row["decisionTime"])
                - float(current_row["decisionTime"]),
            )
        ),
        "nextTaskType": int(next_task_type),
        "secondTaskType": int(second_task_type),
        "secondTaskDuration": float(second_duration_seconds / 60.0),
        "nextTaskReady": int(next_task_ready),
        "recoveryNeeded": int(fatigue_if_continue >= HY),
        "predictedFatigueAfterNextTask": float(fatigue_if_continue),
        "estimatedNextFatigue": float(estimated_next_fatigue),
        "fatigueIfContinue": float(fatigue_if_continue),
        "fatigueIfShort": float(candidate_fatigues[ACTION_SHORT_BREAK]),
        "fatigueIfLong": float(candidate_fatigues[ACTION_LONG_RECOVERY]),
        "minimumSufficientAction": int(minimum_sufficient_action),
        "predictedHyRisk": float(
            max(
                0.0,
                candidate_fatigues[ACTION_CONTINUE] - HY,
            )
        ),
        "fatigueWarning": int(estimated_next_fatigue >= FATIGUE_WARNING),
        "fatigueCritical": int(peak_predicted_fatigue >= HY),
        "selectedRecovery": int(action != ACTION_CONTINUE),
    }


def prepare_decision_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "_sourceRun" not in df.columns:
        df["_sourceRun"] = "single"

    required_columns = [
        "decisionTime",
        "operator",
        "station",
        "mode",
        "region",
        "selectedAction",
        "actionType",
        "recordType",
        "scheduleId",
    ] + STATE_COLUMNS

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Missing columns in decision log: "
            f"{missing_columns}\n\n"
            "Update AnyLogic addDecisionLogRow(), getCurrentState(), and the CSV header."
        )

    numeric_columns = [
        "decisionTime",
        "mode",
        "selectedAction",
    ] + STATE_COLUMNS

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_columns).reset_index(drop=True)

    df["operator"] = df["operator"].astype(str).str.strip()
    df["station"] = df["station"].astype(str).str.strip()
    df["actionType"] = df["actionType"].astype(str).str.strip()
    df["recordType"] = df["recordType"].astype(str).str.strip()
    df["scheduleId"] = df["scheduleId"].astype(str).str.strip()
    df["region"] = df["currentFatigue"].apply(classify_region)

    df = df[df["selectedAction"].isin([0, 1, 2])].copy()
    # Reject legacy exploration rows generated with 1=micro, 2=short, 3=long.
    # The current action schema is 0=continue, 1=short, 2=long.
    valid_action_labels = (
        ((df["selectedAction"] == ACTION_CONTINUE)
         & df["actionType"].eq("continue_working"))
        | ((df["selectedAction"] == ACTION_SHORT_BREAK)
           & df["actionType"].eq("short_break"))
        | ((df["selectedAction"] == ACTION_LONG_RECOVERY)
           & df["actionType"].eq("long_break"))
    )
    df = df[valid_action_labels].copy()
    df = df[df["recordType"].eq("decision")].copy()
    df = df[df["region"].isin(["B", "C"])].copy()
    df = df[df["nextTaskDuration"] > 0.0].copy()

    return df.sort_values(
        ["_sourceRun", "scheduleId", "operator", "decisionTime"]
    ).reset_index(drop=True)


def build_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_decision_log(df)

    if len(df) == 0:
        raise ValueError(
            "No valid Region B/C decisions remain after filtering. "
            "Run a longer exploration simulation or check Hx/Hy."
        )

    transitions = []

    for (source_run, schedule_id, operator_name), group in df.groupby(
        ["_sourceRun", "scheduleId", "operator"],
        sort=False,
    ):
        group = group.sort_values("decisionTime").reset_index(drop=True)

        if len(group) < 2:
            continue

        for i in range(len(group) - 1):
            current_row = group.iloc[i]
            next_row = group.iloc[i + 1]
            actual_action = int(current_row["selectedAction"])
            is_terminal = i == len(group) - 2

            reward_info = calculate_reward_for_action(
                current_row=current_row,
                next_row=next_row,
                action=actual_action,
            )

            transition = {
                "decisionTime": current_row["decisionTime"],
                "sourceRun": source_run,
                "scheduleId": schedule_id,
                "operator": current_row["operator"],
                "station": current_row["station"],
                "mode": int(current_row["mode"]),
                "region": current_row["region"],
                "action": actual_action,
                "actionType": ACTION_NAMES.get(
                    actual_action,
                    str(current_row["actionType"]),
                ),
                "reward": reward_info["reward"],
                "done": int(is_terminal),
                "nextDecisionTime": next_row["decisionTime"],
                "nextRegion": next_row["region"],
            }

            for key, value in reward_info.items():
                if key != "reward":
                    transition[key] = value

            for col in STATE_COLUMNS:
                transition[f"state_{col}"] = current_row[col]
                transition[f"next_{col}"] = next_row[col]

            transitions.append(transition)

    output = pd.DataFrame(transitions)

    if len(output) == 0:
        raise ValueError(
            "No transitions were created. "
            "Each operator needs at least two valid decisions."
        )

    return output


def load_exploration_logs() -> tuple[pd.DataFrame, list[Path]]:
    input_paths = sorted(ANYLOGIC_LOG_DIR.glob(INPUT_PATTERN))

    if not input_paths and LEGACY_INPUT_PATH.exists():
        input_paths = [LEGACY_INPUT_PATH]

    if not input_paths:
        raise FileNotFoundError(
            "No exploration decision logs found. Expected files matching: "
            f"{ANYLOGIC_LOG_DIR / INPUT_PATTERN}"
        )

    frames = []

    for input_path in input_paths:
        frame = pd.read_csv(input_path)
        frame["_sourceRun"] = input_path.stem
        frames.append(frame)

    return pd.concat(frames, ignore_index=True), input_paths


def validate_exploration_logs(df: pd.DataFrame, input_paths: list[Path]) -> None:
    """Reject incomplete or uncontrolled AnyLogic exploration runs."""
    simulation_dependencies = [
        path
        for path in (ANYLOGIC_MODEL_PATH, EXPLORATION_SCHEDULE_PATH)
        if path.exists()
    ]
    if simulation_dependencies:
        simulation_timestamp = max(
            path.stat().st_mtime_ns for path in simulation_dependencies
        )
        stale_logs = [
            path.name
            for path in input_paths
            if path.stat().st_mtime_ns < simulation_timestamp
        ]
        if stale_logs:
            raise ValueError(
                "Exploration logs predate the current AnyLogic model or "
                "merged schedule. "
                "Regenerate every exploration seed before rebuilding the "
                f"dataset. Stale files: {', '.join(stale_logs)}"
            )

    required_columns = {
        "_sourceRun",
        "scheduleId",
        "queueLength",
        "completedJobs",
    }
    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        raise ValueError(
            "Exploration log validation failed. Missing columns: "
            f"{missing_columns}"
        )

    queue_length = pd.to_numeric(df["queueLength"], errors="coerce")
    maximum_queue = float(queue_length.max())

    if maximum_queue > 6.0:
        raise ValueError(
            "Exploration log validation failed: the fixed-schedule queue "
            f"reached {maximum_queue:.0f} entities. This indicates uncontrolled "
            "Source arrivals; regenerate every exploration seed in AnyLogic."
        )

    if not EXPLORATION_SCHEDULE_PATH.exists():
        raise FileNotFoundError(
            "Merged exploration schedule not found: "
            f"{EXPLORATION_SCHEDULE_PATH}"
        )

    schedule = pd.read_csv(EXPLORATION_SCHEDULE_PATH)
    schedule.columns = schedule.columns.str.strip()

    if "schedule_id" not in schedule.columns:
        raise ValueError(
            "Merged exploration schedule is missing the schedule_id column."
        )

    expected_counts = schedule.groupby("schedule_id").size().to_dict()
    expected_schedule_ids = set(expected_counts)
    if "activity_type" in schedule.columns:
        expected_task_count = int(
            schedule["activity_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("task")
            .sum()
        )
    else:
        lambda_values = (
            schedule["lambda_per_second"]
            if "lambda_per_second" in schedule.columns
            else pd.Series(0.0, index=schedule.index)
        )
        expected_task_count = int(
            pd.to_numeric(lambda_values, errors="coerce")
            .fillna(0.0)
            .gt(0.0)
            .sum()
        )

    for source_run, run in df.groupby("_sourceRun", sort=False):
        observed_schedule_ids = set(run["scheduleId"].dropna().astype(str))
        missing_schedule_ids = sorted(
            expected_schedule_ids.difference(observed_schedule_ids)
        )

        if missing_schedule_ids:
            preview = ", ".join(missing_schedule_ids[:5])
            raise ValueError(
                f"Exploration run {source_run} is incomplete: "
                f"{len(missing_schedule_ids)} schedule(s) are missing "
                f"(first: {preview}). Regenerate this seed in AnyLogic."
            )

        for schedule_id, schedule_run in run.groupby("scheduleId", sort=False):
            schedule_id = str(schedule_id)
            expected_activities = expected_counts.get(schedule_id)

            if expected_activities is None:
                raise ValueError(
                    f"Exploration run {source_run} contains unknown schedule "
                    f"{schedule_id}."
                )

            completed_jobs = pd.to_numeric(
                schedule_run["completedJobs"], errors="coerce"
            )
            maximum_completed = float(completed_jobs.max())

            if maximum_completed > expected_activities:
                raise ValueError(
                    f"Exploration run {source_run}, schedule {schedule_id}: "
                    f"completedJobs reached {maximum_completed:.0f}, but the "
                    f"schedule contains only {expected_activities} activities."
                )

        event_log_path = ANYLOGIC_LOG_DIR / (
            str(source_run).replace(
                "rl_decision_log_",
                "rl_event_log_",
                1,
            )
            + ".csv"
        )
        if not event_log_path.exists():
            raise FileNotFoundError(
                "Matching exploration event log not found: "
                f"{event_log_path}"
            )

        events = pd.read_csv(event_log_path)
        events.columns = events.columns.str.strip()
        required_event_columns = {"event", "job_id"}
        missing_event_columns = sorted(
            required_event_columns.difference(events.columns)
        )
        if missing_event_columns:
            raise ValueError(
                f"Exploration event log {event_log_path.name} is missing "
                f"columns: {missing_event_columns}"
            )

        event_names = events["event"].astype(str).str.strip().str.lower()
        job_ids = events["job_id"].astype(str).str.strip().str.lower()
        productive_tasks = job_ids.str.startswith("job_")
        task_starts = int((event_names.eq("start") & productive_tasks).sum())
        task_ends = int((event_names.eq("end") & productive_tasks).sum())

        if task_starts != expected_task_count or task_ends != expected_task_count:
            raise ValueError(
                f"Exploration run {source_run} is incomplete: expected "
                f"{expected_task_count} productive tasks, observed "
                f"{task_starts} starts and {task_ends} ends."
            )

    print(
        "Exploration log validation passed: "
        f"{len(input_paths)} run(s), {len(expected_schedule_ids)} schedules per run."
    )


def main() -> None:
    df_raw, input_paths = load_exploration_logs()

    print("Decision log loaded successfully.")
    print("Input files:")
    for input_path in input_paths:
        print(" -", input_path)
    print("Exploration runs:", len(input_paths))
    print("Raw decisions:", len(df_raw))

    validate_exploration_logs(df_raw, input_paths)

    transitions = build_training_dataset(df_raw)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(OUTPUT_PATH, index=False)

    print("\nTraining transitions created successfully.")
    print("Output file:", OUTPUT_PATH)
    print("Transitions:", len(transitions))

    print("\nTransitions by operator:")
    print(transitions["operator"].value_counts())

    print("\nAction distribution:")
    print(transitions["action"].value_counts().sort_index())

    print("\nSequential transitions used for DQN training:")
    print(len(transitions))

    print("\nExplored action distribution used for DQN training:")
    print(transitions["action"].value_counts().sort_index())

    print(f"\nReward beta: {REWARD_BETA:.3f}")

    print("\nReward summary:")
    print(transitions["reward"].describe())

    print("\nFatigue cost summary:")
    print(transitions["fatiguePenalty"].describe())

    print("\nSchedule-delay cost summary:")
    print(transitions["scheduleDelayCost"].describe())

    print("\nRecovery-time cost summary:")
    print(transitions["recoveryTimeCost"].describe())

    print("\nMakespan cost summary:")
    print(transitions["makespanCost"].describe())

    print("\nSafety cost summary:")
    print(transitions["safetyPenalty"].describe())

    print("\nNet schedule-delay summary [seconds]:")
    print(transitions["scheduleDelaySeconds"].describe())

    print("\nAverage interval fatigue summary:")
    print(transitions["averageIntervalFatigue"].describe())

    print("\nFirst rows:")
    print(transitions.head())


if __name__ == "__main__":
    main()
