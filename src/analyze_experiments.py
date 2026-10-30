from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# Fatigue prediction is currently disabled.
# The predictor module remains available, but the standard analysis pipeline
# does not import or call it.
# from fatigue_extraction_predictor import (
#     MODEL_PATH as FATIGUE_MODEL_PATH,
#     build_fatigue_prediction_models,
#     get_skill_multiplier,
#     predict_extracted_fatigue,
# )
from rl_config import ACTION_NAMES, HX, HY


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "anylogic_logs"
FIGURES_DIR = BASE_DIR / "results" / "figures"
RESULTS_DIR = BASE_DIR / "results" / "csv"
SCHEDULE_DIR = BASE_DIR / "schedule" / "simulation"
# FATIGUE_PREDICTOR_SOURCE_PATH = BASE_DIR / "src" / "fatigue_extraction_predictor.py"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_BASES = {
    "No Recovery Baseline": {
        "event_log":    "rl_event_log_no_recovery",
        "decision_log": "rl_decision_log_no_recovery",
    },
    "Threshold Recovery Baseline": {
        "event_log":    "rl_event_log_threshold_recovery",
        "decision_log": "rl_decision_log_threshold_recovery",
    },
    "Work 60 Recovery 40 Baseline": {
        "event_log":    "rl_event_log_work60_recovery40",
        "decision_log": "rl_decision_log_work60_recovery40",
    },
    "DQN Adaptive Recovery": {
        "event_log":    "rl_event_log_dqn_adaptive_recovery",
        "decision_log": "rl_decision_log_dqn_adaptive_recovery",
    },
}

OPERATOR_HUMAN_TWIN_PARAMETERS = {
    "OperatorA": {"fatigue_speed_impact": 0.40},
    "OperatorB": {"fatigue_speed_impact": 0.60},
    "OperatorC": {"fatigue_speed_impact": 0.45},
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [skip] file not found: {path.name}")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def save_figure_safely(fig: plt.Figure, path: Path, **kwargs) -> None:
    """Save through a temporary PNG to avoid Pillow overwrite failures on Windows."""
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        fig.savefig(temporary_path, **kwargs)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def to_numeric_safe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_event_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "activity" not in df.columns and "task_type" in df.columns:
        df["activity"] = df["task_type"]
    elif "activity" not in df.columns and "job_type" in df.columns:
        df["activity"] = df["job_type"]
    elif "activity" not in df.columns and "job_id" in df.columns:
        # New scheduled-task logs identify activities through job_id.
        df["activity"] = df["job_id"]

    df = to_numeric_safe(df, ["timestamp", "duration", "fatigue",
                               "queueLength", "completedJobs", "processingTime",
                               "nominalProcessingTime"])

    for col in ["operator", "activity", "event", "operatorState"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def normalize_decision_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "recordType" not in df.columns:
        df["recordType"] = "decision"

    numeric_cols = [
        "decisionTime", "operatorType", "mode", "selectedAction",
        "numberLongRecoveries", "numberAdaptiveRecoveries", "timeSinceLastRecovery",
        "currentFatigue", "distanceFromHx", "distanceFromHy", "timeToReachHy",
        "queueLength", "completedJobs", "completedJobsA", "completedJobsB",
        "completedJobsC",
        "lastServiceTime", "operatorAOnBreak", "operatorBOnBreak",
        "operatorCOnBreak",
        "otherOperatorOnBreak", "numberOperatorsOnBreak", "workTime",
        "tdNormCumsumCurrentShift", "lastJobHadBreak",
    ]
    df = to_numeric_safe(df, numeric_cols)

    for col in ["operator", "station", "region", "actionType", "recordType"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "decisionTime" in df.columns:
        df = df.dropna(subset=["decisionTime"]).sort_values("decisionTime").reset_index(drop=True)

    if "selectedAction" in df.columns:
        df["selectedActionName"] = df["selectedAction"].map(ACTION_NAMES).fillna("unknown")
        df.loc[df["recordType"] != "decision", "selectedActionName"] = "observation"

    return df

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def estimate_time_exposure(event_df: pd.DataFrame, threshold: float) -> float:
    """
    Estimate time above a fatigue threshold using linear interpolation between
    consecutive logged events (trapezoidal with exact crossing detection).
    NOTE: the event log has gaps up to ~18 min so this is an approximation;
    for an exact value add a timeAboveHy accumulator variable in AnyLogic.
    """
    if event_df is None or event_df.empty:
        return 0.0
    if not {"timestamp", "operator", "fatigue"}.issubset(event_df.columns):
        return 0.0

    total = 0.0
    for _, group in event_df.groupby("operator"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < 2:
            continue
        ts = group["timestamp"].to_numpy(float)
        fa = group["fatigue"].to_numpy(float)
        for i in range(len(ts) - 1):
            dt, f0, f1 = ts[i + 1] - ts[i], fa[i], fa[i + 1]
            if f0 >= threshold and f1 >= threshold:
                total += dt
            elif f0 >= threshold:
                total += dt * (f0 - threshold) / (f0 - f1) if f0 != f1 else 0.0
            elif f1 >= threshold:
                total += dt * (f1 - threshold) / (f1 - f0) if f1 != f0 else 0.0
    return total


def estimate_total_observed_time(event_df: pd.DataFrame) -> float:
    """Return the summed observation horizon across all operators."""
    if event_df is None or event_df.empty:
        return 0.0
    if not {"timestamp", "operator"}.issubset(event_df.columns):
        return 0.0

    total = 0.0
    for _, group in event_df.groupby("operator"):
        timestamps = group["timestamp"].dropna().to_numpy(dtype=float)
        if len(timestamps) >= 2:
            total += max(float(timestamps.max() - timestamps.min()), 0.0)
    return total


def estimate_working_fatigue(event_df: pd.DataFrame) -> tuple[float, float]:
    """Estimate fatigue only during productive task execution intervals."""
    if event_df is None or event_df.empty:
        return 0.0, 0.0

    df = normalize_event_log(event_df)
    required = {"timestamp", "operator", "activity", "event", "fatigue"}
    if not required.issubset(df.columns):
        return 0.0, 0.0

    weighted_fatigue = 0.0
    working_time = 0.0

    for _, group in df.groupby("operator"):
        open_tasks: dict[str, list[pd.Series]] = {}
        for _, row in group.sort_values("timestamp").iterrows():
            activity = str(row["activity"])
            if not re.fullmatch(r"job_\d+_task_\d+_[A-D]", activity):
                continue

            event = str(row["event"]).lower()
            if event == "start":
                open_tasks.setdefault(activity, []).append(row)
            elif event == "end" and open_tasks.get(activity):
                start = open_tasks[activity].pop(0)
                duration = float(row["timestamp"] - start["timestamp"])
                if duration <= 0.0:
                    continue
                interval_average = 0.5 * (
                    float(start["fatigue"]) + float(row["fatigue"])
                )
                weighted_fatigue += interval_average * duration
                working_time += duration

    average = weighted_fatigue / working_time if working_time > 0.0 else 0.0
    return float(average), float(working_time)


def estimate_time_weighted_average_fatigue(event_df: pd.DataFrame) -> float:
    """
    Estimate average fatigue over time, not over log rows.

    This avoids penalizing scenarios that produce more jobs and therefore
    generate more log events.
    """
    if event_df is None or event_df.empty:
        return 0.0

    if not {"timestamp", "operator", "fatigue"}.issubset(event_df.columns):
        return 0.0

    weighted_sum = 0.0
    total_time = 0.0

    for _, group in event_df.groupby("operator"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        if len(group) < 2:
            continue

        timestamps = group["timestamp"].to_numpy(float)
        fatigue = group["fatigue"].to_numpy(float)

        for i in range(len(group) - 1):
            dt = timestamps[i + 1] - timestamps[i]

            if dt <= 0:
                continue

            # Trapezoidal average fatigue during the interval.
            interval_avg_fatigue = 0.5 * (fatigue[i] + fatigue[i + 1])

            weighted_sum += interval_avg_fatigue * dt
            total_time += dt

    if total_time <= 0:
        return 0.0

    return float(weighted_sum / total_time)


def count_recovery_events(event_df: pd.DataFrame) -> dict:
    empty = {"short_break_count": 0, "long_break_count": 0,
             "total_recovery_time": 0.0,
             "idle_count": 0,
             "total_idle_time": 0.0,
             "total_rest_time": 0.0}

    if event_df is None or event_df.empty:
        return empty
    if "activity" not in event_df.columns or "event" not in event_df.columns:
        return empty

    # Use START rows: AnyLogic records planned duration there and counts
    # recoveries that begin near shift-end even without a matching END row.
    starts = event_df[
        event_df["activity"].str.contains("recovery", case=False, na=False)
        & event_df["event"].eq("start")
    ].copy()

    if starts.empty:
        starts = pd.DataFrame(columns=event_df.columns)

    idle_starts = event_df[
        event_df["activity"].str.contains("idle", case=False, na=False)
        & event_df["event"].eq("start")
    ].copy()
    idle_ends = event_df[
        event_df["activity"].str.contains("idle", case=False, na=False)
        & event_df["event"].eq("end")
    ].copy()

    state_col = "operatorState" if "operatorState" in starts.columns else None
    short = long_ = 0
    if state_col:
        short  = int(starts[state_col].eq("short_break").sum())
        long_  = int(starts[state_col].eq("long_break").sum())

    total_recovery_time = float(starts["duration"].fillna(0.0).sum())
    # Idle START rows have duration zero; END rows contain elapsed idle time.
    total_idle_time = float(idle_ends["duration"].fillna(0.0).sum())

    return {
        "short_break_count":      short,
        "long_break_count":       long_,
        "total_recovery_time":    total_recovery_time,
        "idle_count":             int(len(idle_starts)),
        "total_idle_time":        total_idle_time,
        "total_rest_time":        total_recovery_time + total_idle_time,
    }


def extract_recovery_intervals(event_df: pd.DataFrame) -> dict[str, list[tuple[float, float]]]:
    """Return {operator: [(start, end), ...]} for all recovery and idle windows."""
    if event_df is None or event_df.empty:
        return {}
    df = event_df.copy()
    if "activity" not in df.columns and "task_type" in df.columns:
        df["activity"] = df["task_type"]
    elif "activity" not in df.columns and "job_type" in df.columns:
        df["activity"] = df["job_type"]
    if "activity" not in df.columns or "event" not in df.columns:
        return {}

    rec = df[
        df["activity"].str.contains("recovery|idle", case=False, na=False, regex=True)
    ].copy()
    intervals: dict[str, list[tuple[float, float]]] = {}

    for operator, group in rec.groupby("operator"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        open_s = open_d = None
        for _, row in group.iterrows():
            ev  = str(row["event"]).strip().lower()
            ts  = float(row["timestamp"])
            dur = float(row["duration"]) if pd.notna(row.get("duration")) else None
            if ev == "start":
                open_s, open_d = ts, dur
            elif ev == "end" and open_s is not None:
                if ts > open_s:
                    intervals.setdefault(operator, []).append((open_s, ts))
                open_s = open_d = None
        # Rest started near shift end with no END row: use planned duration.
        if open_s is not None and open_d:
            intervals.setdefault(operator, []).append((open_s, open_s + open_d))

    return intervals


def compute_pairwise_overlap(
    intervals_a: list[tuple[float, float]],
    intervals_b: list[tuple[float, float]],
) -> float:
    intervals_a, intervals_b = sorted(intervals_a), sorted(intervals_b)
    total, i, j = 0.0, 0, 0
    while i < len(intervals_a) and j < len(intervals_b):
        sa, ea = intervals_a[i]
        sb, eb = intervals_b[j]
        overlap = min(ea, eb) - max(sa, sb)
        if overlap > 0:
            total += overlap
        if ea < eb:
            i += 1
        else:
            j += 1
    return total


def compute_multi_operator_overlap(
    intervals_by_operator: dict[str, list[tuple[float, float]]]
) -> float:
    events: list[tuple[float, int]] = []

    for intervals in intervals_by_operator.values():
        for start, end in intervals:
            if end > start:
                events.append((float(start), 1))
                events.append((float(end), -1))

    if not events:
        return 0.0

    events.sort(key=lambda item: (item[0], -item[1]))
    active = 0
    previous_time = events[0][0]
    total = 0.0

    for current_time, delta in events:
        if current_time > previous_time and active >= 2:
            total += current_time - previous_time
        active += delta
        previous_time = current_time

    return total


def compute_simultaneous_recovery_time(event_df: pd.DataFrame) -> float:
    ivs = extract_recovery_intervals(event_df)
    return compute_multi_operator_overlap(ivs)


def compute_rest_overlap_components(event_df: pd.DataFrame) -> dict[str, float]:
    """Split simultaneous rest into controllable and scheduled components."""
    if event_df is None or event_df.empty:
        return {
            "simultaneous_recovery_recovery_time_min": 0.0,
            "simultaneous_recovery_idle_time_min": 0.0,
            "simultaneous_idle_idle_time_min": 0.0,
        }

    df = normalize_event_log(event_df)
    rests = df[
        df["activity"].str.contains(
            "recovery|idle",
            case=False,
            na=False,
            regex=True,
        )
    ].copy()
    events: list[tuple[float, int, str, str]] = []

    for operator, group in rests.groupby("operator"):
        open_rest: tuple[float, str] | None = None

        for _, row in group.sort_values("timestamp").iterrows():
            event = str(row["event"]).strip().lower()
            timestamp = float(row["timestamp"])
            rest_type = (
                "recovery"
                if "recovery" in str(row["activity"]).lower()
                else "idle"
            )

            if event == "start":
                open_rest = (timestamp, rest_type)
            elif event == "end" and open_rest is not None:
                start, started_type = open_rest
                if timestamp > start:
                    events.append((start, 1, started_type, operator))
                    events.append((timestamp, -1, started_type, operator))
                open_rest = None

    if not events:
        return {
            "simultaneous_recovery_recovery_time_min": 0.0,
            "simultaneous_recovery_idle_time_min": 0.0,
            "simultaneous_idle_idle_time_min": 0.0,
        }

    events.sort(key=lambda item: (item[0], -item[1]))
    active: list[tuple[str, str]] = []
    previous_time = events[0][0]
    totals = {
        "simultaneous_recovery_recovery_time_min": 0.0,
        "simultaneous_recovery_idle_time_min": 0.0,
        "simultaneous_idle_idle_time_min": 0.0,
    }

    for current_time, delta, rest_type, operator in events:
        elapsed = current_time - previous_time

        if elapsed > 0.0 and len(active) >= 2:
            recovery_count = sum(
                active_type == "recovery"
                for active_type, _ in active
            )
            idle_count = len(active) - recovery_count

            if recovery_count > 0 and idle_count > 0:
                key = "simultaneous_recovery_idle_time_min"
            elif recovery_count >= 2:
                key = "simultaneous_recovery_recovery_time_min"
            else:
                key = "simultaneous_idle_idle_time_min"

            totals[key] += elapsed

        active_item = (rest_type, operator)
        if delta > 0:
            active.append(active_item)
        elif active_item in active:
            active.remove(active_item)

        previous_time = current_time

    return totals


def productive_task_end_mask(df: pd.DataFrame) -> pd.Series:
    """Identify completed productive tasks and reject idle/recovery/null rows."""
    activity = df.get("activity", pd.Series("", index=df.index)).astype(str).str.lower()
    state = df.get("operatorState", pd.Series("", index=df.index)).astype(str).str.lower()
    processing_time = pd.to_numeric(
        df.get("processingTime", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0)
    return (
        df["event"].astype(str).str.lower().eq("end")
        & (processing_time > 0.0)
        & ~activity.str.contains("idle|recovery|invalid|(?:^|_)null(?:_|$)", na=False)
        & ~state.eq("idle")
    )


def compute_actual_makespan(event_df: pd.DataFrame) -> float:
    """Return the completion time of the last productive scheduled task."""
    if event_df is None or event_df.empty:
        return 0.0

    df = normalize_event_log(event_df)
    required = {"timestamp", "event", "processingTime"}
    if not required.issubset(df.columns):
        return float(df["timestamp"].max()) if "timestamp" in df.columns else 0.0

    is_task_end = productive_task_end_mask(df)
    task_ends = df.loc[is_task_end, "timestamp"]

    # A malformed or legacy log may not contain processing times for task ends.
    return float(task_ends.max()) if not task_ends.empty else float(df["timestamp"].max())


def summarize_event_log(event_df: pd.DataFrame | None) -> dict:
    if event_df is None or event_df.empty:
        return {}

    df = normalize_event_log(event_df)
    shift  = float(df["timestamp"].max() - df["timestamp"].min())
    cj     = int(productive_task_end_mask(df).sum())
    aq     = float(df["queueLength"].mean())   if "queueLength"   in df.columns else 0.0
    mq     = float(df["queueLength"].max())    if "queueLength"   in df.columns else 0.0
    af     = estimate_time_weighted_average_fatigue(df) if "fatigue" in df.columns else 0.0
    awf, working_time = estimate_working_fatigue(df)
    mf     = float(df["fatigue"].max())        if "fatigue"       in df.columns else 0.0
    t_hx   = estimate_time_exposure(df, HX)
    t_hy   = estimate_time_exposure(df, HY)
    rec    = count_recovery_events(df)
    tr     = rec["total_recovery_time"]
    thr    = cj / (shift / 60.0) if shift > 0 else 0.0
    jpm    = cj / tr              if tr   > 0 else float(cj)
    total_rest = rec["total_rest_time"]
    jpr    = cj / total_rest      if total_rest > 0 else float(cj)
    observed_time = estimate_total_observed_time(df)
    frr    = t_hy / observed_time if observed_time > 0 else 0.0
    overlap = compute_rest_overlap_components(df)
    sim = sum(overlap.values())
    makespan = compute_actual_makespan(df)

    return {
        "shift_duration_min":           shift,
        "makespan_min":                 makespan,
        "completed_jobs":               cj,
        "throughput_jobs_per_hour":     thr,
        "avg_queue_length":             aq,
        "max_queue_length":             mq,
        "avg_fatigue":                  af,
        "avg_working_fatigue":          awf,
        "total_working_time_min":       working_time,
        "max_fatigue":                  mf,
        "time_above_Hx_min":            t_hx,
        "time_above_Hy_min":            t_hy,
        "fatigue_risk_ratio":           frr,
        "total_recovery_time_min":      tr,
        "total_idle_time_min":          rec["total_idle_time"],
        "total_rest_time_min":          rec["total_rest_time"],
        "simultaneous_recovery_time_min": sim,
        "jobs_per_recovery_minute":     jpm,
        "jobs_per_rest_minute":         jpr,
        **overlap,
        **rec,
    }


def summarize_decision_log(decision_df: pd.DataFrame | None) -> dict:
    if decision_df is None or decision_df.empty:
        return {}

    df = normalize_decision_log(decision_df)
    df = df[df["recordType"] == "decision"].copy()

    sim_breaks  = int((df["numberOperatorsOnBreak"] >= 2).sum()) if "numberOperatorsOnBreak" in df.columns else 0
    risky       = 0
    if {"otherOperatorOnBreak", "selectedAction"}.issubset(df.columns):
        risky = int(((df["otherOperatorOnBreak"] == 1) & (df["selectedAction"] != 0)).sum())

    sa = df["selectedAction"] if "selectedAction" in df.columns else pd.Series([], dtype=float)

    return {
        "total_decisions":                      len(df),
        "decision_continue_count":              int((sa == 0).sum()),
        "decision_short_break_count":           int((sa == 1).sum()),
        "decision_long_break_count":            int((sa == 2).sum()),
        "risky_break_attempts_when_other_on_break": risky,
        "simultaneous_break_decisions":         sim_breaks,
        **compute_human_twin_health_summary(df),
    }


def compute_human_twin_health_score(fatigue: float, work_time: float,
                                     fatigue_speed_impact: float) -> float:
    fatigue   = float(np.clip(fatigue,   0.0, 1.0))
    work_time = max(float(work_time), 0.0)
    total_risk = (
        0.50 * fatigue
        + 0.25 * float(np.clip(fatigue / HY, 0.0, 1.0))
        + 0.15 * float(np.clip(work_time / 120.0, 0.0, 1.0))
        + 0.10 * float(np.clip((fatigue * fatigue_speed_impact) / 0.70, 0.0, 1.0))
    )
    return float(np.clip(100.0 * (1.0 - total_risk), 0.0, 100.0))


def compute_human_twin_health_summary(decision_df: pd.DataFrame) -> dict:
    required = {"operator", "currentFatigue", "workTime"}
    if decision_df is None or decision_df.empty or not required.issubset(decision_df.columns):
        return {"avg_hths": 0.0, "min_hths": 0.0,
                "hths_below_40_decisions": 0,
                "avg_hths_operator_a": 0.0, "avg_hths_operator_b": 0.0,
                "avg_hths_operator_c": 0.0}

    df = decision_df.copy()
    df["hths"] = df.apply(
        lambda row: compute_human_twin_health_score(
            row.get("currentFatigue", 0.0),
            row.get("workTime", 0.0),
            OPERATOR_HUMAN_TWIN_PARAMETERS.get(
                str(row.get("operator", "")).strip(),
                {"fatigue_speed_impact": 0.50}
            )["fatigue_speed_impact"],
        ),
        axis=1,
    )
    avg_a = float(df.loc[df["operator"] == "OperatorA", "hths"].mean()) if (df["operator"] == "OperatorA").any() else 0.0
    avg_b = float(df.loc[df["operator"] == "OperatorB", "hths"].mean()) if (df["operator"] == "OperatorB").any() else 0.0
    avg_c = float(df.loc[df["operator"] == "OperatorC", "hths"].mean()) if (df["operator"] == "OperatorC").any() else 0.0

    return {
        "avg_hths":               float(df["hths"].mean()),
        "min_hths":               float(df["hths"].min()),
        "hths_below_40_decisions": int((df["hths"] < 40.0).sum()),
        "avg_hths_operator_a":    avg_a,
        "avg_hths_operator_b":    avg_b,
        "avg_hths_operator_c":    avg_c,
    }


def compute_human_centric_index(row: dict) -> float:
    score  =  2.0  * row.get("completed_jobs", 0.0)
    score -= 0.8   * row.get("avg_queue_length", 0.0)
    score -= 3.0   * row.get("time_above_Hy_min", 0.0)
    score -= 0.25  * row.get("total_recovery_time_min", 0.0)
    score -= 25.0  * row.get("risky_break_attempts_when_other_on_break", 0.0)
    return float(score)

# ---------------------------------------------------------------------------
# CSV side-tables
# ---------------------------------------------------------------------------

def save_action_tables(experiment_name: str, decision_df: pd.DataFrame | None) -> None:
    if decision_df is None or decision_df.empty:
        return
    df = normalize_decision_log(decision_df)
    df = df[df["recordType"] == "decision"].copy()
    if "selectedActionName" not in df.columns:
        return

    safe = experiment_name.replace(" ", "_")
    (df["selectedActionName"]
        .value_counts()
        .rename_axis("action")
        .reset_index(name="count")
        .to_csv(RESULTS_DIR / f"{safe}_action_counts.csv", index=False))

    for col, label in [("operator", "operator"), ("region", "region"),
                        ("otherOperatorOnBreak", "other_on_break")]:
        if col in df.columns:
            pd.crosstab(df[col], df["selectedActionName"]).to_csv(
                RESULTS_DIR / f"{safe}_actions_by_{label}.csv")

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _bar_chart(
    df: pd.DataFrame, x_col: str, y_col: str,
    title: str, ylabel: str, filename: str,
    color: str = "#3867d6",
    std_col: str | None = None,
    nonnegative: bool = True,
) -> None:
    """Save a bar chart with optional standard deviation."""
    label_map = {
        "No Recovery Baseline": "No Recovery\nBaseline",
        "Threshold Recovery Baseline": "Threshold Recovery\nBaseline",
        "Work 60 Recovery 40 Baseline": "Work 60 / Recovery 40\nBaseline",
        "DQN Adaptive Recovery": "DQN Adaptive\nRecovery",
    }
    x_labels = [label_map.get(value, value) for value in df[x_col].astype(str)]
    figure_width = max(9.5, 2.35 * len(x_labels))
    fig, ax = plt.subplots(figsize=(figure_width, 5.8))
    y_vals = df[y_col]
    yerr = None

    if std_col and std_col in df.columns:
        std = df[std_col].fillna(0.0).clip(lower=0.0)
        lower = (
            np.minimum(std, y_vals.clip(lower=0.0))
            if nonnegative
            else std
        )
        yerr = np.vstack([lower.to_numpy(), std.to_numpy()])

    x = np.arange(len(x_labels))
    bars = ax.bar(
        x,
        y_vals,
        width=0.62,
        color=color,
        yerr=yerr,
        capsize=5,
        error_kw={"elinewidth": 1.5, "ecolor": "#444"},
    )
    ax.set_title(title, pad=16)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_xticks(x, x_labels)
    ax.tick_params(axis="x", labelsize=9, pad=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    upper_values = []
    for idx, (bar, val) in enumerate(zip(bars, y_vals)):
        upper_error = yerr[1, idx] if yerr is not None else 0.0
        upper_values.append(float(val + upper_error))
        value_label = (
            f"{val:.2f} ± {upper_error:.2f}"
            if yerr is not None
            else f"{val:.2f}"
        )
        ax.annotate(
            value_label,
            xy=(bar.get_x() + bar.get_width() / 2.0, val + upper_error),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
            clip_on=False,
        )

    if upper_values:
        current_bottom, _ = ax.get_ylim()
        chart_top = max(upper_values)
        ax.set_ylim(current_bottom, max(1.0e-6, chart_top * 1.20))

    fig.tight_layout(pad=1.6)
    save_figure_safely(fig, FIGURES_DIR / filename, dpi=200)
    plt.close(fig)
    print(f"  saved: figures/{filename}")


def _average_fatigue_chart(kpi: pd.DataFrame) -> None:
    """Average fatigue with clear policy colors and mean ± standard deviation."""
    chart_df = kpi[["experiment", "avg_fatigue"]].copy()
    if "avg_fatigue_std" in kpi.columns:
        chart_df["std"] = kpi["avg_fatigue_std"].fillna(0.0).clip(lower=0.0)
    else:
        chart_df["std"] = 0.0

    label_map = {
        "No Recovery Baseline": "No Recovery\nBaseline",
        "Work 60 Recovery 40 Baseline": "Work 60 / Recovery 40\nBaseline",
        "Threshold Recovery Baseline": "Threshold Recovery\nBaseline",
        "Work 20 Recovery 80 Baseline": "Work 20 / Recovery 80\nBaseline",
        "DQN Adaptive Recovery": "DQN Adaptive\nRecovery",
    }
    labels = [label_map.get(value, value) for value in chart_df["experiment"]]
    colors = [
        "#8854d0" if value == "DQN Adaptive Recovery" else "#20bf6b"
        for value in chart_df["experiment"]
    ]

    means = chart_df["avg_fatigue"].to_numpy(dtype=float)
    std = chart_df["std"].to_numpy(dtype=float)
    lower_error = np.minimum(std, np.maximum(means, 0.0))
    yerr = np.vstack([lower_error, std])
    x = np.arange(len(chart_df))

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bars = ax.bar(
        x,
        means,
        width=0.64,
        color=colors,
        yerr=yerr if np.any(std > 0.0) else None,
        capsize=6,
        error_kw={"elinewidth": 1.5, "ecolor": "#3d3d3d", "capthick": 1.5},
    )

    max_value = float(np.max(means + std)) if len(means) else 1.0
    upper_limit = max(0.20, min(1.15, max_value * 1.22))
    label_offset = max(0.012, upper_limit * 0.018)

    for bar, mean, deviation in zip(bars, means, std):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean + deviation + label_offset,
            f"{mean:.3f} ± {deviation:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
        )

    ax.set_title("Average Fatigue by Experiment", pad=16)
    ax.set_ylabel("Time-weighted average fatigue")
    ax.set_xlabel("")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, upper_limit)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=1.4)
    save_figure_safely(
        fig,
        FIGURES_DIR / "average_fatigue_by_experiment.png",
        dpi=220,
    )
    plt.close(fig)
    print("  saved: figures/average_fatigue_by_experiment.png")


def _average_working_fatigue_chart(kpi: pd.DataFrame) -> None:
    """Average fatigue measured only while productive tasks are executing."""
    if "avg_working_fatigue" not in kpi.columns:
        return

    chart_df = kpi[["experiment", "avg_working_fatigue"]].copy()
    std_column = "avg_working_fatigue_std"
    chart_df["std"] = (
        kpi[std_column].fillna(0.0).clip(lower=0.0)
        if std_column in kpi.columns
        else 0.0
    )
    label_map = {
        "No Recovery Baseline": "No Recovery\nBaseline",
        "Threshold Recovery Baseline": "Threshold Recovery\nBaseline",
        "Work 60 Recovery 40 Baseline": "Work 60 / Recovery 40\nBaseline",
        "DQN Adaptive Recovery": "DQN Adaptive\nRecovery",
    }
    labels = [label_map.get(value, value) for value in chart_df["experiment"]]
    colors = [
        "#8854d0" if value == "DQN Adaptive Recovery" else "#20bf6b"
        for value in chart_df["experiment"]
    ]
    means = chart_df["avg_working_fatigue"].to_numpy(dtype=float)
    std = chart_df["std"].to_numpy(dtype=float)
    x = np.arange(len(chart_df))

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bars = ax.bar(
        x,
        means,
        width=0.64,
        color=colors,
        yerr=std if np.any(std > 0.0) else None,
        capsize=6,
        error_kw={"elinewidth": 1.5, "ecolor": "#3d3d3d", "capthick": 1.5},
    )
    upper_limit = max(0.20, min(1.15, float(np.max(means + std)) * 1.22))
    label_offset = max(0.012, upper_limit * 0.018)
    for bar, mean, deviation in zip(bars, means, std):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean + deviation + label_offset,
            f"{mean:.3f} ± {deviation:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
        )

    ax.axhline(HY, color="#111111", linestyle="--", linewidth=1.2, label="Hy")
    ax.set_title("Average Working Fatigue by Experiment", pad=16)
    ax.set_ylabel("Task-time-weighted average fatigue")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, upper_limit)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout(pad=1.4)
    save_figure_safely(
        fig,
        FIGURES_DIR / "average_working_fatigue_by_experiment.png",
        dpi=220,
    )
    plt.close(fig)
    print("  saved: figures/average_working_fatigue_by_experiment.png")


def _grouped_action_chart(kpi: pd.DataFrame) -> None:
    """Grouped bar chart: number of decisions per action type per experiment."""
    action_cols = [c for c in [
        "decision_continue_count", "decision_short_break_count",
        "decision_long_break_count",
    ] if c in kpi.columns]

    df = kpi[["experiment"] + action_cols].copy().set_index("experiment")
    df.columns = [c.replace("decision_", "").replace("_count", "") for c in df.columns]

    fig, ax = plt.subplots(figsize=(9, 5))
    df.plot(kind="bar", ax=ax)
    ax.set_title("Recovery Decision Distribution by Experiment")
    ax.set_ylabel("Number of decisions")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Action")
    fig.tight_layout()
    save_figure_safely(fig, FIGURES_DIR / "decision_distribution_by_experiment.png", dpi=200)
    plt.close(fig)
    print("  saved: figures/decision_distribution_by_experiment.png")


def _planned_schedule_makespan() -> float | None:
    """Read the ideal makespan of the input schedule, without recovery delays."""
    candidates = [
        SCHEDULE_DIR / "schedule_v1_01.csv",
        SCHEDULE_DIR / "fixed_task_schedule.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    schedule_df = read_csv_if_exists(path)
    if schedule_df is None or schedule_df.empty:
        return None

    schedule_df = to_numeric_safe(schedule_df, ["scheduled_end_seconds", "processing_time_seconds"])
    if "scheduled_end_seconds" in schedule_df.columns:
        last_end = schedule_df["scheduled_end_seconds"].dropna()
        if not last_end.empty:
            return float(last_end.max()) / 60.0

    # Fallback for simple schedules without planned timestamps.
    if {"operator", "processing_time_seconds"}.issubset(schedule_df.columns):
        per_operator = schedule_df.groupby("operator")["processing_time_seconds"].sum()
        if not per_operator.empty:
            return float(per_operator.max()) / 60.0

    return None


def _makespan_comparison_chart(kpi: pd.DataFrame) -> None:
    """Compare the actual makespan of each recovery policy."""
    if "makespan_min" not in kpi.columns:
        print("  [skip] makespan chart - makespan KPI unavailable")
        return

    chart_rows: list[dict] = []
    for _, row in kpi.iterrows():
        chart_rows.append({
            "scenario": str(row["experiment"]),
            "makespan_min": float(row["makespan_min"]),
            "makespan_min_std": float(row.get("makespan_min_std", 0.0) or 0.0),
            "kind": "dqn" if row["experiment"] == "DQN Adaptive Recovery" else "baseline",
        })

    chart_df = pd.DataFrame(chart_rows)
    chart_df.to_csv(RESULTS_DIR / "makespan_by_experiment.csv", index=False)

    colors = {
        "baseline": "#20bf6b",
        "dqn": "#8854d0",
    }
    label_map = {
        "No Recovery Baseline": "No Recovery\nBaseline",
        "Work 60 Recovery 40 Baseline": "Work 60 / Recovery 40\nBaseline",
        "Threshold Recovery Baseline": "Threshold Recovery\nBaseline",
        "Work 20 Recovery 80 Baseline": "Work 20 / Recovery 80\nBaseline",
        "DQN Adaptive Recovery": "DQN Adaptive\nRecovery",
    }
    labels = [
        label_map.get(value, value)
        for value in chart_df["scenario"].astype(str)
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    values = chart_df["makespan_min"].to_numpy(float)
    std = chart_df["makespan_min_std"].clip(lower=0.0).to_numpy(float)
    x = np.arange(len(chart_df))
    bars = ax.bar(
        x,
        values,
        width=0.64,
        color=[colors[kind] for kind in chart_df["kind"]],
        yerr=std if np.any(std > 0.0) else None,
        capsize=6,
        error_kw={"elinewidth": 1.5, "ecolor": "#3d3d3d", "capthick": 1.5},
    )

    max_value = float(np.max(values + std)) if len(values) else 1.0
    upper_limit = max(1.0, max_value * 1.18)
    label_offset = max(2.0, upper_limit * 0.018)

    ax.set_title("Makespan by Schedule and Recovery Policy", pad=16)
    ax.set_ylabel("Makespan [minutes]")
    ax.set_xlabel("")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, upper_limit)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value, error in zip(bars, values, std):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + error + label_offset,
            f"{value:.2f} ± {error:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
        )

    fig.tight_layout(pad=1.4)
    save_figure_safely(fig, FIGURES_DIR / "makespan_by_experiment.png", dpi=220)
    plt.close(fig)
    print("  saved: figures/makespan_by_experiment.png")


def _fatigue_vs_makespan_chart() -> None:
    """Plot average fatigue against actual makespan for every available seed."""
    rows: list[dict] = []

    for experiment_name, paths in EXPERIMENT_BASES.items():
        event_paths = discover_seed_files(paths["event_log"])

        for event_path in event_paths:
            match = re.search(r"_s(\d+)$", event_path.stem)
            seed = int(match.group(1)) if match else event_path.stem
            event_df = read_csv_if_exists(event_path)
            summary = summarize_event_log(event_df)

            if not summary:
                continue

            rows.append({
                "experiment": experiment_name,
                "seed": seed,
                "average_fatigue": float(summary.get("avg_fatigue", np.nan)),
                "makespan_min": float(summary.get("makespan_min", np.nan)),
            })

    chart_df = pd.DataFrame(rows).dropna(
        subset=["average_fatigue", "makespan_min"]
    )

    if chart_df.empty:
        print("  [skip] fatigue vs makespan chart - no valid event logs")
        return

    output_csv = RESULTS_DIR / "fatigue_vs_makespan_by_seed.csv"
    chart_df.to_csv(output_csv, index=False)

    colors = {
        "No Recovery Baseline": "#20bf6b",
        "Work 60 Recovery 40 Baseline": "#26de81",
        "Threshold Recovery Baseline": "#0fb9b1",
        "DQN Adaptive Recovery": "#7c3aed",
    }
    legend_names = {
        "No Recovery Baseline": "No Recovery",
        "Work 60 Recovery 40 Baseline": "GA",
        "Threshold Recovery Baseline": "Threshold Recovery",
        "DQN Adaptive Recovery": "DQN Adaptive Recovery",
    }

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    mean_points = []

    for experiment_name, group in chart_df.groupby("experiment", sort=False):
        color = colors.get(experiment_name, "#64748b")
        ax.scatter(
            group["average_fatigue"],
            group["makespan_min"],
            s=58,
            alpha=0.82,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            label=legend_names.get(experiment_name, experiment_name),
        )

        mean_fatigue = group["average_fatigue"].mean()
        mean_makespan = group["makespan_min"].mean()
        ax.scatter(
            [mean_fatigue],
            [mean_makespan],
            s=190,
            marker="*",
            color=color,
            edgecolors="#202020",
            linewidths=0.8,
            zorder=5,
        )
        mean_points.append((
            experiment_name,
            float(mean_fatigue),
            float(mean_makespan),
            color,
        ))

    x_values = chart_df["average_fatigue"].to_numpy(float)
    y_values = chart_df["makespan_min"].to_numpy(float)
    ax.set_xlim(float(x_values.min()) - 0.02, float(x_values.max()) + 0.02)
    ax.set_ylim(float(y_values.min()) - 8, float(y_values.max()) + 9)
    label_y = float(y_values.min() + 0.56 * (y_values.max() - y_values.min()))

    for experiment_name, mean_fatigue, mean_makespan, color in mean_points:
        ax.annotate(
            f"{legend_names.get(experiment_name, experiment_name)}\n★  Mean",
            xy=(mean_fatigue, mean_makespan),
            xytext=(mean_fatigue, label_y),
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="semibold",
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.38",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 1.2,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 1.3,
                "shrinkA": 5,
                "shrinkB": 8,
            },
            zorder=6,
        )

    ax.set_title("Average Fatigue versus Makespan by Seed")
    ax.set_xlabel("Average fatigue")
    ax.set_ylabel("Makespan [minutes]")
    ax.grid(True, alpha=0.25)
    ax.legend(
        title="Individual seeds",
        loc="upper left",
        ncol=2,
        fontsize=9.5,
        title_fontsize=10,
        framealpha=0.95,
        columnspacing=1.4,
    )
    fig.tight_layout()
    save_figure_safely(fig, FIGURES_DIR / "fatigue_vs_makespan_by_seed.png", dpi=300)
    save_figure_safely(fig, FIGURES_DIR / "fatigue_vs_makespan_by_seed.pdf")
    plt.close(fig)

    print("  saved: figures/fatigue_vs_makespan_by_seed.png")
    print("  saved: figures/fatigue_vs_makespan_by_seed.pdf")
    print(f"  saved: csv/{output_csv.name}")


def _safe_filename(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _baseline_vs_dqn_kpi_charts(kpi: pd.DataFrame) -> None:
    """
    Create one 2x2 comparison chart for each baseline against the DQN policy.

    Baseline bars use KPI-specific colors; DQN bars always use the same
    violet color to make the adaptive policy visually recognizable.
    """
    dqn_name = "DQN Adaptive Recovery"

    if "experiment" not in kpi.columns:
        print("  [skip] baseline vs DQN charts - missing experiment column")
        return

    if dqn_name not in set(kpi["experiment"]):
        print("  [skip] baseline vs DQN charts - DQN row not found")
        return

    metrics = [
        {
            "column": "completed_jobs",
            "std_column": "completed_jobs_std",
            "title": "Completed jobs",
            "ylabel": "Jobs",
            "baseline_color": "#20bf6b",
        },
        {
            "column": "avg_fatigue",
            "std_column": "avg_fatigue_std",
            "title": "Average fatigue",
            "ylabel": "Fatigue",
            "baseline_color": "#f7b731",
        },
        {
            "column": "max_fatigue",
            "std_column": "max_fatigue_std",
            "title": "Maximum fatigue",
            "ylabel": "Fatigue",
            "baseline_color": "#fa8231",
        },
        {
            "column": "simultaneous_recovery_time_min",
            "std_column": "simultaneous_recovery_time_min_std",
            "title": "Simultaneous rest",
            "ylabel": "Minutes",
            "baseline_color": "#eb3b5a",
        },
    ]

    missing = [m["column"] for m in metrics if m["column"] not in kpi.columns]
    if missing:
        print(f"  [skip] baseline vs DQN charts - missing columns: {missing}")
        return

    dqn_row = kpi.loc[kpi["experiment"] == dqn_name].iloc[0]
    dqn_color = "#7c3aed"

    baselines = [
        exp for exp in kpi["experiment"].tolist()
        if exp != dqn_name
    ]

    for baseline_name in baselines:
        baseline_row = kpi.loc[kpi["experiment"] == baseline_name].iloc[0]

        fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
        axes = axes.flatten()

        for ax, metric in zip(axes, metrics):
            column = metric["column"]
            std_column = metric["std_column"]
            values = [
                float(baseline_row[column]),
                float(dqn_row[column]),
            ]

            yerr = None
            if std_column in kpi.columns:
                yerr = [
                    float(max(baseline_row.get(std_column, 0.0), 0.0)),
                    float(max(dqn_row.get(std_column, 0.0), 0.0)),
                ]

            bars = ax.bar(
                [baseline_name, dqn_name],
                values,
                color=[metric["baseline_color"], dqn_color],
                yerr=yerr,
                capsize=5,
                error_kw={"elinewidth": 1.2, "ecolor": "#444"},
            )

            ax.set_title(metric["title"])
            ax.set_ylabel(metric["ylabel"])
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelrotation=12)

            for bar, value in zip(bars, values):
                y_offset = max(abs(value) * 0.03, 0.02)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + y_offset,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        fig.suptitle(f"{baseline_name} vs DQN Adaptive Recovery", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        filename = f"{_safe_filename(baseline_name)}_vs_dqn_kpi_comparison.png"
        save_figure_safely(fig, FIGURES_DIR / filename, dpi=200)
        plt.close(fig)
        print(f"  saved: figures/{filename}")


def _exploration_action_distribution_by_fatigue_chart() -> None:
    """
    Show how the exploration policy samples actions across fatigue levels.

    This chart validates whether mode 2 generates informative training data:
    the DQN should observe many continue actions at moderate fatigue and a
    progressive increase in recovery actions as fatigue approaches Hy.
    """
    exploration_files = discover_seed_files("rl_decision_log_exploration")
    
    if not exploration_files:
        print("  [skip] exploration action/fatigue chart - exploration logs not found")
        return

    dfs = []
    for path in exploration_files:
        df = read_csv_if_exists(path)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        print("  [skip] exploration action/fatigue chart - empty exploration logs")
        return

    decision_df = pd.concat(dfs, ignore_index=True)

    df = normalize_decision_log(decision_df)
    df = df[df["recordType"] == "decision"].copy()

    required = {"currentFatigue", "selectedAction", "selectedActionName"}
    missing = sorted(required - set(df.columns))

    if missing:
        print(f"  [skip] exploration action/fatigue chart - missing columns: {missing}")
        return

    df = df.dropna(subset=["currentFatigue", "selectedAction"]).copy()
    df = df[df["selectedAction"].isin(ACTION_NAMES.keys())].copy()

    if df.empty:
        print("  [skip] exploration action/fatigue chart - no valid exploration actions")
        return

    bins = [0.0, HX, 0.50, 0.60, 0.70, HY, 1.01]
    labels = [
        "< Hx",
        "0.40-0.50",
        "0.50-0.60",
        "0.60-0.70",
        "0.70-Hy",
        ">= Hy",
    ]

    df["fatigueBand"] = pd.cut(
        df["currentFatigue"].clip(0.0, 1.0),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    action_order = [ACTION_NAMES[action] for action in sorted(ACTION_NAMES)]
    counts = pd.crosstab(df["fatigueBand"], df["selectedActionName"])
    counts = counts.reindex(index=labels, fill_value=0)
    counts = counts.reindex(columns=action_order, fill_value=0)

    percentages = counts.div(
        counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0) * 100.0

    counts.to_csv(RESULTS_DIR / "exploration_action_distribution_by_fatigue_counts.csv")
    percentages.to_csv(RESULTS_DIR / "exploration_action_distribution_by_fatigue_percentages.csv")

    colors = {
        "continue_working": "#2563eb",
        "short_break": "#22c55e",
        "long_break": "#dc2626",
    }

    fig, ax = plt.subplots(figsize=(9, 5.2))
    percentages.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[colors.get(col, "#6b7280") for col in percentages.columns],
    )

    totals = counts.sum(axis=1)
    for idx, total in enumerate(totals):
        if total > 0:
            ax.text(idx, 102, f"n={int(total)}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Exploration Policy: Action Distribution by Fatigue Level")
    ax.set_ylabel("Action share [%]")
    ax.set_xlabel("Fatigue level")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Exploration action", loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    save_figure_safely(fig, FIGURES_DIR / "exploration_action_distribution_by_fatigue.png", dpi=200)
    plt.close(fig)

    print("  saved: figures/exploration_action_distribution_by_fatigue.png")


def _classify_timeline_activity(row: pd.Series) -> str | None:
    activity = str(row.get("activity", row.get("task_type", ""))).strip().lower()
    state = str(row.get("operatorState", "")).strip().lower()

    if "idle" in activity or state == "idle":
        return "idle"
    if "long" in activity or state == "long_break":
        return "long recovery"
    if "short" in activity or state == "short_break":
        return "short recovery"
    if activity.startswith("job_"):
        return "task"
    if "simple" in activity:
        return "simple task"
    if "complex" in activity:
        return "complex task"

    return None


def _build_task_recovery_intervals(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df is None or event_df.empty:
        return pd.DataFrame()

    df = normalize_event_log(event_df)
    required = {"timestamp", "operator", "event", "duration"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    intervals = []

    for _, row in df.iterrows():
        event = str(row.get("event", "")).strip().lower()
        duration = float(row.get("duration", 0.0) or 0.0)
        timestamp = float(row["timestamp"])
        activity_type = _classify_timeline_activity(row)

        if activity_type is None or duration <= 0:
            continue

        if activity_type in {"simple task", "complex task", "task"}:
            if event != "end":
                continue
            start = timestamp - duration
            end = timestamp
        else:
            if event != "start":
                continue
            start = timestamp
            end = timestamp + duration

        if end <= start:
            continue

        intervals.append(
            {
                "operator": str(row["operator"]).strip(),
                "start": max(0.0, start),
                "duration": end - max(0.0, start),
                "activityType": activity_type,
            }
        )

    return pd.DataFrame(intervals)


def _task_recovery_timeline_chart(event_path: Path, chart_name: str, output_name: str) -> None:
    event_df = read_csv_if_exists(event_path)
    if event_df is None or event_df.empty:
        print(f"  [skip] task/recovery timeline - empty log: {event_path.name}")
        return

    intervals = _build_task_recovery_intervals(event_df)
    if intervals.empty:
        print(f"  [skip] task/recovery timeline - no intervals: {event_path.name}")
        return

    operators = sorted(intervals["operator"].unique())
    y_positions = {operator: idx * 10 for idx, operator in enumerate(operators)}

    colors = {
        "simple task": "#20bf6b",
        "complex task": "#3867d6",
        "task": "#3b82f6",
        "idle": "#9ca3af",
        "short recovery": "#8b5cf6",
        "long recovery": "#e11d48",
    }

    fig_height = max(3.0, 1.0 + 0.65 * len(operators))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    for (activity_type, operator), group in intervals.groupby(["activityType", "operator"]):
        y = y_positions[operator]
        bars = list(zip(group["start"].to_numpy(float), group["duration"].to_numpy(float)))

        if bars:
            ax.broken_barh(
                bars,
                (y - 3.0, 6.0),
                facecolors=colors.get(activity_type, "#6b7280"),
                edgecolors="white",
                linewidth=0.4,
                label=activity_type,
            )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.set_yticks([y_positions[operator] for operator in operators])
    ax.set_yticklabels(operators)
    ax.set_xlabel("Simulation time [minutes]")
    ax.set_title(chart_name)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        unique.values(),
        unique.keys(),
        title="Activity",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    fig.tight_layout()
    save_figure_safely(fig, FIGURES_DIR / output_name, dpi=200)
    plt.close(fig)

    intervals.to_csv(RESULTS_DIR / output_name.replace(".png", ".csv"), index=False)
    print(f"  saved: figures/{output_name}")


def _task_recovery_timeline_charts() -> None:
    sources = {
        "Exploration": "rl_event_log_exploration",
        **{
            experiment_name: bases["event_log"]
            for experiment_name, bases in EXPERIMENT_BASES.items()
        },
    }

    for label, base_name in sources.items():
        event_logs = discover_seed_files(base_name)
        if not event_logs:
            print(f"  [skip] task/recovery timeline - no log found for {label}")
            continue

        event_path = max(event_logs, key=lambda path: path.stat().st_mtime)
        safe_name = (
            label.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        _task_recovery_timeline_chart(
            event_path=event_path,
            chart_name=f"{label}: Task and Recovery Timeline",
            output_name=f"{safe_name}_task_recovery_timeline.png",
        )


def _build_executed_schedule_intervals(event_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct completed tasks and recovery intervals from one event log."""
    if event_df is None or event_df.empty:
        return pd.DataFrame()

    df = normalize_event_log(event_df)
    required = {"timestamp", "operator", "event", "duration"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    intervals: list[dict] = []
    for _, row in df.iterrows():
        event = str(row.get("event", "")).strip().lower()
        state = str(row.get("operatorState", "")).strip().lower()
        activity = str(row.get("activity", "")).strip()
        job_id = str(row.get("job_id", activity)).strip()
        timestamp = float(row["timestamp"])
        duration = float(row.get("duration", 0.0) or 0.0)
        processing_time = float(row.get("processingTime", 0.0) or 0.0)
        nominal_time = float(row.get("nominalProcessingTime", 0.0) or 0.0)

        is_recovery = (
            state in {"short_break", "long_break"}
            or activity.lower().startswith("recovery")
        )
        is_idle = state == "idle" or activity.lower().startswith("idle")

        if is_recovery and event == "start" and duration > 0.0:
            intervals.append({
                "operator": str(row["operator"]).strip(),
                "job_id": "recovery",
                "activity_type": state.replace("_", " "),
                "start": timestamp,
                "end": timestamp + duration,
                "actual_duration": duration,
                "nominal_duration": np.nan,
                "duration_delta": np.nan,
            })
        elif event == "end" and not is_recovery and (is_idle or activity):
            actual_duration = processing_time if processing_time > 0.0 else duration
            if actual_duration <= 0.0:
                continue
            start = max(0.0, timestamp - actual_duration)
            intervals.append({
                "operator": str(row["operator"]).strip(),
                "job_id": job_id,
                "activity_type": "idle" if is_idle else "task",
                "start": start,
                "end": timestamp,
                "actual_duration": actual_duration,
                "nominal_duration": nominal_time if not is_idle else np.nan,
                "duration_delta": (
                    actual_duration - nominal_time
                    if not is_idle and nominal_time > 0.0 else np.nan
                ),
            })

    return pd.DataFrame(intervals)


def _executed_schedule_variation_chart(
    event_path: Path,
    chart_name: str,
    output_name: str,
) -> None:
    """Plot the executed schedule, stochastic task-time deviations, and recovery."""
    event_df = read_csv_if_exists(event_path)
    intervals = _build_executed_schedule_intervals(event_df)
    if intervals.empty:
        print(f"  [skip] executed schedule chart - no intervals: {event_path.name}")
        return

    intervals = intervals.sort_values(["operator", "start"]).reset_index(drop=True)
    operators = sorted(intervals["operator"].unique())
    y_positions = {operator: index * 10 for index, operator in enumerate(operators)}
    colors = {
        "task": "#3b82f6",
        "idle": "#9ca3af",
        "short break": "#8b5cf6",
        "long break": "#e11d48",
    }

    fig_height = max(3.4, 1.2 + 0.9 * len(operators))
    fig, ax = plt.subplots(figsize=(16, fig_height))
    seen_labels: set[str] = set()

    for _, row in intervals.iterrows():
        activity_type = str(row["activity_type"])
        start = float(row["start"])
        duration = float(row["actual_duration"])
        y = y_positions[str(row["operator"])]
        color = colors.get(activity_type, "#64748b")
        label = "Task" if activity_type == "task" else activity_type.title()

        if activity_type == "task":
            nominal = float(row["nominal_duration"])
            delta = float(row["duration_delta"])
            base_duration = min(duration, nominal) if nominal > 0.0 else duration
            ax.broken_barh(
                [(start, base_duration)], (y - 3.0, 6.0),
                facecolors=color, edgecolors="white", linewidth=0.5,
                label=label if label not in seen_labels else None,
            )
            seen_labels.add(label)

            if delta > 1e-6:
                ax.broken_barh(
                    [(start + base_duration, delta)], (y - 3.0, 6.0),
                    facecolors="#ef4444", edgecolors="white", linewidth=0.5,
                    label="Extra stochastic time" if "Extra stochastic time" not in seen_labels else None,
                )
                seen_labels.add("Extra stochastic time")
            elif delta < -1e-6:
                ax.broken_barh(
                    [(start, duration)], (y - 3.0, 6.0),
                    facecolors="none", edgecolors="#16a34a", linewidth=1.4, hatch="//",
                    label="Saved stochastic time" if "Saved stochastic time" not in seen_labels else None,
                )
                seen_labels.add("Saved stochastic time")

            if abs(delta) >= 0.25:
                sign = "+" if delta > 0 else ""
                ax.text(start + duration / 2.0, y, f"{row['job_id']}\n{sign}{delta:.1f} m",
                        ha="center", va="center", fontsize=6.5)
        else:
            ax.broken_barh(
                [(start, duration)], (y - 3.0, 6.0),
                facecolors=color, edgecolors="white", linewidth=0.5,
                label=label if label not in seen_labels else None,
            )
            seen_labels.add(label)

    ax.set_yticks([y_positions[operator] for operator in operators])
    ax.set_yticklabels(operators)
    ax.set_xlabel("Simulation time [minutes]")
    ax.set_title(chart_name)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Executed activity")
    fig.tight_layout()
    save_figure_safely(fig, FIGURES_DIR / output_name, dpi=220)
    plt.close(fig)

    intervals.to_csv(RESULTS_DIR / output_name.replace(".png", ".csv"), index=False)
    print(f"  saved: figures/{output_name}")


def _executed_schedule_variation_charts() -> None:
    """Create one post-simulation schedule chart per available experiment log."""
    for experiment_name, bases in EXPERIMENT_BASES.items():
        event_logs = discover_seed_files(bases["event_log"])
        if not event_logs:
            continue
        event_path = max(event_logs, key=lambda path: path.stat().st_mtime)
        _executed_schedule_variation_chart(
            event_path=event_path,
            chart_name=f"{experiment_name}: Executed Schedule, Stochastic Time Variation, and Recovery",
            output_name=f"{_safe_filename(experiment_name)}_executed_schedule_variation.png",
        )


def _schedule_operator_order(intervals: pd.DataFrame) -> list[str]:
    """Keep the production-line order instead of alphabetical ordering."""
    preferred = ["OperatorA", "OperatorB", "OperatorC"]
    present = {str(value).strip() for value in intervals["operator"].dropna()}
    return [operator for operator in preferred if operator in present]


def _post_simulation_schedule_chart(
    event_path: Path,
    experiment_name: str,
    output_stem: str,
) -> None:
    """Create the two compact post-simulation schedule views requested for the thesis."""
    event_df = read_csv_if_exists(event_path)
    intervals = _build_executed_schedule_intervals(event_df)
    if intervals.empty:
        print(f"  [skip] post-simulation schedule - empty log: {event_path.name}")
        return

    intervals["operator"] = intervals["operator"].astype(str).str.strip()
    intervals = intervals[intervals["operator"].isin(
        ["OperatorA", "OperatorB", "OperatorC"]
    )].copy()
    if intervals.empty:
        return

    operators = _schedule_operator_order(intervals)
    y_positions = {operator: (len(operators) - 1 - index) * 10
                   for index, operator in enumerate(operators)}

    # View 1: blue task duration plus the stochastic deviation in red.
    task_intervals = intervals[intervals["activity_type"] == "task"].copy()
    if not task_intervals.empty:
        fig, ax = plt.subplots(figsize=(16, max(3.6, 1.6 + 1.2 * len(operators))))
        seen: set[str] = set()
        for _, row in task_intervals.sort_values(["operator", "start"]).iterrows():
            operator = str(row["operator"])
            start = float(row["start"])
            actual = float(row["actual_duration"])
            nominal = float(row["nominal_duration"])
            delta = float(row["duration_delta"])
            y = y_positions[operator]
            label = "Task" if "Task" not in seen else None

            blue_duration = min(actual, nominal) if nominal > 0.0 else actual
            ax.broken_barh(
                [(start, max(blue_duration, 0.001))],
                (y - 2.8, 5.6),
                facecolors="#2563eb",
                edgecolors="white",
                linewidth=0.45,
                label=label,
            )
            seen.add("Task")

            if abs(delta) > 1e-6 and nominal > 0.0:
                deviation_start = start + blue_duration if delta > 0 else start + actual
                deviation_width = abs(delta)
                ax.broken_barh(
                    [(deviation_start, deviation_width)],
                    (y - 2.8, 5.6),
                    facecolors="#ef4444",
                    edgecolors="#991b1b",
                    linewidth=0.7,
                    hatch="//" if delta < 0 else None,
                    label="Stochastic time deviation" if "Deviation" not in seen else None,
                )
                seen.add("Deviation")
                sign = "+" if delta > 0 else "-"
                ax.text(
                    deviation_start + deviation_width / 2.0,
                    y,
                    f"{sign}{abs(delta):.2f} min",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    fontweight="bold",
                )

        ax.set_yticks([y_positions[operator] for operator in operators])
        ax.set_yticklabels(operators)
        ax.set_xlabel("Simulation time [minutes]")
        ax.set_title(f"{experiment_name}: Tasks and Stochastic Processing-Time Variation")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        save_figure_safely(fig, FIGURES_DIR / f"{output_stem}_stochastic_tasks.png", dpi=220)
        plt.close(fig)

        task_intervals.to_csv(
            RESULTS_DIR / f"{output_stem}_stochastic_tasks.csv", index=False
        )
        print(f"  saved: figures/{output_stem}_stochastic_tasks.png")

    # View 2: task bars in blue and selected recoveries as separate overlays.
    activity_colors = {
        "task": "#2563eb",
        "short break": "#8b5cf6",
        "long break": "#f97316",
    }
    visible = intervals[intervals["activity_type"].isin(activity_colors)].copy()
    if visible.empty:
        return

    fig, ax = plt.subplots(figsize=(16, max(3.6, 1.6 + 1.2 * len(operators))))
    seen: set[str] = set()
    for _, row in visible.sort_values(["operator", "start"]).iterrows():
        operator = str(row["operator"])
        activity_type = str(row["activity_type"])
        label = activity_type.title()
        ax.broken_barh(
            [(float(row["start"]), float(row["actual_duration"]))],
            (y_positions[operator] - 2.8, 5.6),
            facecolors=activity_colors[activity_type],
            edgecolors="white",
            linewidth=0.45,
            label=label if label not in seen else None,
        )
        seen.add(label)

    ax.set_yticks([y_positions[operator] for operator in operators])
    ax.set_yticklabels(operators)
    ax.set_xlabel("Simulation time [minutes]")
    ax.set_title(f"{experiment_name}: Tasks and Selected Recoveries")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Activity")
    fig.tight_layout()
    save_figure_safely(fig, FIGURES_DIR / f"{output_stem}_tasks_and_recoveries.png", dpi=220)
    plt.close(fig)

    visible.to_csv(
        RESULTS_DIR / f"{output_stem}_tasks_and_recoveries.csv", index=False
    )
    print(f"  saved: figures/{output_stem}_tasks_and_recoveries.png")


def _post_simulation_schedule_charts() -> None:
    """Generate only the two clear schedule charts for each experiment."""
    for experiment_name, bases in EXPERIMENT_BASES.items():
        event_logs = discover_seed_files(bases["event_log"])
        if not event_logs:
            continue
        event_path = max(event_logs, key=lambda path: path.stat().st_mtime)
        _post_simulation_schedule_chart(
            event_path=event_path,
            experiment_name=experiment_name,
            output_stem=_safe_filename(experiment_name),
        )


def _compact_schedule_task_label(job_id: object) -> tuple[str, str]:
    """Return the compact task label and its parent job key."""
    value = str(job_id).strip().lower()
    match = re.search(r"job_(\d+)_task_(\d+)", value)
    if match is None:
        return str(job_id), "unknown"
    return f"J{match.group(1)} T{match.group(2)}", f"job_{match.group(1)}"


def _seed_one_policy_schedule_chart(
    event_path: Path,
    experiment_name: str,
    output_name: str,
) -> None:
    """Plot the executed schedule for seed 1 using one color per job."""
    intervals = _build_executed_schedule_intervals(read_csv_if_exists(event_path))
    if intervals.empty:
        print(f"  [skip] seed-1 schedule - empty log: {event_path.name}")
        return

    intervals["operator"] = intervals["operator"].astype(str).str.strip()
    intervals = intervals[
        intervals["operator"].isin(["OperatorA", "OperatorB", "OperatorC"])
    ].copy()
    if intervals.empty:
        return

    operators = _schedule_operator_order(intervals)
    y_positions = {
        operator: len(operators) - 1 - index
        for index, operator in enumerate(operators)
    }
    palette = [
        "#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#CC79A7", "#56B4E9", "#F0E442", "#6A3D9A",
    ]

    task_metadata = intervals["job_id"].map(_compact_schedule_task_label)
    intervals["compact_label"] = task_metadata.map(lambda item: item[0])
    intervals["job_key"] = task_metadata.map(lambda item: item[1])
    job_keys = sorted(
        {value for value in intervals.loc[
            intervals["activity_type"].eq("task"), "job_key"
        ] if value != "unknown"},
        key=lambda value: int(re.search(r"\d+", value).group()),
    )
    job_colors = {
        job_key: palette[index % len(palette)]
        for index, job_key in enumerate(job_keys)
    }
    activity_colors = {
        "idle": "#E5E7EB",
        "short break": "#7C3AED",
        "long break": "#E11D48",
    }

    figure, axis = plt.subplots(figsize=(20, 6.8))
    bar_height = 0.62

    for _, row in intervals.sort_values(["operator", "start"]).iterrows():
        operator = str(row["operator"])
        activity_type = str(row["activity_type"])
        start = float(row["start"])
        duration = float(row["actual_duration"])
        if duration <= 0.0:
            continue

        if activity_type == "task":
            color = job_colors.get(str(row["job_key"]), "#64748B")
            edge_color = "#263238"
        else:
            color = activity_colors.get(activity_type, "#CBD5E1")
            edge_color = "#9CA3AF" if activity_type == "idle" else "#4C1D95"

        y = y_positions[operator]
        axis.broken_barh(
            [(start, duration)],
            (y - bar_height / 2.0, bar_height),
            facecolors=color,
            edgecolors=edge_color,
            linewidth=0.8,
        )

        if activity_type == "task":
            text_color = (
                "#111827"
                if color in {"#E69F00", "#56B4E9", "#F0E442"}
                else "white"
            )
            axis.text(
                start + duration / 2.0,
                y,
                str(row["compact_label"]),
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="semibold",
                color=text_color,
                clip_on=True,
            )

    schedule_end = float(intervals["end"].max())
    axis.set_xlim(0.0, max(1.0, schedule_end * 1.01))
    axis.set_ylim(-0.65, len(operators) - 0.35)
    axis.set_yticks(
        [y_positions[operator] for operator in operators],
        [operator.replace("Operator", "Operator ") for operator in operators],
    )
    axis.set_xlabel("Simulation time (minutes)")
    axis.set_ylabel("Operator")
    axis.set_title(f"{experiment_name} - Executed Schedule (Seed 1)", pad=14)
    axis.grid(axis="x", alpha=0.22)
    axis.set_axisbelow(True)

    legend_handles = [
        Patch(
            facecolor=job_colors[job_key],
            edgecolor="#263238",
            label=job_key.replace("_", " ").title(),
        )
        for job_key in job_keys
    ]
    present_activities = set(intervals["activity_type"].astype(str))
    for activity_type, color in activity_colors.items():
        if activity_type in present_activities:
            legend_handles.append(Patch(
                facecolor=color,
                edgecolor="#6B7280",
                label=activity_type.title(),
            ))
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(6, len(legend_handles)),
        frameon=False,
    )

    figure.tight_layout(pad=1.4)
    save_figure_safely(
        figure,
        FIGURES_DIR / output_name,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(f"  saved: figures/{output_name}")


def _seed_one_policy_schedule_charts() -> None:
    """Generate matching seed-1 schedule charts for all four policies."""
    for experiment_name, bases in EXPERIMENT_BASES.items():
        event_path = DATA_DIR / f"{bases['event_log']}_s1.csv"
        if not event_path.exists():
            print(f"  [skip] seed-1 schedule - file not found: {event_path.name}")
            continue
        _seed_one_policy_schedule_chart(
            event_path=event_path,
            experiment_name=experiment_name,
            output_name=f"seed_1_{_safe_filename(experiment_name)}_schedule.png",
        )


def _clean_input_schedule_chart() -> None:
    """Plot only the productive tasks defined in the fixed input schedule."""
    schedule_path = SCHEDULE_DIR / "fixed_task_schedule.csv"
    schedule_df = read_csv_if_exists(schedule_path)
    if schedule_df is None or schedule_df.empty:
        return

    schedule_df.columns = schedule_df.columns.str.strip()
    required = {
        "operator", "activity_type", "job_id",
        "scheduled_start_seconds", "scheduled_end_seconds",
    }
    if not required.issubset(schedule_df.columns):
        print("  [skip] clean schedule chart - unsupported schedule format")
        return

    tasks = schedule_df[
        schedule_df["activity_type"].astype(str).str.strip().str.lower().eq("task")
    ].copy()
    if tasks.empty:
        return

    tasks = to_numeric_safe(tasks, ["scheduled_start_seconds", "scheduled_end_seconds"])
    tasks = tasks.dropna(subset=["scheduled_start_seconds", "scheduled_end_seconds"])
    tasks["start_min"] = tasks["scheduled_start_seconds"] / 60.0
    tasks["duration_min"] = (
        tasks["scheduled_end_seconds"] - tasks["scheduled_start_seconds"]
    ) / 60.0

    operators = sorted(tasks["operator"].astype(str).unique())
    y_positions = {operator: index * 10 for index, operator in enumerate(operators)}
    difficulty_colors = {"light": "#60a5fa", "medium": "#f59e0b", "heavy": "#ef4444"}

    fig, ax = plt.subplots(figsize=(15, max(3.2, 1.2 + 0.85 * len(operators))))
    used_difficulties: set[str] = set()

    for _, row in tasks.iterrows():
        operator = str(row["operator"])
        difficulty = str(row.get("difficulty", "task")).strip().lower()
        label = difficulty.title()
        ax.broken_barh(
            [(float(row["start_min"]), float(row["duration_min"]))],
            (y_positions[operator] - 3.0, 6.0),
            facecolors=difficulty_colors.get(difficulty, "#3b82f6"),
            edgecolors="white", linewidth=0.6,
            label=label if label not in used_difficulties else None,
        )
        used_difficulties.add(label)

    ax.set_yticks([y_positions[operator] for operator in operators])
    ax.set_yticklabels(operators)
    ax.set_xlabel("Planned time [minutes]")
    ax.set_title("Fixed Input Schedule: Productive Tasks Only")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Task difficulty", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save_figure_safely(fig, FIGURES_DIR / "fixed_task_schedule_clean.png", dpi=220)
    plt.close(fig)

    tasks.to_csv(RESULTS_DIR / "fixed_task_schedule_clean.csv", index=False)
    print("  saved: figures/fixed_task_schedule_clean.png")




def _ensure_fatigue_prediction_model_for_analysis() -> bool:
    """
    Ensure that the polynomial fatigue predictor trained from mode 2 exists.
    The predictor is rebuilt only when missing or older than the exploration log.
    """
    # Fatigue prediction is currently disabled.
    return False

    # event_logs = discover_seed_files("rl_event_log_exploration")
    #
    # if not event_logs:
    #     print("  [skip] fatigue prediction charts - exploration event logs not found")
    #     return False
    #
    # latest_event_log = max(event_logs, key=lambda p: p.stat().st_mtime)
    #
    # model_missing = not FATIGUE_MODEL_PATH.exists()
    # model_outdated = (
    #     FATIGUE_MODEL_PATH.exists()
    #     and max(
    #         latest_event_log.stat().st_mtime,
    #         FATIGUE_PREDICTOR_SOURCE_PATH.stat().st_mtime,
    #     )
    #     > FATIGUE_MODEL_PATH.stat().st_mtime
    # )
    #
    # if model_missing or model_outdated:
    #     reason = "missing" if model_missing else "older than exploration log"
    #     print(f"  refreshing fatigue prediction model ({reason})...")
    #     build_fatigue_prediction_models()
    #
    # return FATIGUE_MODEL_PATH.exists()

    """
    event_logs = discover_seed_files("rl_event_log_exploration")

    if not event_logs:
        print("  [skip] fatigue prediction charts - exploration event logs not found")
        return False
        
    latest_event_log = max(event_logs, key=lambda p: p.stat().st_mtime)

    model_missing = not FATIGUE_MODEL_PATH.exists()
    model_outdated = (
        FATIGUE_MODEL_PATH.exists()
        and max(
            latest_event_log.stat().st_mtime,
            FATIGUE_PREDICTOR_SOURCE_PATH.stat().st_mtime,
        )
        > FATIGUE_MODEL_PATH.stat().st_mtime
    )

    if model_missing or model_outdated:
        reason = "missing" if model_missing else "older than exploration log"
        print(f"  refreshing fatigue prediction model ({reason})...")
        # build_fatigue_prediction_models()

    return FATIGUE_MODEL_PATH.exists()
    """


def _is_recovery_activity(row: pd.Series) -> bool:
    activity = str(row.get("activity", "")).lower()
    state = str(row.get("operatorState", "")).lower()

    return "recovery" in activity or state in {
        "short_break",
        "long_break",
    }


def _build_predicted_fatigue_trace_from_event_log(
    event_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct processing-time-based predicted fatigue from an event log.

    This follows the paper-inspired extraction logic: the predictor is trained
    from mode 2 using processing-time patterns, then evaluated on each scenario
    using cumulative normalized task duration and whether a recovery happened
    before the current job.
    """

    required = {
        "timestamp",
        "operator",
        "activity",
        "event",
        "duration",
        "fatigue",
        "nominalProcessingTime",
        "operatorState",
    }
    missing = sorted(required - set(event_df.columns))
    if missing:
        raise ValueError(f"Missing event log columns for prediction trace: {missing}")

    rows = []

    for operator_name, group in event_df.groupby("operator", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)

        td_norm_cumsum = 0.0
        break_occurred_current_shift = 0
        last_shift_id = None

        for _, row in group.iterrows():
            timestamp = float(row["timestamp"])
            event = str(row["event"]).strip().lower()
            shift_id = int(timestamp // 480.0)
            is_recovery = _is_recovery_activity(row)
            duration = float(row["duration"])

            if last_shift_id is None or shift_id != last_shift_id:
                td_norm_cumsum = 0.0
                break_occurred_current_shift = 0
                last_shift_id = shift_id

            if is_recovery:
                if event == "end":
                    break_occurred_current_shift = 1
                    td_norm_cumsum = 0.0

                continue

            if event != "end":
                continue

            nominal = float(row["nominalProcessingTime"])

            if duration <= 0 or nominal <= 0:
                continue

            skill_multiplier = get_skill_multiplier(
                str(operator_name).strip(),
                str(row.get("task_type", row.get("job_type", row.get("activity", "")))),
            )
            td_norm_cumsum += duration / (nominal * skill_multiplier)

            predicted_fatigue = predict_extracted_fatigue(
                model_df=model_df,
                workstation_id=str(operator_name).strip(),
                td_norm_cumsum=td_norm_cumsum,
                had_break=break_occurred_current_shift,
            )

            rows.append(
                {
                    "timestamp": timestamp,
                    "operator": operator_name,
                    "currentFatigue": float(row["fatigue"]),
                    "predictedExtractedFatigue": predicted_fatigue,
                    "tdNormCumsumCurrentShift": td_norm_cumsum,
                    "lastJobHadBreak": break_occurred_current_shift,
                    "sourceEvent": "job_end",
                }
            )

    return pd.DataFrame(rows)


def _current_vs_predicted_fatigue_chart_for_experiment(
    experiment_name: str,
    event_path: Path,
    model_df: pd.DataFrame,
) -> None:
    if not event_path.exists():
        print(f"  [skip] fatigue prediction chart - {event_path.name} not found")
        return

    event_df = read_csv_if_exists(event_path)
    if event_df is None or event_df.empty:
        print(f"  [skip] fatigue prediction chart - empty log for {experiment_name}")
        return

    event_df = normalize_event_log(event_df)

    if not {"timestamp", "operator", "fatigue"}.issubset(event_df.columns):
        print(f"  [skip] fatigue prediction chart - missing current fatigue columns for {experiment_name}")
        return

    try:
        prediction_df = _build_predicted_fatigue_trace_from_event_log(
            event_df,
            model_df,
        )
    except ValueError as exc:
        print(f"  [skip] fatigue prediction chart - {experiment_name}: {exc}")
        return

    if prediction_df.empty:
        print(f"  [skip] fatigue prediction chart - no prediction points for {experiment_name}")
        return

    safe_name = (
        experiment_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))

    colors = {
        ("OperatorA", "current"): "#2563eb",
        ("OperatorA", "predicted"): "#60a5fa",
        ("OperatorB", "current"): "#dc2626",
        ("OperatorB", "predicted"): "#fb923c",
    }

    for operator_name in ["OperatorA", "OperatorB"]:
        predicted_group = prediction_df[
            prediction_df["operator"] == operator_name
        ].sort_values("timestamp")

        if not predicted_group.empty:
            ax.plot(
                predicted_group["timestamp"],
                predicted_group["currentFatigue"],
                label=f"{operator_name} current fatigue",
                linewidth=2.2,
                color=colors[(operator_name, "current")],
            )

            ax.plot(
                predicted_group["timestamp"],
                predicted_group["predictedExtractedFatigue"],
                label=f"{operator_name} predicted fatigue",
                linewidth=2.0,
                linestyle="--",
                color=colors[(operator_name, "predicted")],
            )

    ax.axhline(HX, color="#6b7280", linestyle=":", linewidth=1.5, label="Hx")
    ax.axhline(HY, color="#111827", linestyle=":", linewidth=1.5, label="Hy")
    ax.set_title(f"{experiment_name}: Current vs Predicted Fatigue")
    ax.set_xlabel("Simulation time [minutes]")
    ax.set_ylabel("Fatigue")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)

    fig.tight_layout()
    output_name = f"{safe_name}_current_vs_predicted_fatigue.png"
    save_figure_safely(fig, FIGURES_DIR / output_name, dpi=200)
    plt.close(fig)

    prediction_df.to_csv(
        RESULTS_DIR / f"{safe_name}_current_vs_predicted_fatigue.csv",
        index=False,
    )

    print(f"  saved: figures/{output_name}")


def _current_vs_predicted_fatigue_charts() -> None:
    # Fatigue prediction is currently disabled.
    return

    """
    if not _ensure_fatigue_prediction_model_for_analysis():
        return

    model_df = pd.read_csv(FATIGUE_MODEL_PATH)
    model_df.columns = model_df.columns.str.strip()

    for experiment_name, base_names in EXPERIMENT_BASES.items():
        event_logs = discover_seed_files(base_names["event_log"])
        if not event_logs:
            print(
                "  [skip] fatigue prediction chart - "
                f"no event log found for {experiment_name}"
            )
            continue

        # One trajectory must come from one seed. Use the most recent run
        # instead of silently plotting an outdated legacy CSV.
        event_path = max(event_logs, key=lambda path: path.stat().st_mtime)
        print(f"  prediction trace source: {event_path.name}")

        _current_vs_predicted_fatigue_chart_for_experiment(
            experiment_name=experiment_name,
            event_path=event_path,
            model_df=model_df,
        )
    """


def generate_charts(kpi: pd.DataFrame) -> None:
    """Generate and save all experiment KPI charts from the summary dataframe."""
    print("\nGenerating charts...")

    _bar_chart(kpi, "experiment", "completed_jobs",
               "Completed Jobs by Experiment", "Completed jobs",
               "completed_jobs_by_experiment.png", "#20bf6b",
               std_col="completed_jobs_std")

    _makespan_comparison_chart(kpi)
    _fatigue_vs_makespan_chart()

    _bar_chart(kpi, "experiment", "time_above_Hy_min",
               "Time Above Critical Fatigue Threshold Hy", "Minutes above Hy",
               "time_above_hy_by_experiment.png", "#eb3b5a",
               std_col="time_above_Hy_min_std")

    _bar_chart(kpi, "experiment", "max_fatigue",
               "Maximum Fatigue by Experiment", "Maximum fatigue",
               "max_fatigue_by_experiment.png", "#fa8231",
               std_col="max_fatigue_std")

    _average_fatigue_chart(kpi)
    _average_working_fatigue_chart(kpi)

    _bar_chart(kpi, "experiment", "total_recovery_time_min",
               "Total Recovery Time by Experiment", "Recovery time [min]",
               "recovery_time_by_experiment.png", "#8854d0",
               std_col="total_recovery_time_min_std")

    _bar_chart(kpi, "experiment", "simultaneous_recovery_time_min",
               "Simultaneous Rest Time by Experiment",
               "Minutes with at least two operators resting",
               "simultaneous_recovery_time_by_experiment.png", "#eb3b5a",
               std_col="simultaneous_recovery_time_min_std")

    # Disabled: Human-Centric Production Index chart.
    # _bar_chart(kpi, "experiment", "human_centric_production_index",
    #            "Human-Centric Production Index", "Index value",
    #            "human_centric_index_by_experiment.png", "#2d98da",
    #            std_col="human_centric_production_index_std",
    #            nonnegative=False)

    # Disabled: average Human Twin Health Score chart.
    # if "avg_hths" in kpi.columns:
    #     _bar_chart(kpi, "experiment", "avg_hths",
    #                "Average Human Twin Health Score by Experiment",
    #                "Average HTHS [0-100]",
    #                "average_hths_by_experiment.png", "#20bf6b",
    #                std_col="avg_hths_std")

    _grouped_action_chart(kpi)
    _baseline_vs_dqn_kpi_charts(kpi)
    _exploration_action_distribution_by_fatigue_chart()
    _post_simulation_schedule_charts()
    _seed_one_policy_schedule_charts()
    # Disabled: all current-vs-predicted fatigue charts.
    # _current_vs_predicted_fatigue_charts()


    print(f"\nAll charts saved to: {FIGURES_DIR}")

# ---------------------------------------------------------------------------
# Multi-seed file discovery & KPI aggregation
# ---------------------------------------------------------------------------

def discover_seed_files(base_name: str, suffix: str = ".csv") -> list[Path]:
    """Find seeded logs, with a single-file fallback."""
    pattern = f"{base_name}_s*{suffix}"
    seed_files = sorted(DATA_DIR.glob(pattern))
    if seed_files:
        return seed_files
    legacy = DATA_DIR / f"{base_name}{suffix}"
    return [legacy] if legacy.exists() else []


def index_seed_files(paths: list[Path], base_name: str) -> dict[str, Path]:
    indexed = {}

    for path in paths:
        suffix = path.stem.removeprefix(base_name).lstrip("_")
        indexed[suffix or "single"] = path

    return indexed


def aggregate_kpi_rows(rows: list[dict]) -> dict:
    """
    Given a list of KPI dicts (one per seed), return a single dict
    with mean values for numeric fields and a '_std' companion for each.
    """
    if not rows:
        return {}
    if len(rows) == 1:
        return rows[0]

    df = pd.DataFrame(rows)
    result = {}
    for col in df.columns:
        if col == "experiment":
            result[col] = df[col].iloc[0]
        elif pd.api.types.is_numeric_dtype(df[col]):
            result[col]              = float(df[col].mean())
            result[f"{col}_std"]    = float(df[col].std(ddof=1))
        else:
            result[col] = df[col].iloc[0]
    return result


def _canonical_schedule_job_id(value: object) -> str:
    """Remove the operator suffix used in the event log job identifier."""
    text = str(value).strip()
    parts = text.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in {"A", "B", "C", "D"}:
        return parts[0]
    return text


def _reference_schedule_time_data() -> dict:
    """Load the nominal fixed schedule used by the current simulations."""
    candidates = [
        SCHEDULE_DIR / "schedule_v1_01.csv",
        SCHEDULE_DIR / "fixed_task_schedule.csv",
    ]
    schedule_path = next((path for path in candidates if path.exists()), None)
    if schedule_path is None:
        return {
            "planned_task_time_min": np.nan,
            "planned_makespan_min": np.nan,
            "task_time_by_operator": {},
            "job_time_by_operator": {},
        }

    schedule = pd.read_csv(schedule_path)
    schedule.columns = schedule.columns.str.strip()
    if "activity_type" in schedule.columns:
        schedule = schedule[
            schedule["activity_type"].astype(str).str.lower().eq("task")
        ].copy()
    schedule = to_numeric_safe(
        schedule,
        ["processing_time_seconds", "scheduled_end_seconds"],
    )
    schedule["operator"] = schedule["operator"].astype(str).str.strip()
    schedule["job_key"] = schedule["job_id"].map(_canonical_schedule_job_id)

    task_time_by_operator = (
        schedule.groupby("operator")["processing_time_seconds"].sum()
        / 60.0
    ).to_dict()
    job_time_by_operator = {
        (str(row["operator"]), str(row["job_key"])): float(
            row["processing_time_seconds"]
        ) / 60.0
        for _, row in schedule.iterrows()
    }
    planned_makespan = (
        float(schedule["scheduled_end_seconds"].max()) / 60.0
        if "scheduled_end_seconds" in schedule.columns
        and schedule["scheduled_end_seconds"].notna().any()
        else np.nan
    )
    return {
        "planned_task_time_min": float(schedule["processing_time_seconds"].sum() / 60.0),
        "planned_makespan_min": planned_makespan,
        "task_time_by_operator": task_time_by_operator,
        "job_time_by_operator": job_time_by_operator,
    }


def _time_decomposition_for_log(
    event_df: pd.DataFrame,
    schedule_data: dict,
) -> dict:
    """Decompose task time into planned, stochastic and fatigue components."""
    df = normalize_event_log(event_df)
    activity = df.get("activity", pd.Series("", index=df.index)).astype(str).str.lower()
    state = df.get("operatorState", pd.Series("", index=df.index)).astype(str).str.lower()
    task_ends = df[
        df["event"].astype(str).str.lower().eq("end")
        & (df["processingTime"].fillna(0.0) > 0.0)
        & ~activity.str.contains("idle|recovery", na=False)
        & ~state.eq("idle")
    ].copy()

    planned_values = []
    sampled_values = []
    actual_values = []
    unmatched = 0
    for _, row in task_ends.iterrows():
        operator = str(row.get("operator", "")).strip()
        job_key = _canonical_schedule_job_id(row.get("job_id", row.get("activity", "")))
        planned = schedule_data["job_time_by_operator"].get((operator, job_key))
        if planned is None:
            unmatched += 1
            continue
        planned_values.append(float(planned))
        sampled_values.append(float(row.get("nominalProcessingTime", 0.0) or 0.0))
        actual_values.append(float(row.get("processingTime", 0.0) or 0.0))

    planned_task_time = float(np.sum(planned_values)) if planned_values else np.nan
    sampled_task_time = float(np.sum(sampled_values)) if sampled_values else np.nan
    actual_task_time = float(np.sum(actual_values)) if actual_values else np.nan
    stochastic_delta = sampled_task_time - planned_task_time
    fatigue_delta = actual_task_time - sampled_task_time
    recovery = count_recovery_events(df)
    planned_makespan = schedule_data["planned_makespan_min"]
    actual_makespan = compute_actual_makespan(df)

    return {
        "planned_task_time_min": planned_task_time,
        "sampled_nominal_task_time_min": sampled_task_time,
        "actual_task_time_min": actual_task_time,
        "stochastic_time_delta_min": stochastic_delta,
        "fatigue_time_overhead_min": fatigue_delta,
        "total_task_time_overhead_min": actual_task_time - planned_task_time,
        "natural_idle_time_min": recovery["total_idle_time"],
        "selected_recovery_time_min": recovery["total_recovery_time"],
        "total_rest_time_min": recovery["total_rest_time"],
        "planned_makespan_min": planned_makespan,
        "actual_makespan_min": actual_makespan,
        "makespan_increase_min": actual_makespan - planned_makespan,
        "completed_task_events": int(len(task_ends)),
        "unmatched_task_events": int(unmatched),
    }


def generate_makespan_time_decomposition_tables() -> None:
    """Export auditable makespan decomposition tables for the thesis."""
    schedule_data = _reference_schedule_time_data()
    per_seed_rows = []

    for experiment_name, bases in EXPERIMENT_BASES.items():
        event_base = bases["event_log"]
        event_files = index_seed_files(
            discover_seed_files(event_base), event_base
        )
        for seed, event_path in event_files.items():
            event_df = read_csv_if_exists(event_path)
            if event_df is None or event_df.empty:
                continue
            per_seed_rows.append({
                "experiment": experiment_name,
                "seed": seed,
                **_time_decomposition_for_log(event_df, schedule_data),
            })

    if not per_seed_rows:
        print("  [skip] makespan decomposition - no event logs found")
        return

    per_seed = pd.DataFrame(per_seed_rows)
    numeric_columns = [
        column for column in per_seed.columns
        if column not in {"experiment", "seed"}
    ]
    summary_rows = []
    for experiment, group in per_seed.groupby("experiment", sort=False):
        row = {"experiment": experiment, "seed_count": int(len(group))}
        for column in numeric_columns:
            row[column] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    per_seed_path = RESULTS_DIR / "makespan_time_decomposition_by_seed.csv"
    summary_path = RESULTS_DIR / "makespan_time_decomposition_summary.csv"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    reference = summary[
        summary["experiment"].eq("No Recovery Baseline")
    ]
    if not reference.empty:
        ref = reference.iloc[0]
        comparison = summary[["experiment", "seed_count"]].copy()
        comparison["makespan_reduction_min"] = ref["actual_makespan_min"] - summary["actual_makespan_min"]
        comparison["makespan_reduction_percent"] = (
            comparison["makespan_reduction_min"]
            / ref["actual_makespan_min"] * 100.0
        )
        comparison["task_time_reduction_min"] = ref["actual_task_time_min"] - summary["actual_task_time_min"]
        comparison["fatigue_overhead_reduction_min"] = ref["fatigue_time_overhead_min"] - summary["fatigue_time_overhead_min"]
        comparison["recovery_time_min"] = summary["selected_recovery_time_min"]
        comparison_path = RESULTS_DIR / "makespan_time_decomposition_vs_no_recovery.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"  saved: {comparison_path.name}")

    print(f"  saved: {per_seed_path.name}")
    print(f"  saved: {summary_path.name}")


def generate_makespan_decomposition_charts() -> None:
    """Create readable figures from the makespan decomposition summary."""
    summary_path = RESULTS_DIR / "makespan_time_decomposition_summary.csv"
    comparison_path = RESULTS_DIR / "makespan_time_decomposition_vs_no_recovery.csv"

    if not summary_path.exists():
        print("  [skip] makespan decomposition charts - summary table not found")
        return

    df = pd.read_csv(summary_path)
    if df.empty:
        return

    labels = df["experiment"].astype(str).str.replace(" Baseline", "", regex=False)
    x = np.arange(len(df))
    width = 0.20

    # Task-time decomposition is additive and is measured across all operators.
    fig, ax = plt.subplots(figsize=(13, 6.5))
    components = [
        ("planned_task_time_min", "Planned task time", "#9ecae1"),
        ("sampled_nominal_task_time_min", "Sampled nominal time", "#6baed6"),
        ("actual_task_time_min", "Effective task time", "#2171b5"),
        ("fatigue_time_overhead_min", "Fatigue overhead", "#de2d26"),
    ]
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(components))
    for offset, (column, label, color) in zip(offsets, components):
        if column in df:
            ax.bar(x + offset, df[column], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Time [minutes, summed over operators]")
    ax.set_title("Task-Time Decomposition by Experiment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure_safely(fig, FIGURES_DIR / "task_time_decomposition_by_experiment.png", dpi=220)
    plt.close(fig)
    print("  saved: figures/task_time_decomposition_by_experiment.png")

    # These quantities are shown together for interpretation, not summed:
    # operators work in parallel, so idle/recovery time is not added to makespan.
    fig, ax = plt.subplots(figsize=(13, 6.5))
    components = [
        ("actual_makespan_min", "Actual makespan", "#303030"),
        ("natural_idle_time_min", "Natural idle", "#95a5a6"),
        ("selected_recovery_time_min", "Selected recovery", "#8e44ad"),
    ]
    offsets = np.linspace(-width, width, len(components))
    for offset, (column, label, color) in zip(offsets, components):
        if column in df:
            ax.bar(x + offset, df[column], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Time [minutes]")
    ax.set_title("Makespan and Rest-Time Components")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure_safely(fig, FIGURES_DIR / "makespan_and_rest_components.png", dpi=220)
    plt.close(fig)
    print("  saved: figures/makespan_and_rest_components.png")

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        comparison = comparison[comparison["experiment"].astype(str) != "No Recovery Baseline"]
        if not comparison.empty:
            labels = comparison["experiment"].astype(str).str.replace(" Baseline", "", regex=False)
            fig, ax = plt.subplots(figsize=(11, 5.8))
            values = comparison["makespan_reduction_percent"]
            bars = ax.bar(labels, values, color=["#27ae60", "#f39c12", "#7d3cff"][:len(values)])
            ax.axhline(0.0, color="#333333", linewidth=0.8)
            ax.set_ylabel("Makespan reduction vs no-recovery baseline [%]")
            ax.set_title("Makespan Improvement Relative to No Recovery")
            ax.tick_params(axis="x", rotation=18)
            ax.grid(axis="y", alpha=0.25)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.8,
                        f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
            fig.tight_layout()
            save_figure_safely(
                fig,
                FIGURES_DIR / "makespan_reduction_vs_no_recovery.png",
                dpi=220,
            )
            plt.close(fig)
            print("  saved: figures/makespan_reduction_vs_no_recovery.png")


# ---------------------------------------------------------------------------
# Operator-level fatigue analysis
# ---------------------------------------------------------------------------

OPERATOR_ORDER = ["OperatorA", "OperatorB", "OperatorC"]
OPERATOR_COLORS = {
    "OperatorA": "#2f7d32",
    "OperatorB": "#3030d8",
    "OperatorC": "#d83b8b",
}


def _operator_fatigue_metrics(event_df: pd.DataFrame) -> list[dict]:
    """Calculate time-aware fatigue metrics for every operator in one log."""
    df = normalize_event_log(event_df)
    required = {"timestamp", "operator", "fatigue"}
    if df.empty or not required.issubset(df.columns):
        return []

    df["fatigue"] = pd.to_numeric(df["fatigue"], errors="coerce")
    df = df.dropna(subset=["fatigue"])
    rows: list[dict] = []

    for operator, group in df.groupby("operator", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        timestamps = group["timestamp"].to_numpy(dtype=float)
        fatigue = group["fatigue"].to_numpy(dtype=float)

        if len(group) >= 2:
            durations = np.maximum(np.diff(timestamps), 0.0)
            interval_fatigue = 0.5 * (fatigue[:-1] + fatigue[1:])
            total_time = float(durations.sum())
            time_weighted_average = (
                float(np.sum(interval_fatigue * durations) / total_time)
                if total_time > 0.0
                else float(fatigue.mean())
            )
            time_above_hx = estimate_time_exposure(group, HX)
            time_above_hy = estimate_time_exposure(group, HY)
        else:
            total_time = 0.0
            time_weighted_average = float(fatigue.mean())
            time_above_hx = 0.0
            time_above_hy = 0.0

        average_working_fatigue, working_time = estimate_working_fatigue(group)

        rows.append({
            "operator": str(operator),
            "average_fatigue": time_weighted_average,
            "average_working_fatigue": average_working_fatigue,
            "maximum_fatigue": float(fatigue.max()),
            "minimum_fatigue": float(fatigue.min()),
            "time_above_Hx_min": time_above_hx,
            "time_above_Hy_min": time_above_hy,
            "observations": int(len(group)),
            "observed_time_min": total_time,
            "working_time_min": working_time,
        })

    return rows


def _load_operator_fatigue_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load all available event logs and return summary plus representative traces."""
    summary_rows: list[dict] = []
    representative_traces: dict[str, pd.DataFrame] = {}

    for experiment_name, bases in EXPERIMENT_BASES.items():
        event_base = bases["event_log"]
        event_files = discover_seed_files(event_base)
        if not event_files:
            continue

        for event_path in event_files:
            event_df = read_csv_if_exists(event_path)
            if event_df is None or event_df.empty:
                continue

            normalized = normalize_event_log(event_df)
            if normalized.empty:
                continue

            metrics = _operator_fatigue_metrics(normalized)
            seed_match = re.search(r"_s([^.]*)\.csv$", event_path.name)
            seed_id = seed_match.group(1) if seed_match else "unknown"

            for metric in metrics:
                summary_rows.append({
                    "experiment": experiment_name,
                    "seed": seed_id,
                    **metric,
                })

            # Use the first available seed as a readable representative trace.
            representative_traces.setdefault(experiment_name, normalized)

    return pd.DataFrame(summary_rows), representative_traces


def _operator_fatigue_charts() -> None:
    """Export operator fatigue tables and figures for the thesis analysis."""
    raw_summary, representative_traces = _load_operator_fatigue_data()
    if raw_summary.empty:
        print("  [skip] operator fatigue charts: no event logs found")
        return

    summary = (
        raw_summary.groupby(["experiment", "operator"], as_index=False)
        .agg(
            average_fatigue=("average_fatigue", "mean"),
            average_working_fatigue=("average_working_fatigue", "mean"),
            maximum_fatigue=("maximum_fatigue", "mean"),
            minimum_fatigue=("minimum_fatigue", "mean"),
            time_above_Hx_min=("time_above_Hx_min", "mean"),
            time_above_Hy_min=("time_above_Hy_min", "mean"),
            observations=("observations", "mean"),
            observed_time_min=("observed_time_min", "mean"),
            working_time_min=("working_time_min", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    summary_path = RESULTS_DIR / "operator_fatigue_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    experiments = list(EXPERIMENT_BASES.keys())
    operators = [
        operator for operator in OPERATOR_ORDER
        if operator in set(summary["operator"])
    ]

    # Figure 1: average fatigue by operator and experiment.
    average_table = summary.pivot(
        index="operator", columns="experiment", values="average_fatigue"
    ).reindex(index=operators, columns=experiments)
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(operators))
    width = 0.78 / max(len(experiments), 1)
    for i, experiment in enumerate(experiments):
        values = average_table[experiment].to_numpy(dtype=float)
        offset = (i - (len(experiments) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width * 0.92,
            label=experiment,
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.012,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
    ax.axhline(HX, color="#777777", linestyle=":", linewidth=1.2, label="Hx")
    ax.axhline(HY, color="#111111", linestyle="--", linewidth=1.2, label="Hy")
    ax.set_xticks(x, operators)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Time-weighted average fatigue")
    ax.set_title("Average Fatigue by Operator and Experiment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure_safely(
        fig,
        FIGURES_DIR / "average_fatigue_by_operator_and_experiment.png",
        dpi=220,
    )
    plt.close(fig)

    # Figure 2: working fatigue by operator and experiment.
    working_table = summary.pivot(
        index="operator", columns="experiment", values="average_working_fatigue"
    ).reindex(index=operators, columns=experiments)
    fig, ax = plt.subplots(figsize=(15, 7))
    for i, experiment in enumerate(experiments):
        values = working_table[experiment].to_numpy(dtype=float)
        offset = (i - (len(experiments) - 1) / 2.0) * width
        bars = ax.bar(x + offset, values, width * 0.92, label=experiment)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.012,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
    ax.axhline(HY, color="#111111", linestyle="--", linewidth=1.2, label="Hy")
    ax.set_xticks(x, operators)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Task-time-weighted average fatigue")
    ax.set_title("Average Working Fatigue by Operator and Experiment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure_safely(
        fig,
        FIGURES_DIR / "average_working_fatigue_by_operator_and_experiment.png",
        dpi=220,
    )
    plt.close(fig)

    # Figure 3: maximum fatigue by operator and experiment.
    maximum_table = summary.pivot(
        index="operator", columns="experiment", values="maximum_fatigue"
    ).reindex(index=operators, columns=experiments)
    fig, ax = plt.subplots(figsize=(15, 7))
    for i, experiment in enumerate(experiments):
        values = maximum_table[experiment].to_numpy(dtype=float)
        offset = (i - (len(experiments) - 1) / 2.0) * width
        ax.bar(x + offset, values, width * 0.92, label=experiment)
    ax.axhline(HY, color="#111111", linestyle="--", linewidth=1.2, label="Hy")
    ax.set_xticks(x, operators)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean maximum fatigue")
    ax.set_title("Maximum Fatigue by Operator and Experiment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure_safely(
        fig,
        FIGURES_DIR / "maximum_fatigue_by_operator_and_experiment.png",
        dpi=220,
    )
    plt.close(fig)

    # Figure 4: one readable fatigue timeline per experiment.
    trace_experiments = [
        experiment for experiment in experiments
        if experiment in representative_traces
    ]
    n = len(trace_experiments)
    if n:
        columns = 2
        rows = int(np.ceil(n / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(16, 4.8 * rows), squeeze=False)
        axes_flat = axes.ravel()
        for ax, experiment in zip(axes_flat, trace_experiments):
            trace = representative_traces[experiment]
            for operator in operators:
                group = trace[trace["operator"] == operator].sort_values("timestamp")
                if group.empty:
                    continue
                ax.plot(
                    group["timestamp"],
                    group["fatigue"],
                    label=operator,
                    color=OPERATOR_COLORS.get(operator),
                    linewidth=1.5,
                )
            ax.axhline(HX, color="#777777", linestyle=":", linewidth=1.0)
            ax.axhline(HY, color="#111111", linestyle="--", linewidth=1.0)
            ax.set_title(f"{experiment} (representative seed)", fontsize=10)
            ax.set_xlabel("Simulation time [min]")
            ax.set_ylabel("Fatigue")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.22)
        for ax in axes_flat[n:]:
            ax.axis("off")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=len(operators), fontsize=9)
        fig.suptitle("Fatigue Evolution by Operator", fontsize=16, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save_figure_safely(
            fig,
            FIGURES_DIR / "fatigue_timeline_by_operator_and_experiment.png",
            dpi=220,
        )
        plt.close(fig)

    print(f"\nOperator fatigue summary saved to: {summary_path}")
    print("Operator fatigue charts saved to results/figures.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = []

    for experiment_name, bases in EXPERIMENT_BASES.items():
        event_base    = bases["event_log"]
        decision_base = bases["decision_log"]

        event_files = index_seed_files(
            discover_seed_files(event_base),
            event_base,
        )
        decision_files = index_seed_files(
            discover_seed_files(decision_base),
            decision_base,
        )
        seed_ids = sorted(set(event_files) | set(decision_files))
        n_seeds = len(seed_ids)

        if n_seeds == 0:
            print(f"  [skip] {experiment_name}: no log files found")
            continue

        seed_label = f"{n_seeds} seed{'s' if n_seeds > 1 else ''}"
        print(f"  {experiment_name}: {seed_label} found")

        seed_rows: list[dict] = []

        for idx, seed_id in enumerate(seed_ids):
            ev_path = event_files.get(seed_id)
            dec_path = decision_files.get(seed_id)

            event_df    = read_csv_if_exists(ev_path)  if ev_path  else None
            decision_df = read_csv_if_exists(dec_path) if dec_path else None

            event_summary    = summarize_event_log(event_df)
            decision_summary = summarize_decision_log(decision_df)

            if not event_summary and not decision_summary:
                continue

            seed_row = {
                "experiment": experiment_name,
                **event_summary,
                **decision_summary,
            }
            seed_row["human_centric_production_index"] = compute_human_centric_index(seed_row)
            seed_rows.append(seed_row)

            # Save per-seed action tables using the first seed's decision log
            if idx == 0 and decision_df is not None:
                save_action_tables(experiment_name, decision_df)

        if seed_rows:
            rows.append(aggregate_kpi_rows(seed_rows))

    if not rows:
        raise ValueError("No experiment logs found. Run AnyLogic and export logs first.")

    summary = pd.DataFrame(rows)

    preferred_columns = [
        "experiment", "completed_jobs", "throughput_jobs_per_hour",
        "makespan_min",
        "avg_queue_length", "max_queue_length", "avg_fatigue",
        "avg_working_fatigue", "total_working_time_min", "max_fatigue",
        "time_above_Hx_min", "time_above_Hy_min", "fatigue_risk_ratio",
        "total_recovery_time_min", "total_idle_time_min", "total_rest_time_min",
        "simultaneous_recovery_time_min",
        "simultaneous_recovery_recovery_time_min",
        "simultaneous_recovery_idle_time_min",
        "simultaneous_idle_idle_time_min",
        "jobs_per_recovery_minute", "jobs_per_rest_minute",
        "short_break_count",
        "long_break_count", "idle_count", "total_decisions",
        "decision_continue_count", "decision_short_break_count",
        "decision_long_break_count",
        "risky_break_attempts_when_other_on_break", "simultaneous_break_decisions",
        "avg_hths", "min_hths", "hths_below_40_decisions",
        "avg_hths_operator_a", "avg_hths_operator_b",
        "avg_hths_operator_c",
        "human_centric_production_index",
    ]
    # Include mean columns and their _std companions
    std_cols   = [f"{c}_std" for c in preferred_columns if f"{c}_std" in summary.columns]
    existing   = [c for c in preferred_columns if c in summary.columns] + std_cols
    summary    = summary[existing]

    kpi_path = RESULTS_DIR / "experiment_kpi_summary.csv"
    summary.to_csv(kpi_path, index=False)

    print("\nExperiment KPI summary (mean across seeds):")
    mean_cols = [c for c in existing if not c.endswith("_std")]
    print(summary[mean_cols].to_string(index=False))
    print(f"\nKPI saved to: {kpi_path}")

    # Export an auditable decomposition of the final makespan.
    generate_makespan_time_decomposition_tables()
    generate_makespan_decomposition_charts()
    generate_charts(summary)
    _operator_fatigue_charts()


if __name__ == "__main__":
    main()
