from __future__ import annotations

import numpy as np


HX = 0.40
HY = 0.80

# Fatigue weight in R = -(beta * fatigue + delay + safety).
REWARD_BETA = 1.0

ACTION_CONTINUE = 0
ACTION_SHORT_BREAK = 1
ACTION_LONG_RECOVERY = 2

N_ACTIONS = 3

ACTION_NAMES = {
    ACTION_CONTINUE: "continue_working",
    ACTION_SHORT_BREAK: "short_break",
    ACTION_LONG_RECOVERY: "long_break",
}

# Central training preset used by train_from_transitions.py.
TRAINING_PROFILE = "three_operator_schedule"

TRAINING_PRESETS = {
    "three_operator_schedule": {
        "random_seed": 42,
        "training_steps": 30000,
        "target_update_frequency": 500,
        "report_interval": 100,
        "learning_rate": 0.00025,
        "gamma": 0.95,
        "batch_size": 128,
        "memory_multiplier": 4,
        "minimum_memory_size": 15000,
    },
}

TRAINING_CONFIG = TRAINING_PRESETS[TRAINING_PROFILE]

STATE_COLUMNS = [
    "operatorType",
    "numberLongRecoveries",
    "numberAdaptiveRecoveries",
    "timeSinceLastRecovery",
    "currentFatigue",
    "timeToReachHy",
    "lastServiceTime",
    "otherOperatorOnBreak",
    "numberOperatorsOnBreak",
    "workTime",
    "tdNormCumsumCurrentShift",
    "lastJobHadBreak",
    "fatigueA",
    "fatigueB",
    "fatigueC",
    "operatorAOnBreak",
    "operatorBOnBreak",
    "operatorCOnBreak",
    "numberOperatorsWorking",
    "globalAverageFatigue",
    "globalMaxFatigue",
    "globalMinFatigue",
    "fatigueImbalance",
    "nextTaskType",
    "nextTaskDuration",
    "predictedFatigueAfterNextTask",
    "secondTaskType",
    "predictedFatigueAfterTwoTasks",
    "timeToNextIdle",
    "nextIdleDuration",
    "remainingScheduledDuration",
    "secondTaskDuration",
    "nextTaskReady",
]


MAX_LONG_RECOVERIES = 100.0
MAX_ADAPTIVE_RECOVERIES = 300.0
MAX_TIME_SINCE_RECOVERY = 500.0
MAX_TIME_TO_REACH_HY = 500.0
MAX_SERVICE_TIME = 60.0
MAX_WORK_TIME = 240.0
MAX_TD_NORM_CUMSUM = 120.0
MAX_TIME_TO_NEXT_IDLE = 600.0
MAX_NEXT_IDLE_DURATION = 120.0
MAX_REMAINING_SCHEDULED_DURATION = 1200.0


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default


def normalize_state_values(
    operator_type,
    number_long_recoveries,
    number_adaptive_recoveries,
    time_since_last_recovery,
    current_fatigue,
    time_to_reach_hy,
    last_service_time,
    other_operator_on_break,
    number_operators_on_break,
    work_time,
    td_norm_cumsum_current_shift,
    last_job_had_break,
    fatigue_a,
    fatigue_b,
    fatigue_c,
    operator_a_on_break,
    operator_b_on_break,
    operator_c_on_break,
    number_operators_working,
    global_average_fatigue,
    global_max_fatigue,
    global_min_fatigue,
    fatigue_imbalance,
    next_task_type,
    next_task_duration,
    predicted_fatigue_after_next_task,
    second_task_type,
    predicted_fatigue_after_two_tasks,
    time_to_next_idle,
    next_idle_duration,
    remaining_scheduled_duration,
    second_task_duration,
    next_task_ready,
) -> np.ndarray:
    operator_type = np.clip(safe_float(operator_type), 0.0, 2.0)
    number_long_recoveries = np.clip(safe_float(number_long_recoveries), 0.0, MAX_LONG_RECOVERIES)
    number_adaptive_recoveries = np.clip(safe_float(number_adaptive_recoveries), 0.0, MAX_ADAPTIVE_RECOVERIES)
    time_since_last_recovery = np.clip(safe_float(time_since_last_recovery), 0.0, MAX_TIME_SINCE_RECOVERY)
    current_fatigue = np.clip(safe_float(current_fatigue), 0.0, 1.0)
    time_to_reach_hy = np.clip(safe_float(time_to_reach_hy), 0.0, MAX_TIME_TO_REACH_HY)
    last_service_time = np.clip(safe_float(last_service_time), 0.0, MAX_SERVICE_TIME)
    other_operator_on_break = np.clip(safe_float(other_operator_on_break), 0.0, 1.0)
    number_operators_on_break = np.clip(safe_float(number_operators_on_break), 0.0, 3.0)
    work_time = np.clip(safe_float(work_time), 0.0, MAX_WORK_TIME)
    td_norm_cumsum_current_shift = np.clip(
        safe_float(td_norm_cumsum_current_shift),
        0.0,
        MAX_TD_NORM_CUMSUM,
    )
    last_job_had_break = np.clip(safe_float(last_job_had_break), 0.0, 1.0)
    fatigue_a = np.clip(safe_float(fatigue_a), 0.0, 1.0)
    fatigue_b = np.clip(safe_float(fatigue_b), 0.0, 1.0)
    fatigue_c = np.clip(safe_float(fatigue_c), 0.0, 1.0)
    operator_a_on_break = np.clip(safe_float(operator_a_on_break), 0.0, 1.0)
    operator_b_on_break = np.clip(safe_float(operator_b_on_break), 0.0, 1.0)
    operator_c_on_break = np.clip(safe_float(operator_c_on_break), 0.0, 1.0)
    number_operators_working = np.clip(safe_float(number_operators_working), 0.0, 3.0)
    global_average_fatigue = np.clip(safe_float(global_average_fatigue), 0.0, 1.0)
    global_max_fatigue = np.clip(safe_float(global_max_fatigue), 0.0, 1.0)
    global_min_fatigue = np.clip(safe_float(global_min_fatigue), 0.0, 1.0)
    fatigue_imbalance = np.clip(safe_float(fatigue_imbalance), 0.0, 1.0)
    next_task_type = np.clip(safe_float(next_task_type), 0.0, 2.0)
    next_task_duration = np.clip(safe_float(next_task_duration), 0.0, MAX_SERVICE_TIME)
    predicted_fatigue_after_next_task = np.clip(
        safe_float(predicted_fatigue_after_next_task),
        0.0,
        1.0,
    )
    second_task_type = np.clip(safe_float(second_task_type), 0.0, 2.0)
    predicted_fatigue_after_two_tasks = np.clip(
        safe_float(predicted_fatigue_after_two_tasks),
        0.0,
        1.0,
    )
    time_to_next_idle = np.clip(
        safe_float(time_to_next_idle),
        0.0,
        MAX_TIME_TO_NEXT_IDLE,
    )
    next_idle_duration = np.clip(
        safe_float(next_idle_duration),
        0.0,
        MAX_NEXT_IDLE_DURATION,
    )
    remaining_scheduled_duration = np.clip(
        safe_float(remaining_scheduled_duration),
        0.0,
        MAX_REMAINING_SCHEDULED_DURATION,
    )
    second_task_duration = np.clip(
        safe_float(second_task_duration),
        0.0,
        MAX_SERVICE_TIME,
    )
    next_task_ready = np.clip(safe_float(next_task_ready), 0.0, 1.0)

    return np.array(
        [
            operator_type / 2.0,
            number_long_recoveries / MAX_LONG_RECOVERIES,
            number_adaptive_recoveries / MAX_ADAPTIVE_RECOVERIES,
            time_since_last_recovery / MAX_TIME_SINCE_RECOVERY,
            current_fatigue,
            time_to_reach_hy / MAX_TIME_TO_REACH_HY,
            last_service_time / MAX_SERVICE_TIME,
            other_operator_on_break,
            number_operators_on_break / 3.0,
            work_time / MAX_WORK_TIME,
            td_norm_cumsum_current_shift / MAX_TD_NORM_CUMSUM,
            last_job_had_break,
            fatigue_a,
            fatigue_b,
            fatigue_c,
            operator_a_on_break,
            operator_b_on_break,
            operator_c_on_break,
            number_operators_working / 3.0,
            global_average_fatigue,
            global_max_fatigue,
            global_min_fatigue,
            fatigue_imbalance,
            next_task_type / 2.0,
            next_task_duration / MAX_SERVICE_TIME,
            predicted_fatigue_after_next_task,
            second_task_type / 2.0,
            predicted_fatigue_after_two_tasks,
            time_to_next_idle / MAX_TIME_TO_NEXT_IDLE,
            next_idle_duration / MAX_NEXT_IDLE_DURATION,
            remaining_scheduled_duration / MAX_REMAINING_SCHEDULED_DURATION,
            second_task_duration / MAX_SERVICE_TIME,
            next_task_ready,
        ],
        dtype=np.float32,
    )


def classify_region(fatigue: float) -> str:
    fatigue = safe_float(fatigue)

    if fatigue < HX:
        return "A"
    if fatigue < HY:
        return "B"
    return "C"
