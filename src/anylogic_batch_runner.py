"""Launch isolated AnyLogic runs and verify their completion markers."""

from __future__ import annotations

import csv
import os
import subprocess
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(r"C:\Users\pietr\Models\rl_agent_multi_op\rl_agent_multi_op.alp")
ANYLOGIC_EXE = Path(
    r"C:\Program Files\AnyLogic 8.9 Personal Learning Edition\AnyLogic.exe"
)
EXPERIMENT_NAME = "Simulation"
DATA_DIR = BASE_DIR / "data" / "anylogic_logs"
RUN_LOG_DIR = BASE_DIR / "results" / "csv"
COMPILE_MARKER = 'Compile "rl_agent_multi_op"'


def process_ids(image_name: str) -> set[int]:
    result = subprocess.run(
        [
            "tasklist",
            "/FI",
            f"IMAGENAME eq {image_name}",
            "/FO",
            "CSV",
            "/NH",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    found: set[int] = set()

    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 2 or row[0].lower() != image_name.lower():
            continue
        try:
            found.add(int(row[1]))
        except ValueError:
            continue

    return found


def anylogic_process_ids() -> set[int]:
    return process_ids("AnyLogic.exe")


def java_process_ids() -> set[int]:
    return process_ids("java.exe") | process_ids("javaw.exe")


def terminate_process_tree(process_id: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(process_id), "/T", "/F"],
        capture_output=True,
        text=True,
        check=False,
    )


def cleanup_run_processes(
    launcher: subprocess.Popen | None,
    preserved_anylogic: set[int],
    preserved_java: set[int],
) -> None:
    """Stop only processes created by the run, preserving earlier ones."""
    if launcher is not None and launcher.poll() is None:
        terminate_process_tree(launcher.pid)

    for process_id in sorted(anylogic_process_ids() - preserved_anylogic):
        terminate_process_tree(process_id)

    for process_id in sorted(java_process_ids() - preserved_java):
        terminate_process_tree(process_id)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        new_anylogic = anylogic_process_ids() - preserved_anylogic
        new_java = java_process_ids() - preserved_java
        if not new_anylogic and not new_java:
            break
        time.sleep(0.5)

    time.sleep(2.0)


def compilation_started(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        return COMPILE_MARKER in log_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return False


def log_tail(log_path: Path, line_count: int = 12) -> str:
    if not log_path.exists():
        return "The AnyLogic log has not been created yet."
    try:
        lines = log_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError as error:
        return f"Unable to read the AnyLogic log: {error}"
    return "\n".join(lines[-line_count:]) or "The AnyLogic log is empty."


def remove_file_with_retry(
    path: Path,
    *,
    attempts: int = 20,
    delay: float = 0.25,
    required: bool = True,
) -> None:
    """Remove a file after its Windows writer has released the handle."""
    last_error: PermissionError | None = None

    for _ in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(delay)

    if required and last_error is not None:
        raise last_error


def validate_environment() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"AnyLogic model not found: {MODEL_PATH}")
    if not ANYLOGIC_EXE.exists():
        raise FileNotFoundError(f"AnyLogic executable not found: {ANYLOGIC_EXE}")
    if anylogic_process_ids():
        raise RuntimeError(
            "Close AnyLogic before starting the batch. The runner opens and "
            "closes one isolated simulation at a time."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_simulation(
    *,
    mode: int,
    baseline: int,
    seed: int,
    output_paths: tuple[Path, Path],
    label: str,
    timeout: float,
    startup_timeout: float,
    retries: int,
    overwrite: bool,
) -> None:
    """Run one seed with bounded startup, completion, and retry times."""
    if not overwrite and all(
        path.exists() and path.stat().st_size > 0 for path in output_paths
    ):
        print(f"[skip] {label}, seed={seed}: CSV files already exist", flush=True)
        return

    environment = os.environ.copy()
    environment.update(
        {
            "FATIGUE_RL_AUTOMATED": "1",
            "FATIGUE_RL_MODE": str(mode),
            "FATIGUE_RL_BASELINE": str(baseline),
            "FATIGUE_RL_SEED": str(seed),
        }
    )

    log_path = RUN_LOG_DIR / f"anylogic_{label}_s{seed}.log"
    completion_marker = output_paths[0].with_suffix(".done")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        for path in (*output_paths, completion_marker):
            remove_file_with_retry(path)

        print(
            f"[run] {label}, seed={seed}, attempt={attempt}/{retries}",
            flush=True,
        )

        preserved_anylogic = anylogic_process_ids()
        preserved_java = java_process_ids()
        launcher: subprocess.Popen | None = None
        run_completed = False
        log_mode = "w" if attempt == 1 else "a"

        try:
            with log_path.open(log_mode, encoding="utf-8") as log_file:
                log_file.write(
                    f"\n=== {label}, seed={seed}, attempt={attempt}/{retries} ===\n"
                )
                log_file.flush()

                launcher = subprocess.Popen(
                    [
                        str(ANYLOGIC_EXE),
                        "-r",
                        str(MODEL_PATH),
                        EXPERIMENT_NAME,
                    ],
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(
                        subprocess,
                        "CREATE_NEW_PROCESS_GROUP",
                        0,
                    ),
                )

                started_at = time.monotonic()
                startup_deadline = started_at + startup_timeout
                run_deadline = started_at + timeout

                while time.monotonic() < run_deadline:
                    outputs_ready = all(
                        path.exists() and path.stat().st_size > 0
                        for path in output_paths
                    )
                    if completion_marker.exists() and outputs_ready:
                        run_completed = True
                        print(f"[done] {label}, seed={seed}", flush=True)
                        return

                    if (
                        time.monotonic() >= startup_deadline
                        and not compilation_started(log_path)
                    ):
                        raise TimeoutError(
                            f"AnyLogic did not compile within {startup_timeout:.0f} s.\n"
                            f"Last log lines:\n{log_tail(log_path)}"
                        )

                    if launcher.poll() is not None:
                        raise RuntimeError(
                            "AnyLogic closed before exporting both CSV files."
                        )

                    time.sleep(1.0)

                raise TimeoutError(
                    f"Simulation exceeded the {timeout:.0f} s timeout."
                )
        except (RuntimeError, TimeoutError) as error:
            last_error = error
            print(f"[retry] {label}, seed={seed}: {error}", flush=True)
        finally:
            cleanup_run_processes(
                launcher,
                preserved_anylogic,
                preserved_java,
            )
            if run_completed:
                remove_file_with_retry(
                    completion_marker,
                    required=False,
                )

        if attempt < retries:
            print("        restarting the same run", flush=True)

    raise RuntimeError(
        f"{label}, seed={seed} failed after {retries} attempts. "
        f"See {log_path}."
    ) from last_error


def normalized_seeds(seeds: list[int]) -> list[int]:
    if not seeds or any(seed < 1 for seed in seeds):
        raise ValueError("Seeds must be positive integers.")
    return list(dict.fromkeys(seeds))
