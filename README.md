# Fatigue RL Agent

This project combines Python-generated job schedules, an AnyLogic simulation of fatigue in three operators, and a Deep Q-Network (DQN) that chooses whether to continue working, take a short recovery, or take a long recovery. Task order and operator assignments are fixed by the schedule; actual task durations and recovery decisions are determined during simulation.

## Project folders

| Path | Contents |
| --- | --- |
| `src/` | Schedule generation, dataset preparation, DQN training and inference, baselines, and results analysis. |
| `schedule/exploration/` | Merged schedule CSV used for exploration in AnyLogic mode 2. |
| `schedule/simulation/` | Evaluation schedule and offline GA recovery plan. |
| `data/anylogic_logs/` | Event and decision logs exported by AnyLogic for each seed. |
| `data/agent_log/` | Transition dataset used to train the DQN. |
| `models/` | Trained DQN checkpoint (`dqn_fatigue_agent.pth`). |
| `results/csv/` | KPI tables, training history, and batch-run logs. |
| `results/figures/` | Simulation and performance figures. |
| `results/schedule/` | Figures of the generated schedules. |
| `notes/` | Quick commands, notes, backups, and legacy material. |

The active AnyLogic model is stored outside this folder at `C:\Users\pietr\Models\rl_agent_multi_op\rl_agent_multi_op.alp`.

## Python scripts

| File in `src/` | Purpose |
| --- | --- |
| `scheduling_generator.py` | Uses PyJobShop to create random schedules, AnyLogic-compatible CSV files, and Gantt charts; merges exploration schedules. |
| `run_exploration_batch.py` | Runs AnyLogic mode 2 for the requested seeds and collects exploration logs. |
| `anylogic_batch_runner.py` | Handles AnyLogic startup, timeouts, output checks, retries, and process cleanup. |
| `build_training_dataset.py` | Validates exploration logs, calculates the reward for each observed action, and builds transitions. |
| `rl_config.py` | Defines actions, fatigue thresholds, the 33 state features, normalization, and training settings. |
| `dqn_agent.py` | Implements the Q-network, replay memory, and DQN updates. |
| `train_from_transitions.py` | Trains the DQN and saves the checkpoint, loss history, and policy check. |
| `dqn_recovery_server.py` | Loads the trained model and sends actions to AnyLogic over TCP port 5005. |
| `generate_static_ga_baseline.py` | Optimizes a fixed recovery plan before simulation using a genetic algorithm. |
| `run_baseline_batch.py` | Prepares the GA plan and runs the three baselines for the selected seeds. |
| `analyze_experiments.py` | Produces KPI tables and figures from simulation logs. |

## Typical workflow

Run these commands from the project root in a VS Code PowerShell terminal. Change `--count`, `--prefix`, and `--seeds` as needed. `--overwrite` replaces existing files with the same names.

1. Generate schedules for exploration. `--count` is the number of schedules; `--merged-name` names the CSV that AnyLogic reads in mode 2.

   ```powershell
   py src/scheduling_generator.py --count 50 --seed 42 --prefix schedule_training --merged-name schedule_training_merged --output-set exploration --overwrite
   ```

2. Run exploration in AnyLogic mode 2 for the selected seeds. The batch script starts AnyLogic itself, so close any manually opened AnyLogic instance first.

   ```powershell
   py src/run_exploration_batch.py --seeds 1 2 3 4 5 --overwrite
   ```

3. Build the training dataset and train the agent.

   ```powershell
   py src/build_training_dataset.py
   py src/train_from_transitions.py
   ```

4. Run the three baselines in AnyLogic mode 1. The batch script also starts AnyLogic automatically.

   ```powershell
   py src/run_baseline_batch.py --seeds 1 2 3 4 5 --overwrite
   ```

5. Run the DQN-agent simulation manually as described below. After the simulations finish, generate the result tables and figures:

   ```powershell
   py src/analyze_experiments.py
   ```

## Running a simulation with the DQN agent

1. Open the project folder in VS Code. In its terminal, start the inference server:

   ```powershell
   py src/dqn_recovery_server.py
   ```

2. Wait for the terminal message `=== DQN SERVER LISTENING on 127.0.0.1:5005 ===`. Keep this terminal and the server running. If the model cannot be loaded, resolve the reported error before starting the experiment.
3. Open the AnyLogic model. You may launch AnyLogic while the server is starting, but do not run the agent experiment until the listening message appears.
4. In AnyLogic, set `experiment mode = 3` and run the **Simulation** experiment. AnyLogic sends each DQN decision state to the Python server and receives the selected action.
5. For a faster run, click **Virtual** in the AnyLogic simulation speed controls.

Python dependencies are listed in `requirements.txt`. A more detailed technical description is available in `THESIS_PROJECT_HANDOFF.md`.
