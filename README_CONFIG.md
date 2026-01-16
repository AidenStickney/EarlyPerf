# EarlyPerf Configuration Guide

This document explains the parameters used in `config.json` for controlling the simulation, training, and prediction pipeline.

## Simulation Parameters

*   **`TOTAL_INSTRUCTIONS`** (int): 
    *   The total number of instructions to be simulated. 
    *   This value is used as both the limit for simulation (`--simulation-instructions`) and the estimate for phase calculation (`-e`).
    *   **Example**: `700000000` (700 Million)

*   **`SIM_POINTS`** (int):
    *   The number of total phases to generate during each simulation of data collection.
    *   This defines the "resolution"/"sampling rate" of the time-series data used by the ML model.
    *   The ML model expects an input vector of this size (e.g., if set to 10, the model expects 10 sequential measurements). This is the preview window for early performance prediction.
    *   **Example**: `280`

*   **`PREVIEW_SIM_POINTS`** (int):
    *   The number of simulation points to use during the prediction preview run.
    *   This allows for a shorter simulation during prediction to quickly gather initial data for the model.
    *   **Example**: `10` (3.57% of full simulation if `SIM_POINTS` is 280)

### Derived Logic
*   **Phase Length**: The simulation effectively splits `TOTAL_INSTRUCTIONS` into `SIM_POINTS` phases.
    *   Example: `700,000,000 / 280 = 2,500,000` instructions per phase/sample.
## Trace Management
*   **`TRACE_EXTENSIONS`** (list of strings): 
    *   A list of file extensions that should be stripped from trace filenames to determine the benchmark name.
    *   Order matters partially, but the system auto-sorts by length (longest first) to prevent partial matching errors.
    *   **Example**: `[".champsimtrace.xz", ".trace.xz"]`
## Paths & Environment
*   **`CHAMPSIM_DIR`**: Path to the ChampSim root directory.
*   **`TRACE_DIR`**: Directory containing trace files.
*   **`TOP_TRACE_FOLDER`**: Contains `TRACE_DIR` and is used for nested folder structures.
*   **`DB_FILE`**: Path to the SQLite database used for configuration management.
*   **`SIMULATION_RUN`**: A label for the current batch of benchmarks/traces (used for organization of experiments).
*   **`CACTI_DIR`**: Path to the CACTI directory (used for accurate cache latencies).
*   **`DATA_DIR`**: Directory where parsed data and datasets are stored (default: `data`).
*   **`OUTPUT_DIR`**: Directory where raw simulation outputs (JSON/logs) are stored (default: `output`).
*   **`MODELS_DIR`**: Directory where trained ML models are saved (default: `models`).

## Job Submission & Execution
*   **`TRACES_TO_RUN`** (int): Maximum number of traces to process in a single run.
*   **`TRACES_OFFSET`** (int): Offset/startIndex for the list of traces (useful for splitting work across multiple runs).
*   **`WALL_TIME`** (int): Maximum execution time (in hours) requested for SLURM jobs.
*   **`MEM_SIZE_PER_JOB_MB`** (int): Memory requested for SLURM job in MB.
*   **`ACCOUNT`** (string): SLURM account string for job allocation.
*   **`NUM_CONFIGS`** (int): Number of random configurations to generate per benchmark trace.

## Model Features
*   **`SMA_WINDOW_SIZE`** (int): Window size used for Simple Moving Average smoothing of IPC curves in `parser.py`.
