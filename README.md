# EarlyPerf: Early Performance Prediction for ChampSim

This repository contains tools for predicting ChampSim simulation performance using early simulation data.

## Project Structure

*   `random_runner.py`: Script to generate random ChampSim configurations and SLURM job scripts.
*   `parser.py`: Main script to parse ChampSim output JSON and logs into a single pickle dataset (per benchmark).
*   `train.py`: Script to train the ExtraTrees regression model on the parsed dataset and save the trained model.
*   `predict.py`: Script to generate predictions using trained models.
*   `tests/evaluate_model.py`: Script to evaluate model performance with various holdout strategies.
*   `mappers.json`: Defines mappings from database indices to ChampSim configuration strings.
*   `config.json`: Top-level configuration for simulation parameters and paths.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone --recursive https://github.com/stickneyaiden/EarlyPerf.git
    # If already cloned:
    # git submodule update --init --recursive
    ```
2.  **Build Submodules**:
    *   **ChampSim**:
        ```bash
        cd ChampSim
        git submodule update --init  # Setup vcpkg submodule
        ./vcpkg/bootstrap-vcpkg.sh
        ./vcpkg/vcpkg install
        # To verify build (optional):
        # ./config.sh champsim_config.json
        # make
        cd ..
        ```
    *   **CACTI**:
        ```bash
        cd cacti
        make
        cd ..
        ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment**: Ensure you have Python 3.8+ loaded.
4.  **Configuration**:
    *   Copy `config.json.template` to `config.json` and edit it with your specific paths and account information.
        ```bash
        cp config.json.template config.json
        # Edit config.json
        ```
    *   Copy `scripts/slurm_template.txt.template` to `scripts/slurm_template.txt` and customize it for your SLURM environment.
        ```bash
        cp scripts/slurm_template.txt.template scripts/slurm_template.txt
        # Edit scripts/slurm_template.txt
        ```

## Usage
Note: By using the provided pretrained models in the `models/` directory, you can skip directly to the Prediction step.

### 1. Data Generation
To generate new training data by running simulations:
1.  Configure the simulation parameters in `config.json`.
2.  Modify the SLURM job template in `scripts/slurm_template.txt` to match your environment.
3.  Run the generation script:
    ```bash
    python3 scripts/random_runner.py
    ```
This will compile random configurations and create bash scripts to be submitted to SLURM. Simulation outputs (json and logs) will be stored in `output/<SIMULATION_RUN>/`.

### 2. Run Simulations
Submit the generated SLURM scripts located in `batch/<SIMULATION_RUN>/`.

### 3. Data Parsing
Parse raw simulation outputs (JSON/Logs) into a structured dataset (pickle).
```bash
python3 parser.py
```
This looks for data in `output/<SIMULATION_RUN>/json`/`output/<SIMULATION_RUN>/logs` and saves to `data/aggregated_output/`.

### 4. Training
Train performance prediction models.
```bash
python3 train.py
# Optionally limit training data to NUM_CONFIGS defined in config.json:
# python3 train.py --limit-data
```
The script uses the parsed data for the specified simulation run (`SIMULATION_RUN`) with the defined `PREVIEW_SIM_POINTS`.
Models are saved to the `models/` directory.

### 5. Prediction
Run predictions using `predict.py` (you must have trained models first).
```bash
python3 predict.py --trace <path_to_trace> --binary <path_to_champsim_binary> --config <path_to_champsim_config>
```
Options:
*   `--models_dir`: Directory containing trained models (default: `models/`).
*   `--model_name`: Specific model file to use (optional).
*   `--compare`: If set, runs a full simulation to compare actual IPC.
*   `--compare_instructions`: Number of instructions for the full comparison run (default: 700 Million).

The script uses `PREVIEW_SIM_POINTS` and `TOTAL_INSTRUCTIONS` from `config.json` to determine the number of instructions for the preview run.

### 6. Evaluation & Testing
Use the tools in `tests/` to evaluate model performance. These tools do not use pretrained models and instead rely on the parsed dataset.

*   **General Evaluation** (hold out a random configuration):
    ```bash
    python3 tests/evaluate_model.py --pickle <path_to_pickle> --mode random
    ```
*   **Index Holdout** (e.g., hold out configuration at index 5):
    ```bash
    python3 tests/evaluate_model.py --pickle <path_to_pickle> --mode index --index 5
    ```
*   **Group Holdout** (e.g., hold out all `tage_sc_l` branch predictors):
    ```bash
    python3 tests/evaluate_model.py \
      --pickle <path_to_pickle> \
      --mode group \
      --feature BranchPredictor \
      --value tage_sc_l
    ```
*   **K-Fold Cross Validation** (e.g., 3-fold):
    ```bash
    python3 tests/evaluate_model.py --pickle <path_to_pickle> --mode kfold --k 3
    ```
*   **Leave-One-Out Cross Validation**:
    ```bash
    python3 tests/evaluate_model.py --pickle <path_to_pickle> --mode loo
    ```

Options:
*   `--pickle`: Path to the pickle file containing parsed data.
*   `--mode`: Holdout mode: `random`, `index`, `group`, `kfold`, or `loo`.
*   `--index`: Index of configuration to hold out (for mode=`index`).
*   `--feature`: Feature name to hold out (for mode=`group`).
*   `--value`: Feature value to hold out (for mode=`group`).
*   `--mappers`: Path to `mappers.json` (required for mode=`group`).
*   `--duration`: Duration of simulation samples/phases to use (preview).
*   `--k`: Number of folds for k-fold cross validation (for mode=`kfold`).

This tool does not use `config.json` and requires the user to specify the pickle file and other parameters directly.


## External Dependencies
*   **ChampSim**: This tool expects ChampSim simulation outputs. The `ChampSim` directory is currently excluded from git via `.gitignore`.
*   **CACTI**: Used for power estimation (referenced in configuration).
