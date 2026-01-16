# Tutorial: Unseen Architectural Component Experiment

This tutorial guides you through the process of evaluating a new, unseen architectural component (e.g., a new branch predictor) using EarlyPerf.

## Prerequisites

- EarlyPerf environment set up.
- ChampSim repository initialized.

## Step 1: Write the New Component

First, implement your new component in the ChampSim repository.

Once implemented, you need to register it in `mappers.json` so EarlyPerf can recognize it in configurations.

Open `mappers.json` and find the relevant section (e.g., `BranchPredictor`). Add your new component string:

```json
"BranchPredictor": {
    "0": "bimodal",
    "1": "gshare",
    "2": "hashed_perceptron",
    "3": "perceptron",
    "4": "tage-sc-l",
    "5": "my_new_bp" 
}
```

## Step 2: Make a ChampSim Config

Create a new ChampSim configuration file (e.g., `my_experiment.json`) that uses your new component. You can base this on existing configurations in `ChampSim/champsim_config.json` or the sample files.

Example `my_experiment.json`:
```json
{
    "executable_name": "champsim_my_new_bp",
    "block_size": 64,
    "page_size": 4096,
    "heartbeat": 10000000,
    "num_cores": 1,
    "ooo_cpu": [ {
                    "branch_predictor": "my_new_bp" 
                    ...
                 }
               ],
    "DIB": { ... },
    "L1I": { ... },
    "L1D": { ... },
    ...
}
```

Ensure other parameters are tuned to your liking.

## Step 3: Build ChampSim Binary

Navigate to the ChampSim directory and build your new configuration.

```bash
cd ChampSim
./config.sh /path/to/my_experiment.json
make
cd ..
```

This will generate the binary specified in your config (e.g., `ChampSim/bin/champsim_my_new_bp`).

## Step 4: Run Prediction

Now run `predict.py` with the trace workload you want to test on. You can use the provided pretrained models. The script will automatically try to find the correct model for the trace.

### Important: Phase Resolution Matching

The phase resolution (instructions per simulation point) must match between your `config.json` (EarlyPerf config) and the trained model.

In `config.json`:
```json
{
    "TOTAL_INSTRUCTIONS": 700000000,
    "SIM_POINTS": 280,
    "PREVIEW_SIM_POINTS": 10,
    ...
}
```
Resolution = `TOTAL_INSTRUCTIONS` / `SIM_POINTS`.
Example: 700,000,000 / 280 = 2,500,000 instructions per point.

The `predict.py` script checks this against the model's metadata. If they don't match, you'll get a warning.

It also uses `PREVIEW_SIM_POINTS` (e.g., 10) to determine how long the preview simulation runs.

### Command

```bash
python3 predict.py \
    --trace /path/to/trace.champsimtrace.xz \
    --binary ChampSim/bin/champsim_my_new_bp \
    --config my_experiment.json \
    --models_dir models
```

**What happens:**
1. **Preview Simulation**: The script interprets the `config.json` to calculate the number of preview instructions (e.g. 10 points * 2.5M instr/point = 25M instructions) and runs the new ChampSim binary for that many instructions.
2. **Feature Extraction**: It parses the output of this short run.
3. **Prediction**: Uses the selected model to predict the full run performance (Cumulative IPC).

## Step 5: Comparison (Optional)

To validate your performance gains or prediction accuracy, you can build a "baseline" ChampSim binary using a standard component (e.g., `bimodal` or `hashed_perceptron`).

1. Create `baseline_config.json` with the standard component.
2. Build it: `./config.sh baseline_config.json && make`.
3. Run `predict.py` on this baseline to get its predicted IPC.

You can also run the full simulation to verify the actual IPC and the model's accuracy if you have the time/resources:

```bash
python3 predict.py \
    --trace /path/to/trace.champsimtrace.xz \
    --binary ChampSim/bin/champsim_my_new_bp \
    --config my_experiment.json \
    --compare
```

This will run the full simulation (defined by `TOTAL_INSTRUCTIONS`) and report the absolute error of the prediction.

> **Note**: Since the model was trained on standard components, the prediction for the baseline should be quite accurate. Your new component's prediction stability will depend on how distinct its behavior is from the trained features.
