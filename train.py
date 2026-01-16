import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import ExtraTreesRegressor
import warnings
from tqdm import tqdm
from utils.feature_utils import generator, target_final_nonsplit

warnings.filterwarnings("ignore")

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

def get_data_dir(config):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, config.get("DATA_DIR", "data"))
    return data_dir

def list_pickles(data_dir, sim_run):
    path = os.path.join(data_dir, sim_run)
    if not os.path.exists(path):
        path = data_dir
    
    if os.path.exists(path):
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pickle") or f.endswith(".pkl")]
    return []

def train_model(config, args):
    data_dir = get_data_dir(config)
    sim_run = config["SIMULATION_RUN"]
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.get("MODELS_DIR", "models"))
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    pickles = list_pickles(data_dir, sim_run)
    if not pickles:
        tqdm.write(f"No data found in {data_dir}/{sim_run}")
        return

    tqdm.write(f"Found {len(pickles)} benchmark data files.")

    duration = config.get("PREVIEW_SIM_POINTS", 10)
    
    for pkl_path in tqdm(pickles, desc="Training models"):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            benchmark_name = os.path.basename(pkl_path).replace(".pickle", "").replace(".pkl", "")
            
            from utils.feature_utils import get_clean_trace_name
            
            # Clean benchmark name
            if "champsimtrace" in benchmark_name:
                 cleaned = get_clean_trace_name(benchmark_name)
                 if cleaned != benchmark_name:
                     benchmark_name = cleaned
                 else:
                     benchmark_name = benchmark_name.split(".champsimtrace")[0]
            
            if args.limit_data:
                limit = config.get("NUM_CONFIGS", 0)
                if limit > 0 and len(data) > limit:
                   tqdm.write(f"Limiting {benchmark_name} data from {len(data)} to {limit} samples.")
                   data = {k: data[k] for k in list(data)[:limit]}
            
            for k in data:
                # Safety for miss ratios
                if "dcache_read_miss_ratio" not in data[k] and "dcache_read_misses" in data[k]:
                     pass
            
            # Generate features
            keys, y_value_names_dict = generator(data, n=duration, train_after_warmup=False, warmup_period=80)
            
            target_key = "cumulative_ipc"
            if target_key not in keys:
                 continue
                 
            X_list, y_list_series = keys[target_key]
            feature_names = y_value_names_dict[target_key]
            
            # Get final values
            y_final = [series[-1] for series in y_list_series]
            
            X = pd.DataFrame(X_list, columns=feature_names)
            y = np.array(y_final)
            
            # Train
            model = ExtraTreesRegressor(n_jobs=-1, random_state=42)
            model.fit(X, y)
            
            # Calculate resolution for metadata
            total_instr = config.get("TOTAL_INSTRUCTIONS", 700000000)
            sim_points = config.get("SIM_POINTS", 280)
            phase_resolution = int(total_instr / sim_points)

            # Save model with metadata
            save_packet = {
                "model": model,
                "feature_names": feature_names,
                "phase_resolution": phase_resolution,
                "config_summary": {
                    "total_instructions": total_instr,
                    "sim_points": sim_points,
                    "preview_sim_points": duration
                }
            }
            
            # Model name includes benchmark name, resolution of phases, duration of preview in simulation points, and 'et' for Extra Trees
            model_filename = f"{benchmark_name}_res{phase_resolution}_pts{duration}_et.pkl"
            save_path = os.path.join(models_dir, model_filename)

            with open(save_path, "wb") as f:
                pickle.dump(save_packet, f)
            tqdm.write(f"Saved model to {save_path}")
            
        except Exception as e:
            tqdm.write(f"Failed to process {pkl_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EarlyPerf models.")
    parser.add_argument("--limit-data", action="store_true", help="Limit the number of training samples to NUM_CONFIGS from config.json")
    args_cli = parser.parse_args()
    
    config = load_config()
    train_model(config, args_cli)
