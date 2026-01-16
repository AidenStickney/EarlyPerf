import argparse
import os
import json
import subprocess
import pickle
import numpy as np
import pandas as pd
from utils.feature_utils import generator
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.json_reader import parse_output
except ImportError:
    print("Error: Could not import utils.json_reader")
    sys.exit(1)

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def load_mappers(mapper_path="mappers.json"):
    with open(mapper_path, "r") as f:
        return json.load(f)

def encode_config_str(config_path, mappers):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    params_to_encode = [
        "Frequency", "iFetchBufferSize", "DecodeBufferSize", "DispatchBufferSize", "RegisterFileSize", "ROBSize",
        "LQSize", "SQSize", "FetchWidth", "DecodeWidth", "DispatchWidth", "ExecuteWidth",
        "LQWidth", "SQWidth", "RetireWidth", "MispredictPenalty", "SchedulerSize", "BranchPredictor",
        "DIBWindowSize", "DIBSets", "DIBWays", "L1ISets", "L1IWays", "L1IRQSize", "L1IWQSize",
        "L1IPQSize", "L1IMSHRSize", "L1IPrefetcher", "L1DSets", "L1DWays", "L1DRQSize",
        "L1DWQSize", "L1DPQSize", "L1DMSHRSize", "L1DPrefetcher", "L2Sets", "L2Ways",
        "L2Prefetcher", "LLCSets", "LLCWays", "LLCPrefetcher",
        "PhysicalMemoryChannels", "PhysicalMemoryRanks"
    ]
    
    encoded_values = []
    
    # General Params
    for param in params_to_encode:
        found = False
        for section, fields in mappers["json_mappings"].items():
            if param in fields:
                field_name = fields[param]
                section_data = config.get(section, None)
                
                val = None
                if isinstance(section_data, list):
                     val = section_data[0].get(field_name) if section_data else None
                elif isinstance(section_data, dict):
                     val = section_data.get(field_name) if section_data else None
                
                if val is not None:
                    encoding = None
                    mapping_dict = mappers["mappers"].get(param, {})
                    for k, v in mapping_dict.items():
                        if str(v) == str(val):
                            encoding = k
                            break
                    
                    if encoding is not None:
                        encoded_values.append(encoding)
                        found = True
                break
        
        if not found: # Fallback
            encoded_values.append("0")

    # BTB
    btb_keys = [
        ("BTBReturnStackMaxSize", "return_stack_max_size", "48"), # Default "0" -> 48
         ("BTBReturnStackNumCallSizeTrackers", "return_stack_num_call_size_trackers", "768"),
         ("BTBIndirectPredictorSize", "indirect_predictor_size", "3072"),
         ("BTBDirectPredictorSets", "direct_predictor_sets", "512")
    ]
    
    btb_direct_sets = 512 # Fallback
    
    for param_name, config_key, default_val in btb_keys:
        val = config.get("BTB", {}).get(config_key)
        if val is None: val = config.get(config_key)
        
        encoding = "0"
        if val is not None:
             mapping_dict = mappers["mappers"].get(param_name, {})
             for k, v in mapping_dict.items():
                 if str(v) == str(val):
                     encoding = k
                     break
        
        encoded_values.append(encoding)
        
        if param_name == "BTBDirectPredictorSets":
             btb_direct_sets = encoding

    # Cache latencies
    for cache in ["L1I", "L1D", "L2C", "LLC"]:
        lat = config.get(cache, {}).get("latency", 1)
        encoded_values.append(str(lat))
        
    # Physical Memory
    pm_keys = [
        "data_rate", "bankgroups", "banks", "bank_rows", "bank_columns", "channel_width",
        "wq_size", "rq_size", "tCAS", "tRCD", "tRP", "tRAS", "refresh_period", "refreshes_per_period"
    ]
    pm_data = config.get("physical_memory", {})
    for k in pm_keys:
        val = pm_data.get(k, 0)
        encoded_values.append(str(val))
        
    return "_".join(encoded_values), btb_direct_sets

def run_prediction(args):
    model_name = args.model_name
    from utils.feature_utils import get_clean_trace_name

    config = load_config()
    
    # Check for resolution consistency if metadata exists
    curr_total = config.get("TOTAL_INSTRUCTIONS", 700000000)
    curr_points = config.get("SIM_POINTS", 280)
    num_sim_points = config.get("PREVIEW_SIM_POINTS", 10)
    curr_resolution = int(curr_total / curr_points)

    if not model_name:
         base = get_clean_trace_name(args.trace)
         # Try to find specific model matching current config first
         model_name = f"{base}_res{curr_resolution}_pts{num_sim_points}_et.pkl"
         
         # Fallback to old style just in case
         if not os.path.exists(os.path.join(args.models_dir, model_name)):
             legacy_name = f"{base}_et.pkl"
             if os.path.exists(os.path.join(args.models_dir, legacy_name)):
                 model_name = legacy_name
    
    model_path = os.path.join(args.models_dir, model_name)
    if not os.path.exists(model_path):
        # Fallback of fallback is to find any model matching the base name
        files = os.listdir(args.models_dir)
        matches = [f for f in files if base in f and f.endswith(".pkl")]
        if matches:
            model_path = os.path.join(args.models_dir, matches[0])
            print(f"Auto-selected model: {matches[0]}")
        else:
            print(f"Error: Model {model_path} not found and no matches for {base}.")
            return

    # Run Preview Simulation
    json_output = "preview_out.json"
    log_output = "preview_out.log"
    
    if os.path.exists(json_output): os.remove(json_output)
    if os.path.exists(log_output): os.remove(log_output)
    
    config = load_config()
    num_sim_points = config.get("PREVIEW_SIM_POINTS", 10)

    # Calculate number of instructions for preview
    number_instructions = int((config.get("TOTAL_INSTRUCTIONS", 700000000) / config.get("SIM_POINTS", 280)) * num_sim_points)
    
    cmd = [
        args.binary,
        "--no-repeat-traces",
        "-e", str(number_instructions),
        "-p", str(num_sim_points),
        "-w", "0",
        "--simulation-instructions", str(number_instructions),
        "--json", json_output,
        args.trace
    ]
    
    print(f"Running preview simulation: {' '.join(cmd)}")

    # Measure time taken
    start_time = time.time()
    
    with open(log_output, "w") as f_log:
        process = subprocess.Popen(cmd, stdout=f_log, stderr=subprocess.STDOUT)
        process.wait()

    end_time = time.time()
    elapsed_time = end_time - start_time
        
    if process.returncode != 0:
        print("Error: ChampSim simulation failed.")
        return

    print(f"Preview simulation completed in {elapsed_time:.2f} seconds.")

    # Parse output
    mappers = load_mappers()
    config_key_str, btb_sets = encode_config_str(args.config, mappers)
    
    with open(log_output, "r") as f:
        log_content = f.read()
    with open(json_output, "r") as f:
        json_data = json.load(f)
    
    parsed_data = parse_output(json_data, log_content, mappers, "preview_config", args.config, btb_sets)
    
    if not parsed_data:
        print("Error: Failed to parse simulation output.")
        return
        
    # Generate features
    input_data = { config_key_str : parsed_data }
    
    keys, feature_names_dict = generator(input_data, n=num_sim_points, train_after_warmup=False, warmup_period=0) 
    
    target_key = "cumulative_ipc"
    if target_key not in keys:
        print("Error: cumulative_ipc not found in features.")
        return
        
    X_list, _ = keys[target_key]
    feature_names = feature_names_dict[target_key]
    
    X = pd.DataFrame(X_list, columns=feature_names)
    
    # Predict using loaded model
    with open(model_path, "rb") as f:
        loaded_data = pickle.load(f)

    # Check for resolution consistency if metadata exists
    curr_total = config.get("TOTAL_INSTRUCTIONS", 700000000)
    curr_points = config.get("SIM_POINTS", 280)
    curr_resolution = int(curr_total / curr_points)

    if isinstance(loaded_data, dict) and "phase_resolution" in loaded_data:
        trained_resolution = loaded_data["phase_resolution"]
        if trained_resolution != curr_resolution:
            print(f"WARNING: Model resolution mismatch!")
            print(f"Model trained with phase size: {trained_resolution}")
            print(f"Current config phase size: {curr_resolution}")
    
    if isinstance(loaded_data, dict) and "model" in loaded_data:
        model = loaded_data["model"]
        model_features = loaded_data.get("feature_names", [])
    else:
        model = loaded_data
        model_features = []
        
    if model_features:
        missing = [f for f in model_features if f not in X.columns]
        if missing:
            print(f"Error: The following features expected by the model are missing from input: {missing}")
            return
            
        X = X[model_features]
    else:
        print("Warning: Model does not contain feature names. Assuming input feature order matches training.")
        
    prediction = model.predict(X)
    print(f"\nPredicted Cumulative IPC: {prediction[0]:.4f}")

    # Can run full simulation for comparison (may be slow)
    if args.compare:
        print(f"\nRunning full simulation for comparison ({args.compare_instructions} instructions)...")
        json_output_full = "full_out.json"
        log_output_full = "full_out.log"
        
        if os.path.exists(json_output_full): os.remove(json_output_full)
        if os.path.exists(log_output_full): os.remove(log_output_full)
        
        cmd_full = [
            args.binary,
            "--warmup-instructions", "0",
            "--simulation-instructions", str(args.compare_instructions),
            "--json", json_output_full,
            args.trace
        ]
        
        print(f"Running: {' '.join(cmd_full)}")

        start_time_full = time.time()
        
        with open(log_output_full, "w") as f_log:
            process = subprocess.Popen(cmd_full, stdout=f_log, stderr=subprocess.STDOUT)
            process.wait()

        end_time_full = time.time()
        elapsed_time_full = end_time_full - start_time_full
            
        if process.returncode != 0:
            print("Error: Full ChampSim simulation failed.")
            return

        print(f"Full simulation completed in {elapsed_time_full:.2f} seconds.")

        if os.path.exists(json_output_full):
            with open(json_output_full, "r") as f:
                try:
                    full_data = json.load(f)
                    if isinstance(full_data, list):
                        sim_stats = full_data[-1].get("sim", {}).get("cores", [{}])[0]
                    else:
                        sim_stats = full_data.get("sim", {}).get("cores", [{}])[0]
                    
                    instr = sim_stats.get("instructions", 0)
                    cycles = sim_stats.get("cycles", 0)
                    
                    if cycles > 0:
                        actual_ipc = instr / cycles
                        print(f"\nActual Cumulative IPC: {actual_ipc:.4f}")
                        
                        error = abs(actual_ipc - prediction[0]) / actual_ipc * 100
                        print(f"Absolute Error: {error:.2f}%")
                    else:
                        print("Error: Cycles count is 0, cannot calculate IPC.")
                        
                except Exception as e:
                    print(f"Error parsing full simulation output: {e}")
        else:
            print("Error: Full simulation JSON output not found.")

if __name__ == "__main__":
    try:
        config = load_config()
        default_instructions = config.get("TOTAL_INSTRUCTIONS", 700000000)
    except:
        default_instructions = 700000000

    parser = argparse.ArgumentParser(description="Predict performance using EarlyPerf model.")
    parser.add_argument("--trace", required=True, help="Path to evaluation trace")
    parser.add_argument("--binary", required=True, help="Path to ChampSim binary")
    parser.add_argument("--config", required=True, help="Path to ChampSim config JSON used for binary")
    parser.add_argument("--models_dir", default="models", help="Directory containing trained models")
    parser.add_argument("--model_name", help="Specific model file to use (optional)")
    
    # Comparison args
    parser.add_argument("--compare", action="store_true", help="Run full simulation to compare actual IPC")
    parser.add_argument("--compare_instructions", type=int, default=700000000, help="Number of instructions for full comparison run")
    
    args = parser.parse_args()
    run_prediction(args)

