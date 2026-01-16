import random
import json
import subprocess
import sys
from datetime import datetime
import time
import math
import os
import re
import sqlite3
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)

CHAMPSIM_DIR = resolve_path(config["CHAMPSIM_DIR"])
TOTAL_INSTRUCTIONS = config["TOTAL_INSTRUCTIONS"]
SIM_INSTRUCTIONS = TOTAL_INSTRUCTIONS
ESTIMATED_INSTRUCTIONS = TOTAL_INSTRUCTIONS

DB_FILE = resolve_path(config["DB_FILE"])
TRACES_TO_RUN = config["TRACES_TO_RUN"]
TRACES_OFFSET = config["TRACES_OFFSET"]
TRACE_DIR = resolve_path(config["TRACE_DIR"])
WALL_TIME = config["WALL_TIME"]
MEM_SIZE_PER_JOB_MB = config["MEM_SIZE_PER_JOB_MB"]
NUM_CONFIGS = int(config["NUM_CONFIGS"])
SIM_POINTS = config["SIM_POINTS"]
SIMULATION_RUN = config["SIMULATION_RUN"]
ACCOUNT = config["ACCOUNT"]
CACTI_PATH = resolve_path(config["CACTI_DIR"])

SLURM_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slurm_template.txt")
try:
    with open(SLURM_TEMPLATE_PATH, "r") as f:
        SLURM_TEMPLATE = f.read()
except FileNotFoundError:
    print(f"Error: SLURM template not found at {SLURM_TEMPLATE_PATH}")
    sys.exit(1)

# Init the database
def initialize_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS configs (
                        simulation_run TEXT,
                        trace TEXT NOT NULL,
                        sim_instructions INTEGER,
                        estimated_instructions INTEGER,
                        sim_points INTEGER,
                        frequency INTEGER,
                        ifetch_buffer_size INTEGER,
                        decode_buffer_size INTEGER,
                        dispatch_buffer_size INTEGER,
                        register_file_size INTEGER,
                        rob_size INTEGER,
                        lq_size INTEGER,
                        sq_size INTEGER,
                        fetch_width INTEGER,
                        decode_width INTEGER,
                        dispatch_width INTEGER,
                        execute_width INTEGER,
                        lq_width INTEGER,
                        sq_width INTEGER,
                        retire_width INTEGER,
                        mispredict_penalty INTEGER,
                        scheduler_size INTEGER,
                        branch_predictor TEXT,
                        btb_return_stack_max_size INTEGER,
                        btb_return_stack_num_call_size_trackers INTEGER,
                        btb_indirect_predictor_size INTEGER,
                        btb_direct_predictor_sets INTEGER,
                        dib_window_size INTEGER,
                        dib_sets INTEGER,
                        dib_ways INTEGER,
                        l1i_sets INTEGER,
                        l1i_ways INTEGER,
                        l1i_rq_size INTEGER,
                        l1i_wq_size INTEGER,
                        l1i_pq_size INTEGER,
                        l1i_mshr_size INTEGER,
                        l1i_latency INTEGER,
                        l1i_prefetcher TEXT,
                        l1d_sets INTEGER,
                        l1d_ways INTEGER,
                        l1d_rq_size INTEGER,
                        l1d_wq_size INTEGER,
                        l1d_pq_size INTEGER,
                        l1d_mshr_size INTEGER,
                        l1d_latency INTEGER,
                        l1d_prefetcher TEXT,
                        l2_sets INTEGER,
                        l2_ways INTEGER,
                        l2_latency INTEGER,
                        l2_prefetcher TEXT,
                        llc_sets INTEGER,
                        llc_ways INTEGER,
                        llc_latency INTEGER,
                        llc_prefetcher TEXT,
                        physical_memory_model TEXT,
                        physical_memory_data_rate INTEGER,
                        physical_memory_channels INTEGER,
                        physical_memory_ranks INTEGER,
                        physical_memory_bankgroups INTEGER,
                        physical_memory_banks INTEGER,
                        physical_memory_bank_rows INTEGER,
                        physical_memory_bank_columns INTEGER,
                        physical_memory_channel_width INTEGER,
                        physical_memory_wq_size INTEGER,
                        physical_memory_rq_size INTEGER,
                        physical_memory_tCAS INTEGER,
                        physical_memory_tRCD INTEGER,
                        physical_memory_tRP INTEGER,
                        physical_memory_tRAS INTEGER,
                        physical_memory_refresh_period INTEGER,
                        physical_memory_refreshes_per_period INTEGER,
                        directory TEXT,
                        timestamp TEXT,
                        status TEXT,
                        result TEXT,
                        duration REAL,
                        binary_name TEXT,
                        additional_info TEXT,
                        PRIMARY KEY (simulation_run, trace, sim_instructions, estimated_instructions, sim_points, frequency, ifetch_buffer_size, decode_buffer_size, dispatch_buffer_size, register_file_size, rob_size, lq_size, sq_size)
                    )''')
    conn.commit()
    conn.close()

# Decode the encoded configuration to human-readable format
def decode(act_encoded, mappers):
    act_decoded = {}
    parameter_keys = list(mappers["mappers"].keys())
    if isinstance(act_encoded, dict):
        for key in parameter_keys:
            act_decoded[key] = mappers["mappers"][key][str(act_encoded[key])]

        # Manually decode PhysicalMemory parameters and cache latencies
        for key in list(act_encoded.keys()):
            if (key.startswith("PhysicalMemory") or "Latency" in key) and key not in ["PhysicalMemoryModel", "PhysicalMemoryRanks", "PhysicalMemoryChannels"]:
                # Already decoded
                act_decoded[key] = act_encoded[key]
    

    return act_decoded

def decode_from_db(act_encoded, mappers):
    act_decoded = {}

    parameter_mappings = mappers.get("json_mappings", {})
    mappers_dict = mappers.get("mappers", {})

    for group_name, parameters in parameter_mappings.items():
        for human_readable, encoded_key in parameters.items():
            # Check if the encoded key exists in the input dictionary
            if encoded_key in act_encoded:
                encoded_value = str(act_encoded[encoded_key])
                # Decode using the mappers dictionary
                if human_readable in mappers_dict and encoded_value in mappers_dict[human_readable]:
                    act_decoded[human_readable] = mappers_dict[human_readable][encoded_value]
                else:
                    act_decoded[human_readable] = act_encoded[encoded_key]
            else:
                act_decoded[human_readable] = None

    return act_decoded


# Check if the configuration is already in the database
def check_config_in_db(config, traces, simulation_run):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for trace in traces:
        cursor.execute('''SELECT COUNT(*) FROM configs WHERE
                          simulation_run = ? AND trace = ? AND sim_instructions = ? AND 
                          estimated_instructions = ? AND sim_points = ? AND frequency = ? AND 
                          ifetch_buffer_size = ? AND decode_buffer_size = ? AND 
                          dispatch_buffer_size = ? AND register_file_size = ? AND rob_size = ? AND lq_size = ? AND sq_size = ?''',
                       (simulation_run, trace, SIM_INSTRUCTIONS, ESTIMATED_INSTRUCTIONS, SIM_POINTS, 
                        (config["Frequency"] if "Frequency" in config else config["frequency"]),
                        (config["iFetchBufferSize"] if "iFetchBufferSize" in config else config["ifetch_buffer_size"]),
                        (config["DecodeBufferSize"] if "DecodeBufferSize" in config else config["decode_buffer_size"]),
                        (config["DispatchBufferSize"] if "DispatchBufferSize" in config else config["dispatch_buffer_size"]),
                        (config["RegisterFileSize"] if "RegisterFileSize" in config else config["register_file_size"]),
                        (config["ROBSize"] if "ROBSize" in config else config["rob_size"]),
                        (config["LQSize"] if "LQSize" in config else config["lq_size"]),
                        (config["SQSize"] if "SQSize" in config else config["sq_size"])))
        
        count = cursor.fetchone()[0]
        if count > 0:
            conn.close()
            return False

    conn.close()
    return True

# Get latencies for caches using cacti
def get_cache_latencies(sets, ways, cache_type, freq):
    cache_config_file = os.path.join(CACTI_PATH, f"cache_{cache_type}.cfg")
    cache_size = sets * ways * 64
    with open(cache_config_file, "r") as cache_file:
        cache_data = cache_file.read()
        cache_data = re.sub(r'-size \(bytes\) \d+', f'-size (bytes) {cache_size}', cache_data)
        cache_data = re.sub(r'-associativity \d+', f'-associativity {ways}', cache_data)
        with open(cache_config_file, "w") as cache_file:
            cache_file.write(cache_data)

    command = f"cd {CACTI_PATH} && ./cacti -infile {cache_config_file}"
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = process.communicate()
    if err.decode().strip():
        print(err.decode())
        return -1
        
    access_time = 0
    access_time_found = False
    for line in out.decode().split("\n"):
        if "Access time (ns):" in line:
            access_time = float(line.split(":")[1].strip())
            access_time_found = True
            break
            
    if not access_time_found:
        print("Error: Access time not found in cacti output")
        print(out.decode())
        return -1

    # Convert to cycles
    latency = round((access_time * freq) / 1000)

    return latency
    

# Select a configuration
def select_config(traces, iter):
    with open('mappers.json', 'r') as json_file:
        mappers = json.load(json_file)
        act_encoded = {}
        parameter_keys = list(mappers["mappers"].keys())
        stage_widths = ["FetchWidth", "DecodeWidth", "DispatchWidth", "ExecuteWidth", "LQWidth", "SQWidth", "RetireWidth"]
        found_good_value = False
        while not found_good_value: # Keep generating configurations until a new one is found
            for key in parameter_keys:
                if key in stage_widths: # skip stage widths for now
                    continue

                act_encoded[key] = random.randint(0, len(mappers["mappers"][key].keys()) - 1)

                # If the key is L1DPrefetcher and the value is "spp_dev", set the L1DMSHRSize >= 16
                if key == "L1DPrefetcher" and mappers["mappers"][key][str(act_encoded[key])] == "spp_dev":
                    index = 0
                    for i, value in enumerate(mappers["mappers"]["L1DMSHRSize"].values()):
                        if value == 16:
                            index = i
                            break

                    act_encoded["L1DMSHRSize"] = random.randint(index, len(mappers["mappers"]["L1DMSHRSize"].keys()) - 1)


            # Generate random stage widths with "steps" between stages
            last_value = 0
            for i in range(len(stage_widths)):
                # If first stage, set to a random value using the mappers
                if i == 0:
                    new_index = random.randint(0, len(mappers["mappers"][stage_widths[i]].keys()) - 1)
                    act_encoded[stage_widths[i]] = str(new_index)
                    new_value = mappers["mappers"][stage_widths[i]][str(new_index)]
                    last_value = new_value
                else:
                    max_step = 4
                    # Random move between [-4, 4] of the last value, use guassian distribution to favor smaller steps
                    step = 5
                    stage_max_width = list(mappers["mappers"][stage_widths[i]].values())[-1]
                    stage_min_width = list(mappers["mappers"][stage_widths[i]].values())[0]
                    step_found = False
                    while not step_found:
                        step = round(random.gauss(0, 2))
                        new_value = last_value + step
                        if new_value > stage_max_width or new_value < stage_min_width or abs(step) > max_step:
                            continue
                        else:
                            step_found = True
                    index_found = False
                    for j, value in enumerate(mappers["mappers"][stage_widths[i]].values()):
                        if value == new_value:
                            act_encoded[stage_widths[i]] = str(j)
                            index_found = True
                            break
                    if not index_found:
                        print("Error: Value not found in mapping")
                        continue

                    last_value = new_value

            good_latencies = True

            # Pick a random DRAM model
            with open("dram_models.json", "r") as dram_file:
                dram_models = json.load(dram_file)
                dram_model = random.choice(list(dram_models.keys()))
                act_encoded["PhysicalMemoryModel"] = dram_model
                for key in dram_models[dram_model]:
                    encoded_key = "PhysicalMemory" + key
                    act_encoded[encoded_key] = dram_models[dram_model][key]

                if "DDR3" in dram_model:
                    channels = random.choice(["0", "1", "2"])
                    act_encoded["PhysicalMemoryChannels"] = channels

                # Check total memory size
                size =  pow(int(act_encoded["PhysicalMemoryChannels"]),2) * pow(int(act_encoded["PhysicalMemoryRanks"]),2) *  act_encoded["PhysicalMemoryBankGroups"] * act_encoded["PhysicalMemoryBanks"] * act_encoded["PhysicalMemoryBankRows"] * act_encoded["PhysicalMemoryBankColumns"] * act_encoded["PhysicalMemoryChannelWidth"] / 8
                
                size = size / 1000000000 # Convert to GB

                # Ideally less than MEM_SIZE_PER_JOB_MB
                if size > 140:
                    good_latencies = False
            
            # L1I (cacti)
            if good_latencies:
                l1i_latency = get_cache_latencies(
                                                mappers["mappers"]["L1ISets"][str(act_encoded["L1ISets"])], 
                                                mappers["mappers"]["L1IWays"][str(act_encoded["L1IWays"])], 
                                                "l1i", 
                                                mappers["mappers"]["Frequency"][str(act_encoded["Frequency"])]
                                            )
                if l1i_latency == -1:
                    good_latencies = False
                    print("Error: L1I latency not found in cacti output")
                act_encoded["L1ILatency"] = l1i_latency

            #L1D
            if good_latencies:
                act_encoded["L1DLatency"] = get_cache_latencies(
                                                mappers["mappers"]["L1DSets"][str(act_encoded["L1DSets"])], 
                                                mappers["mappers"]["L1DWays"][str(act_encoded["L1DWays"])], 
                                                "l1d", 
                                                mappers["mappers"]["Frequency"][str(act_encoded["Frequency"])]
                                            )
                if act_encoded["L1DLatency"] == -1:
                    good_latencies = False
                    print("Error: L1D latency not found in cacti output")

            #L2
            if good_latencies:
                act_encoded["L2Latency"] = get_cache_latencies(
                                                mappers["mappers"]["L2Sets"][str(act_encoded["L2Sets"])], 
                                                mappers["mappers"]["L2Ways"][str(act_encoded["L2Ways"])], 
                                                "l2", 
                                                mappers["mappers"]["Frequency"][str(act_encoded["Frequency"])]
                                            )
                if act_encoded["L2Latency"] == -1:
                    good_latencies = False
                    print("Error: L2 latency not found in cacti output")

            #LLC
            if good_latencies:
                act_encoded["LLCLatency"] = get_cache_latencies(
                                                mappers["mappers"]["LLCSets"][str(act_encoded["LLCSets"])], 
                                                mappers["mappers"]["LLCWays"][str(act_encoded["LLCWays"])], 
                                                "l3", 
                                                mappers["mappers"]["Frequency"][str(act_encoded["Frequency"])]
                                            )
                if act_encoded["LLCLatency"] == -1:
                    good_latencies = False
                    print("Error: LLC latency not found in cacti output")

            if good_latencies:
                found_good_value = check_config_in_db(act_encoded, traces, SIMULATION_RUN)

        # Configuration will be saved for each trace later
        act_decoded = decode(act_encoded, mappers)
        binary_id = get_unique_binary_name("run_champsim", start_index=iter).split("_")[-1]
        write_to_json(act_decoded, mappers, binary_id)
        return act_encoded

# Write the configuration to a JSON file for ChampSim
def write_to_json(action, mappers, id):
    def validate_param(param_name):
        allowed = set(int(x) for x in mappers["mappers"].get(param_name, {}).values())
        value = int(action[param_name])
        if value not in allowed:
            raise ValueError(f"Unsafe value for {param_name}: {value}. Allowed: {sorted(allowed)}")
        return value

    btb_params = [
        "BTBReturnStackMaxSize",
        "BTBReturnStackNumCallSizeTrackers",
        "BTBIndirectPredictorSize",
        "BTBDirectPredictorSets"
    ]

    for param in btb_params:
        validate_param(param)

    champsim_ctrl_file = "champsim_config_" + str(id) + ".json"
    with open(f"{CHAMPSIM_DIR}/champsim_config.json", "r+") as JsonFile:
        data = json.load(JsonFile)
        data["executable_name"] = "run_champsim_" + str(id)
        for section, mappings in mappers["json_mappings"].items():
            for action_key, json_key in mappings.items():
                if section in data:
                    if isinstance(data[section], list): 
                        data[section][0][json_key] = action[action_key]
                    else:
                        data[section][json_key] = action[action_key]
        with open(os.path.join(CHAMPSIM_DIR, "configs", champsim_ctrl_file), "w+") as JsonFile:
            json.dump(data, JsonFile, indent=4)

    # Write BTB configuration
    with open(os.path.join(CHAMPSIM_DIR, "btb", "basic_btb", "return_stack.h"), "r+") as btb_return_stack_file:
        btb_return_stack_data = btb_return_stack_file.read()
        btb_return_stack_data = re.sub(r'static constexpr std::size_t max_size = \d+;', f'static constexpr std::size_t max_size = {action["BTBReturnStackMaxSize"]};', btb_return_stack_data)
        btb_return_stack_data = re.sub(r'static constexpr std::size_t num_call_size_trackers = \d+;', f'static constexpr std::size_t num_call_size_trackers = {action["BTBReturnStackNumCallSizeTrackers"]};', btb_return_stack_data)
        with open(os.path.join(CHAMPSIM_DIR, "btb", "basic_btb", "return_stack.h"), "w+") as btb_return_stack_file:
            btb_return_stack_file.write(btb_return_stack_data)
        
    with open(os.path.join(CHAMPSIM_DIR, "btb", "basic_btb", "indirect_predictor.h"), "r+") as btb_indirect_predictor_file:
        btb_indirect_predictor_data = btb_indirect_predictor_file.read()
        btb_indirect_predictor_data = re.sub(r'static constexpr std::size_t size = \d+;', f'static constexpr std::size_t size = {action["BTBIndirectPredictorSize"]};', btb_indirect_predictor_data)
        with open(os.path.join(CHAMPSIM_DIR, "btb", "basic_btb", "indirect_predictor.h"), "w+") as btb_indirect_predictor_file:
            btb_indirect_predictor_file.write(btb_indirect_predictor_data)

    with open(os.path.join(CHAMPSIM_DIR, "btb", "basic_btb", "direct_predictor.h"), "r+") as btb_direct_predictor_file:
        btb_direct_predictor_data = btb_direct_predictor_file.read()
        btb_direct_predictor_data = re.sub(r'static constexpr std::size_t sets = \d+;', f'static constexpr std::size_t sets = {action["BTBDirectPredictorSets"]};', btb_direct_predictor_data)
        with open(os.path.join(CHAMPSIM_DIR, "btb", "basic_btb", "direct_predictor.h"), "w+") as btb_direct_predictor_file:
            btb_direct_predictor_file.write(btb_direct_predictor_data)


# Get a unique binary name in ./bin directory
def get_unique_binary_name(base_name="run_champsim", start_index=0):
    binary_dir = "./bin"
    count = start_index
    binary_name = base_name + "_" + str(count)
    while os.path.exists(os.path.join(binary_dir, binary_name)):
        count += 1
        binary_name = base_name + "_" + str(count)
    return binary_name

# Save configuration to the database, including the binary name
def save_config_to_db(trace, config, binary_name, additional_info=None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Check if the configuration is already in the database
    cursor.execute('''SELECT COUNT(*) FROM configs WHERE simulation_run = ? AND trace = ? AND
                      sim_instructions = ? AND estimated_instructions = ? AND sim_points = ? AND
                      frequency = ? AND ifetch_buffer_size = ? AND decode_buffer_size = ? AND 
                      dispatch_buffer_size = ? AND register_file_size = ? AND rob_size = ? AND lq_size = ? AND sq_size = ?''',
                     (SIMULATION_RUN, trace, SIM_INSTRUCTIONS, ESTIMATED_INSTRUCTIONS, SIM_POINTS,
                     (config["Frequency"] if "Frequency" in config else config["frequency"]),
                        (config["iFetchBufferSize"] if "iFetchBufferSize" in config else config["ifetch_buffer_size"]),
                        (config["DecodeBufferSize"] if "DecodeBufferSize" in config else config["decode_buffer_size"]),
                        (config["DispatchBufferSize"] if "DispatchBufferSize" in config else config["dispatch_buffer_size"]),
                        (config["RegisterFileSize"] if "RegisterFileSize" in config else config["register_file_size"]),
                        (config["ROBSize"] if "ROBSize" in config else config["rob_size"]),
                        (config["LQSize"] if "LQSize" in config else config["lq_size"]),
                        (config["SQSize"] if "SQSize" in config else config["sq_size"])))
    count = cursor.fetchone()[0]

    if count == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "pending"

        cursor.execute('''INSERT INTO configs (simulation_run, trace, sim_instructions, estimated_instructions, sim_points, frequency, ifetch_buffer_size, decode_buffer_size,
                            dispatch_buffer_size, register_file_size, rob_size, lq_size, sq_size, fetch_width, decode_width, dispatch_width, execute_width, lq_width, sq_width, retire_width,
                            mispredict_penalty, scheduler_size, branch_predictor, btb_return_stack_max_size, btb_return_stack_num_call_size_trackers, btb_indirect_predictor_size, btb_direct_predictor_sets, dib_window_size, dib_sets, dib_ways, l1i_sets, l1i_ways, l1i_rq_size, l1i_wq_size, l1i_pq_size,
                            l1i_mshr_size, l1i_latency, l1i_prefetcher, l1d_sets, l1d_ways, l1d_rq_size, l1d_wq_size, l1d_pq_size, l1d_mshr_size, l1d_latency, l1d_prefetcher, l2_sets, l2_ways,
                            l2_latency, l2_prefetcher, llc_sets, llc_ways, llc_latency, llc_prefetcher, physical_memory_model, physical_memory_data_rate, physical_memory_channels, physical_memory_ranks, physical_memory_bankgroups,
                            physical_memory_banks, physical_memory_bank_rows, physical_memory_bank_columns, physical_memory_channel_width, physical_memory_wq_size, physical_memory_rq_size,
                            physical_memory_tCAS, physical_memory_tRCD, physical_memory_tRP, physical_memory_tRAS, physical_memory_refresh_period, physical_memory_refreshes_per_period,
                            directory, timestamp, status, result, duration, binary_name, additional_info)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            (SIMULATION_RUN, trace, SIM_INSTRUCTIONS, ESTIMATED_INSTRUCTIONS, SIM_POINTS,
                            (config["Frequency"] if "Frequency" in config else config["frequency"]),
                            (config["iFetchBufferSize"] if "iFetchBufferSize" in config else config["ifetch_buffer_size"]),
                            (config["DecodeBufferSize"] if "DecodeBufferSize" in config else config["decode_buffer_size"]),
                            (config["DispatchBufferSize"] if "DispatchBufferSize" in config else config["dispatch_buffer_size"]),
                            (config["RegisterFileSize"] if "RegisterFileSize" in config else config["register_file_size"]),
                            (config["ROBSize"] if "ROBSize" in config else config["rob_size"]),
                            (config["LQSize"] if "LQSize" in config else config["lq_size"]),
                            (config["SQSize"] if "SQSize" in config else config["sq_size"]),
                            (config["FetchWidth"] if "FetchWidth" in config else config["fetch_width"]),
                            (config["DecodeWidth"] if "DecodeWidth" in config else config["decode_width"]),
                            (config["DispatchWidth"] if "DispatchWidth" in config else config["dispatch_width"]),
                            (config["ExecuteWidth"] if "ExecuteWidth" in config else config["execute_width"]),
                            (config["LQWidth"] if "LQWidth" in config else config["lq_width"]),
                            (config["SQWidth"] if "SQWidth" in config else config["sq_width"]),
                            (config["RetireWidth"] if "RetireWidth" in config else config["retire_width"]),
                            (config["MispredictPenalty"] if "MispredictPenalty" in config else config["mispredict_penalty"]),
                            (config["SchedulerSize"] if "SchedulerSize" in config else config["scheduler_size"]),
                            (config["BranchPredictor"] if "BranchPredictor" in config else config["branch_predictor"]),
                            (config["BTBReturnStackMaxSize"] if "BTBReturnStackMaxSize" in config else config["btb_return_stack_max_size"]),
                            (config["BTBReturnStackNumCallSizeTrackers"] if "BTBReturnStackNumCallSizeTrackers" in config else config["btb_return_stack_num_call_size_trackers"]),
                            (config["BTBIndirectPredictorSize"] if "BTBIndirectPredictorSize" in config else config["btb_indirect_predictor_size"]),
                            (config["BTBDirectPredictorSets"] if "BTBDirectPredictorSets" in config else config["btb_direct_predictor_sets"]),
                            (config["DIBWindowSize"] if "DIBWindowSize" in config else config["dib_window_size"]),
                            (config["DIBSets"] if "DIBSets" in config else config["dib_sets"]),
                            (config["DIBWays"] if "DIBWays" in config else config["dib_ways"]),
                            (config["L1ISets"] if "L1ISets" in config else config["l1i_sets"]),
                            (config["L1IWays"] if "L1IWays" in config else config["l1i_ways"]),
                            (config["L1IRQSize"] if "L1IRQSize" in config else config["l1i_rq_size"]),
                            (config["L1IWQSize"] if "L1IWQSize" in config else config["l1i_wq_size"]),
                            (config["L1IPQSize"] if "L1IPQSize" in config else config["l1i_pq_size"]),
                            (config["L1IMSHRSize"] if "L1IMSHRSize" in config else config["l1i_mshr_size"]),
                            (config["L1ILatency"] if "L1ILatency" in config else config["l1i_latency"]),
                            (config["L1IPrefetcher"] if "L1IPrefetcher" in config else config["l1i_prefetcher"]),
                            (config["L1DSets"] if "L1DSets" in config else config["l1d_sets"]),
                            (config["L1DWays"] if "L1DWays" in config else config["l1d_ways"]),
                            (config["L1DRQSize"] if "L1DRQSize" in config else config["l1d_rq_size"]),
                            (config["L1DWQSize"] if "L1DWQSize" in config else config["l1d_wq_size"]),
                            (config["L1DPQSize"] if "L1DPQSize" in config else config["l1d_pq_size"]),
                            (config["L1DMSHRSize"] if "L1DMSHRSize" in config else config["l1d_mshr_size"]),
                            (config["L1DLatency"] if "L1DLatency" in config else config["l1d_latency"]),
                            (config["L1DPrefetcher"] if "L1DPrefetcher" in config else config["l1d_prefetcher"]),
                            (config["L2Sets"] if "L2Sets" in config else config["l2_sets"]),
                            (config["L2Ways"] if "L2Ways" in config else config["l2_ways"]),
                            (config["L2Latency"] if "L2Latency" in config else config["l2_latency"]),
                            (config["L2Prefetcher"] if "L2Prefetcher" in config else config["l2_prefetcher"]),
                            (config["LLCSets"] if "LLCSets" in config else config["llc_sets"]),
                            (config["LLCWays"] if "LLCWays" in config else config["llc_ways"]),
                            (config["LLCLatency"] if "LLCLatency" in config else config["llc_latency"]),
                            (config["LLCPrefetcher"] if "LLCPrefetcher" in config else config["llc_prefetcher"]),
                            (config["PhysicalMemoryModel"] if "PhysicalMemoryModel" in config else config["physical_memory_model"]),
                            (config["PhysicalMemoryDataRate"] if "PhysicalMemoryDataRate" in config else config["physical_memory_data_rate"]),
                            (config["PhysicalMemoryChannels"] if "PhysicalMemoryChannels" in config else config["physical_memory_channels"]),
                            (config["PhysicalMemoryRanks"] if "PhysicalMemoryRanks" in config else config["physical_memory_ranks"]),
                            (config["PhysicalMemoryBankGroups"] if "PhysicalMemoryBankGroups" in config else config["physical_memory_bankgroups"]),
                            (config["PhysicalMemoryBanks"] if "PhysicalMemoryBanks" in config else config["physical_memory_banks"]),
                            (config["PhysicalMemoryBankRows"] if "PhysicalMemoryBankRows" in config else config["physical_memory_bank_rows"]),
                            (config["PhysicalMemoryBankColumns"] if "PhysicalMemoryBankColumns" in config else config["physical_memory_bank_columns"]),
                            (config["PhysicalMemoryChannelWidth"] if "PhysicalMemoryChannelWidth" in config else config["physical_memory_channel_width"]),
                            (config["PhysicalMemoryWQSize"] if "PhysicalMemoryWQSize" in config else config["physical_memory_wq_size"]),
                            (config["PhysicalMemoryRQSize"] if "PhysicalMemoryRQSize" in config else config["physical_memory_rq_size"]),
                            (config["PhysicalMemorytCAS"] if "PhysicalMemorytCAS" in config else config["physical_memory_tCAS"]),
                            (config["PhysicalMemorytRCD"] if "PhysicalMemorytRCD" in config else config["physical_memory_tRCD"]),
                            (config["PhysicalMemorytRP"] if "PhysicalMemorytRP" in config else config["physical_memory_tRP"]),
                            (config["PhysicalMemorytRAS"] if "PhysicalMemorytRAS" in config else config["physical_memory_tRAS"]),   
                            (config["PhysicalMemoryRefreshPeriod"] if "PhysicalMemoryRefreshPeriod" in config else config["physical_memory_refresh_period"]),
                            (config["PhysicalMemoryRefreshesPerPeriod"] if "PhysicalMemoryRefreshesPerPeriod" in config else config["physical_memory_refreshes_per_period"]),
                            os.getcwd().split("/")[-1], timestamp, status, None, None, binary_name, additional_info))
        conn.commit()

    conn.close()
    return count == 0 

# Compile ChampSim with a unique binary name and save it to the DB
def setup_champsim(action_dict, iter):
    binary_name = get_unique_binary_name("run_champsim", start_index=iter)
    binary_index = binary_name.split("_")[-1]
    config_path = os.path.join(CHAMPSIM_DIR, "configs", "champsim_config_" + str(binary_index) + ".json")

    process = subprocess.Popen(
        [f"{CHAMPSIM_DIR}/config.sh", config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=CHAMPSIM_DIR
    )
    out, err = process.communicate()
    if err.decode().strip():
        print(err.decode())
        sys.exit(1)

    process = subprocess.Popen(
        ["make", "-C", CHAMPSIM_DIR], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = process.communicate()
    if "error" in err.decode().lower():
        print(err.decode())
        sys.exit(1)
    
    return binary_name

# Create batch job script with a unique binary name
def create_batch_job(repo_path, trace, trace_fp, config, binary_name):
    total_seconds = WALL_TIME * 3600
    hours = math.floor(total_seconds // 3600)
    minutes = math.floor((total_seconds % 3600) // 60)
    seconds = math.floor(total_seconds % 60)
    config_num = binary_name.split("_")[-1]
    
    # Use ROOT_DIR to construct paths
    batch_dir = os.path.join(ROOT_DIR, f"batch/{SIMULATION_RUN}/{trace}")
    script_path = os.path.join(batch_dir, f"batch_script_{trace}_{config_num}.sh")
    os.makedirs(batch_dir, exist_ok=True)
    
    slurm_script = SLURM_TEMPLATE.format(
        trace=trace, 
        output_location=os.path.join(ROOT_DIR, f"output/{SIMULATION_RUN}/logs/logs_long_{trace}_{config_num}.log"),
        iter=config_num,
        dur=f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}",
        mem=MEM_SIZE_PER_JOB_MB,
        repo_path=ROOT_DIR,
        json_output=os.path.join(ROOT_DIR, f"output/{SIMULATION_RUN}/json/json_long_{trace}_{config_num}.json"),
        champsim_dir=CHAMPSIM_DIR,
        binary_name=binary_name,
        script_path=script_path,
        sim_instructions=SIM_INSTRUCTIONS, 
        estimated_instructions=ESTIMATED_INSTRUCTIONS,
        sim_points=SIM_POINTS,
        account=ACCOUNT,
        trace_fp=trace_fp,
        freq = config["Frequency"] if "Frequency" in config else config["frequency"],
        ifetch= config["iFetchBufferSize"] if "iFetchBufferSize" in config else config["ifetch_buffer_size"],
        decode= config["DecodeBufferSize"] if "DecodeBufferSize" in config else config["decode_buffer_size"],
        dispatch= config["DispatchBufferSize"] if "DispatchBufferSize" in config else config["dispatch_buffer_size"],
        rob= config["ROBSize"] if "ROBSize" in config else config["rob_size"]
    )
    with open(script_path, "w") as script_file:
        script_file.write(slurm_script)

def extract_number_from_trace(filename):
    match = re.search(r'_(\d+)\.', filename)
    if match:
         return int(match.group(1))
    
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
        
    return -1

# Get configuration from the database based on binary name
def get_config_from_db(binary_name):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM configs WHERE binary_name = ? LIMIT 1''', (binary_name,))
    row = cursor.fetchone()
    if row:
        columns = [description[0] for description in cursor.description]
        config = dict(zip(columns, row))
        conn.close()
        return config
    else:
        conn.close()
        return None

def main():
    initialize_db()
    traces = os.listdir(TRACE_DIR)
    traces.sort(key=extract_number_from_trace)
    traces = traces[TRACES_OFFSET:]

    with open("mappers.json", "r") as json_file:
        mappers = json.load(json_file)

        # Ensure the output and batch directories exist
        os.makedirs(f"output/{SIMULATION_RUN}/logs", exist_ok=True)
        os.makedirs(f"output/{SIMULATION_RUN}/json", exist_ok=True)
        os.makedirs(f"batch/{SIMULATION_RUN}", exist_ok=True)
        
        os.makedirs(f"{CHAMPSIM_DIR}/bin", exist_ok=True)
        os.makedirs(f"{CHAMPSIM_DIR}/configs", exist_ok=True)

        # Get existing binaries
        existing_binaries = [f for f in os.listdir(f"{CHAMPSIM_DIR}/bin") if f.startswith("run_champsim")]

        # If more binaries exist than needed, ignore the extras
        if len(existing_binaries) > NUM_CONFIGS:
            existing_binaries = existing_binaries[:NUM_CONFIGS]

        # Process existing binaries
        print(f"Processing existing binaries in /bin")
        if existing_binaries:
            with tqdm(total=len(existing_binaries), desc="Processing Binaries", unit="binary") as bin_pbar:
                for binary_name in existing_binaries:
                    tqdm.write(f"Processing binary: {binary_name}")
                    config = get_config_from_db(binary_name)
                    if config:
                        for trace in traces[:TRACES_TO_RUN]:
                            # Check if output file (json) already exists
                            binary_id = binary_name.split("_")[-1]
                            json_output = f"{os.getcwd()}/output/{SIMULATION_RUN}/json/json_long_{trace}_{binary_id}.json"
                            if not os.path.exists(json_output):
                                trace_fp = os.path.join(TRACE_DIR, trace)
                                create_batch_job(os.getcwd(), trace, trace_fp, config, binary_name)
                                save_config_to_db(trace, config, binary_name)
                            else:
                                print(f"\tOutput file {json_output} already exists. Skipping")
                    else:
                        print(f"\tConfiguration for binary {binary_name} not found in database.")
                    bin_pbar.update(1)

        # Check if more configurations are needed
        num_existing_configs = len(existing_binaries)
        if NUM_CONFIGS > num_existing_configs:
            num_new_configs = NUM_CONFIGS - num_existing_configs
            tqdm.write(f"\nGenerating {num_new_configs} new configurations")
            with tqdm(total=num_new_configs, desc="Generating Configurations", unit="config") as config_pbar:
                for iter in range(num_existing_configs, NUM_CONFIGS):
                    action_dict = select_config(traces, iter)
                    binary_name = setup_champsim(action_dict, iter)
                    for trace in traces[:TRACES_TO_RUN]:
                        trace_fp = os.path.join(TRACE_DIR, trace)
                        create_batch_job(os.getcwd(), trace, trace_fp, action_dict, binary_name)
                        save_config_to_db(trace, action_dict, binary_name)
                    config_pbar.update(1)

if __name__ == "__main__":
    main()
