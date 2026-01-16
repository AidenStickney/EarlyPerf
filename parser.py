import os
import json
import pickle
import re
import sys
import sqlite3
from utils.json_reader import parse_output

debug = False

def debug_print(*args, **kwargs):
    if debug:
        print(*args, **kwargs)

CHAMPSIM_PATH = json.load(open("config.json", "r"))["CHAMPSIM_DIR"]
MAPPERS_PATH = "mappers.json"
SIMULATION_RUNS = [json.load(open("config.json", "r"))["SIMULATION_RUN"]]
TOP_TRACE_FOLDER = json.load(open("config.json", "r"))["TOP_TRACE_FOLDER"]
TRACE_DIR = json.load(open("config.json", "r"))["TRACE_DIR"]
NUM_CONFIGS = int(json.load(open("config.json", "r"))["NUM_CONFIGS"])
DB_FILE = json.load(open("config.json", "r"))["DB_FILE"]
DATA_DIR = json.load(open("config.json", "r")).get("DATA_DIR", "data")
OUTPUT_DIR = json.load(open("config.json", "r")).get("OUTPUT_DIR", "output")

def get_config_from_db(cursor, trace, config_number, mappers):
    binary_name = f"run_champsim_{config_number}"
    trace_name = trace if trace.endswith(".champsimtrace.xz") else f"{trace}.champsimtrace.xz"
    
    query = """
        SELECT 
            frequency, ifetch_buffer_size, decode_buffer_size, dispatch_buffer_size, register_file_size,
            rob_size, lq_size, sq_size, fetch_width, decode_width, dispatch_width, execute_width,
            lq_width, sq_width, retire_width, mispredict_penalty, scheduler_size, branch_predictor, 
            btb_return_stack_max_size, btb_return_stack_num_call_size_trackers, btb_indirect_predictor_size, btb_direct_predictor_sets, 
            dib_window_size, dib_sets, dib_ways, 
            l1i_sets, l1i_ways, l1i_rq_size, l1i_wq_size, l1i_pq_size, l1i_mshr_size, l1i_latency, l1i_prefetcher, 
            l1d_sets, l1d_ways, l1d_rq_size, l1d_wq_size, l1d_pq_size, l1d_mshr_size, l1d_latency, l1d_prefetcher, 
            l2_sets, l2_ways, l2_latency, l2_prefetcher, 
            llc_sets, llc_ways, llc_latency, llc_prefetcher, 
            physical_memory_model, physical_memory_data_rate, physical_memory_channels, physical_memory_ranks,
            physical_memory_bankgroups, physical_memory_banks, physical_memory_bank_rows,
            physical_memory_bank_columns, physical_memory_channel_width, physical_memory_wq_size,
            physical_memory_rq_size, physical_memory_tCAS, physical_memory_tRCD, physical_memory_tRP,
            physical_memory_tRAS, physical_memory_refresh_period, physical_memory_refreshes_per_period
        FROM configs 
        WHERE trace = ? AND binary_name = ?
    """
    cursor.execute(query, (trace_name, binary_name))
    row = cursor.fetchone()
    
    if not row:
        return None, None, None
        
    (frequency, ifetch_buffer_size, decode_buffer_size, dispatch_buffer_size, register_file_size,
    rob_size, lq_size, sq_size, fetch_width, decode_width, dispatch_width, execute_width,
    lq_width, sq_width, retire_width, mispredict_penalty, scheduler_size, branch_predictor, 
    btb_return_stack_max_size, btb_return_stack_num_call_size_trackers, btb_indirect_predictor_size, btb_direct_predictor_sets, 
    dib_window_size, dib_sets, dib_ways, 
    l1i_sets, l1i_ways, l1i_rq_size, l1i_wq_size, l1i_pq_size, l1i_mshr_size, l1i_latency, l1i_prefetcher, 
    l1d_sets, l1d_ways, l1d_rq_size, l1d_wq_size, l1d_pq_size, l1d_mshr_size, l1d_latency, l1d_prefetcher, 
    l2_sets, l2_ways, l2_latency, l2_prefetcher, 
    llc_sets, llc_ways, llc_latency, llc_prefetcher, 
    physical_memory_model, physical_memory_data_rate, physical_memory_channels, physical_memory_ranks,
    physical_memory_bankgroups, physical_memory_banks, physical_memory_bank_rows,
    physical_memory_bank_columns, physical_memory_channel_width, physical_memory_wq_size,
    physical_memory_rq_size, physical_memory_tCAS, physical_memory_tRCD, physical_memory_tRP,
    physical_memory_tRAS, physical_memory_refresh_period, physical_memory_refreshes_per_period) = row

    def get_val(cat, idx): # Helper to map index to value
        cat_map = mappers["mappers"].get(cat, {})
        val = cat_map.get(str(idx))
        if val is not None:
             return val
        if cat == "iFetchBufferSize" or cat == "BranchPredictor" or cat == "instruction_buffer_size":
             print(f"DEBUG: cat={cat} idx={idx} (type {type(idx)}) lookup_key='{str(idx)}' val={val} map_keys={list(cat_map.keys())[:5]}")
        return idx

    config = {
        "block_size": 64, 
        "ooo_cpu": [{
            "branch_predictor": get_val("BranchPredictor", branch_predictor),
            "frequency": get_val("Frequency", frequency),
            "ifetch_buffer_size": get_val("iFetchBufferSize", ifetch_buffer_size),
            "decode_buffer_size": get_val("DecodeBufferSize", decode_buffer_size),
            "dispatch_buffer_size": get_val("DispatchBufferSize", dispatch_buffer_size),
            "rob_size": get_val("ROBSize", rob_size),
            "lq_size": get_val("LQSize", lq_size),
            "sq_size": get_val("SQSize", sq_size),
            "fetch_width": get_val("FetchWidth", fetch_width),
            "decode_width": get_val("DecodeWidth", decode_width),
            "dispatch_width": get_val("DispatchWidth", dispatch_width),
            "execute_width": get_val("ExecuteWidth", execute_width),
            "lq_width": get_val("LQWidth", lq_width),
            "sq_width": get_val("SQWidth", sq_width),
            "retire_width": get_val("RetireWidth", retire_width),
            "mispredict_penalty": get_val("MispredictPenalty", mispredict_penalty),
            "scheduler_size": get_val("SchedulerSize", scheduler_size)
        }],
        "DIB": {
            "window_size": get_val("DIBWindowSize", dib_window_size),
            "sets": get_val("DIBSets", dib_sets),
            "ways": get_val("DIBWays", dib_ways)
        },
        "L1I": {
            "sets": get_val("L1ISets", l1i_sets),
            "ways": get_val("L1IWays", l1i_ways),
            "rq_size": get_val("L1IRQSize", l1i_rq_size),
            "wq_size": get_val("L1IWQSize", l1i_wq_size),
            "pq_size": get_val("L1IPQSize", l1i_pq_size),
            "mshr_size": get_val("L1IMSHRSize", l1i_mshr_size),
            "prefetcher": get_val("L1IPrefetcher", l1i_prefetcher),
            "latency": get_val("L1ILatency", l1i_latency) if mappers["mappers"].get("L1ILatency") else float(l1i_latency)
        },
        "L1D": {
            "sets": get_val("L1DSets", l1d_sets),
            "ways": get_val("L1DWays", l1d_ways),
            "rq_size": get_val("L1DRQSize", l1d_rq_size),
            "wq_size": get_val("L1DWQSize", l1d_wq_size),
            "pq_size": get_val("L1DPQSize", l1d_pq_size),
            "mshr_size": get_val("L1DMSHRSize", l1d_mshr_size),
            "prefetcher": get_val("L1DPrefetcher", l1d_prefetcher),
            "latency": get_val("L1DLatency", l1d_latency) if mappers["mappers"].get("L1DLatency") else float(l1d_latency)
        },
        "L2C": {
            "sets": get_val("L2Sets", l2_sets),
            "ways": get_val("L2Ways", l2_ways),
            "mshr_size": 32,
            "prefetcher": get_val("L2Prefetcher", l2_prefetcher),
            "latency": get_val("L2Latency", l2_latency) if mappers["mappers"].get("L2Latency") else float(l2_latency)
        },
        "LLC": {
            "sets": get_val("LLCSets", llc_sets),
            "ways": get_val("LLCWays", llc_ways),
            "mshr_size": 64,
            "prefetcher": get_val("LLCPrefetcher", llc_prefetcher),
            "latency": get_val("LLCLatency", llc_latency) if mappers["mappers"].get("LLCLatency") else float(llc_latency)
        },
        "physical_memory": {
            "data_rate": get_val("PhysicalMemoryDataRate", physical_memory_data_rate),
            "channels": get_val("PhysicalMemoryChannels", physical_memory_channels),
            "ranks": get_val("PhysicalMemoryRanks", physical_memory_ranks),
            "bankgroups": get_val("PhysicalMemoryBankGroups", physical_memory_bankgroups),
            "banks": get_val("PhysicalMemoryBanks", physical_memory_banks),
            "bank_rows": get_val("PhysicalMemoryBankRows", physical_memory_bank_rows),
            "bank_columns": get_val("PhysicalMemoryBankColumns", physical_memory_bank_columns),
            "channel_width": get_val("PhysicalMemoryChannelWidth", physical_memory_channel_width),
            "wq_size": get_val("PhysicalMemoryWQSize", physical_memory_wq_size),
            "rq_size": get_val("PhysicalMemoryRQSize", physical_memory_rq_size),
            "tCAS": get_val("PhysicalMemorytCAS", physical_memory_tCAS),
            "tRCD": get_val("PhysicalMemorytRCD", physical_memory_tRCD),
            "tRP": get_val("PhysicalMemorytRP", physical_memory_tRP),
            "tRAS": get_val("PhysicalMemorytRAS", physical_memory_tRAS),
            "refresh_period": get_val("PhysicalMemoryRefreshPeriod", physical_memory_refresh_period),
            "refreshes_per_period": get_val("PhysicalMemoryRefreshesPerPeriod", physical_memory_refreshes_per_period)
        }
    }
    
    # Construct key string which is underscore separated encoded values
    encoded_values = [
        str(frequency), str(ifetch_buffer_size), str(decode_buffer_size), str(dispatch_buffer_size), str(register_file_size),
        str(rob_size), str(lq_size), str(sq_size), str(fetch_width), str(decode_width), str(dispatch_width), str(execute_width),
        str(lq_width), str(sq_width), str(retire_width), str(mispredict_penalty), str(scheduler_size), str(branch_predictor),
        str(dib_window_size), str(dib_sets), str(dib_ways),
        str(l1i_sets), str(l1i_ways), str(l1i_rq_size), str(l1i_wq_size), str(l1i_pq_size), str(l1i_mshr_size), str(l1i_prefetcher),
        str(l1d_sets), str(l1d_ways), str(l1d_rq_size), str(l1d_wq_size), str(l1d_pq_size), str(l1d_mshr_size), str(l1d_prefetcher),
        str(l2_sets), str(l2_ways), str(l2_prefetcher),
        str(llc_sets), str(llc_ways), str(llc_prefetcher),
        str(physical_memory_channels), str(physical_memory_ranks),
        str(btb_return_stack_max_size), str(btb_return_stack_num_call_size_trackers), str(btb_indirect_predictor_size), str(btb_direct_predictor_sets),
        str(l1i_latency), str(l1d_latency), str(l2_latency), str(llc_latency),
        str(physical_memory_data_rate), str(physical_memory_bankgroups), str(physical_memory_banks), str(physical_memory_bank_rows),
        str(physical_memory_bank_columns), str(physical_memory_channel_width), str(physical_memory_wq_size), str(physical_memory_rq_size),
        str(physical_memory_tCAS), str(physical_memory_tRCD), str(physical_memory_tRP), str(physical_memory_tRAS),
        str(physical_memory_refresh_period), str(physical_memory_refreshes_per_period)
    ]
    
    key_string = "_".join(encoded_values)
    
    btb_val = btb_direct_predictor_sets
    btb_sets_map = mappers["mappers"].get("BTBDirectPredictorSets", {})
    for k, v in btb_sets_map.items():
        if v == btb_direct_predictor_sets:
            btb_val = int(k)
            break
            
    return config, key_string, btb_val

def parser():
    with open(MAPPERS_PATH, 'r') as f:
        mappers = json.load(f)
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    for simulation_run in SIMULATION_RUNS:
        DATA_LONG_PATH = os.path.join(DATA_DIR, simulation_run)
        if not os.path.exists(DATA_LONG_PATH):
            os.makedirs(DATA_LONG_PATH)

        JSON_DIR = os.path.join(OUTPUT_DIR, simulation_run, "json")
        LOG_DIR = os.path.join(OUTPUT_DIR, simulation_run, "logs")
        
        if not os.path.exists(JSON_DIR):
            print(f"No output found at {JSON_DIR}")
            continue

        json_outputs = os.listdir(JSON_DIR)
        
        # Group by benchmark
        bench_files = {}
        for f in json_outputs:
            if not f.endswith(".json") or "json_long_" not in f: continue
            parts = f.replace("json_long_", "").split("_")
            config_id = parts[-1].replace(".json", "")
            trace = "_".join(parts[:-1])
            
            if trace not in bench_files: bench_files[trace] = []
            bench_files[trace].append((config_id, f))
            
        print(f"Found {len(bench_files)} benchmarks.")
        
        for benchmark, configs in bench_files.items():
            output_pickle_path = os.path.join(DATA_LONG_PATH, f"{benchmark}_parsed_data.pickle")
            if os.path.exists(output_pickle_path):
                continue

            if len(configs) < NUM_CONFIGS:
                continue

            print(f"Parsing benchmark {benchmark} with {len(configs)} configs.")
                
            all_data = {}
            for config_id, json_filename in configs:
                try:
                    log_filename = f"logs_long_{benchmark}_{config_id}.log"
                    log_path = os.path.join(LOG_DIR, log_filename)
                    if not os.path.exists(log_path):
                        log_path = log_path.replace(".log", ".txt")
                        if not os.path.exists(log_path):
                            continue
                    
                    with open(os.path.join(JSON_DIR, json_filename), "r") as f:
                        json_data = json.load(f)
                    with open(log_path, "r") as f:
                        log_content = f.read()
                        
                    config_dict, key_string, btb_sets = get_config_from_db(cursor, benchmark, config_id, mappers)
                    
                    if config_dict is None:
                        continue
                        
                    output_data = parse_output(json_data, log_content, mappers, config_id, config_dict, btb_sets)
                    
                    if output_data:
                        all_data[key_string] = output_data
                        
                except Exception as e:
                    print(f"Error parsing {benchmark} {config_id}: {e}")
            
            if all_data:
                print(f"Saving to {output_pickle_path}")
                with open(output_pickle_path, "wb+") as f:
                    pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                 print(f"No data for {benchmark}")

    conn.close()

if __name__ == "__main__":
    parser()
