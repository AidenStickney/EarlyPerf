import numpy as np
import os
import json

def get_clean_trace_name(filename):
    """
    Strips known trace extensions to return the base benchmark name.
    """
    base = os.path.basename(filename)
    
    extensions = []
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            extensions = config.get("TRACE_EXTENSIONS", [])
    except Exception:
        pass
        
    # Default fallbacks
    if not extensions:
        extensions = [
            ".champsimtrace.xz", 
            ".limit.champsimtrace.xz",
            ".champsimtrace", 
            ".trace.xz", 
            ".trace", 
            ".xz"
        ]
        
    # Ordered by length to catch longest match first
    extensions.sort(key=len, reverse=True)
    
    for ext in extensions:
        if base.endswith(ext):
            return base[:-len(ext)]
    
    return base

def encode_config(config_data, extra_params=None):
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
    
    pass

def target_final_nonsplit(y):
    """
    Take the last value of the target series.
    """
    new_ys = []
    for y_indiv in y:
        new_ys.append(y_indiv[-1])
    return new_ys

def generator(input_data, n=10, train_after_warmup=False, warmup_period=80):
    """
    Process raw simulation data into features (X) and targets (y).
    input_data: Dict { "key_string" : { metric: [values] } }
    """
    keys_temp = {}
    y_value_names_dict_temp = {}
    
    # Params
    sma_window_size = 5
    
    # Basic keys to ignore
    bad_keys = [
        "global_predictor_bits", "chooser_predictor_entries", "chooser_predictor_bits", 
        "global_predictor_entries", "fp_instructions", "committed_fp_instructions", 
        "total_instructions", "committed_instructions", "area", "peak_power", 
        "total_leakage", "peak_dynamic", "subthreshold_leakage", "gate_leakage", 
        "runtime_dynamic", "dcache_read_miss_ratio", "dcache_write_miss_ratio", 
        "icache_read_miss_ratio",
    ]

    for key, key_data in input_data.items():
        if key == "champsim_info": continue 

        if len(keys_temp) == 0:
            for metric in key_data:
                if metric != "champsim_info":
                    keys_temp[metric] = [[], []] # Features, Targets

        for target_key in keys_temp.keys():
            y_values = []
            y_values_names = []
            
            # Dynamic time series features
            if n != 0:
                for metric in keys_temp.keys():
                    if metric != "champsim_info" and metric not in bad_keys:
                        start__ = warmup_period if train_after_warmup else 0
                        slice_ = key_data[metric][start__ : start__ + n]
                        
                        # Raw time series features
                        if metric != target_key:
                            y_values_names.extend([f"{metric}_{i}" for i in range(len(slice_))])
                            y_values.extend(slice_)
                        else:
                            y_values_names.extend([f"{metric}_{i}" for i in range(len(slice_))])
                            y_values.extend(slice_)
                        
                        # SMA
                        if len(slice_) >= sma_window_size:
                            sma_values = np.convolve(slice_, np.ones(sma_window_size)/sma_window_size, mode='valid')
                            y_values_names.extend([f"sma_{metric}_{i}" for i in range(len(sma_values))])
                            y_values.extend(sma_values.tolist())

            # Static architectural features
            arch_features = [
                "Frequency", "iFetchBufferSize", "DecodeBufferSize", "DispatchBufferSize", "RegisterFileSize", "ROBSize",
                "LQSize", "SQSize", "FetchWidth", "DecodeWidth", "DispatchWidth", "ExecuteWidth",
                "LQWidth", "SQWidth", "RetireWidth", "MispredictPenalty", "SchedulerSize", "BranchPredictor",
                "DIBWindowSize", "DIBSets", "DIBWays", "L1ISets", "L1IWays", "L1IRQSize", "L1IWQSize",
                "L1IPQSize", "L1IMSHRSize", "L1IPrefetcher", "L1DSets", "L1DWays", "L1DRQSize",
                "L1DWQSize", "L1DPQSize", "L1DMSHRSize", "L1DPrefetcher", "L2Sets", "L2Ways",
                "L2Prefetcher", "LLCSets", "LLCWays", "LLCPrefetcher",
                "PhysicalMemoryChannels", "PhysicalMemoryRanks","btbReturnStackMaxSize",
                "btbReturnStackNumCallSizeTrackers", "btbIndirectPredictorSize", "btbDirectPredictorSets",
                "L1ILatency", "L1DLatency", "L2Latency", "LLCLatency", "PhysicalMemoryDataRate",
                "PhysicalMemoryBankGroups", "PhysicalMemoryBanks", "PhysicalMemoryBankRows",
                "PhysicalMemoryBankColumns", "PhysicalMemoryChannelWidth", "PhysicalMemoryWQSize",
                "PhysicalMemoryRQSize", "PhysicalMemorytCAS", "PhysicalMemorytRCD",
                "PhysicalMemorytRP", "PhysicalMemorytRAS", "PhysicalMemoryRefreshPeriod",
                "PhysicalMemoryRefreshesPerPeriod"
            ]
            y_values_names.extend(arch_features)
            
            try:
                key_values = [int(k) for k in key.split("_")]
                
                y_values.extend(key_values)
            except ValueError: # unusable
                pass

            # Target series
            if target_key in key_data:
                target_series = key_data[target_key]
                keys_temp[target_key][0].append(y_values)
                keys_temp[target_key][1].append(target_series)
            
            if target_key not in y_value_names_dict_temp:
                y_value_names_dict_temp[target_key] = y_values_names

    return keys_temp, y_value_names_dict_temp