import json
import re
import xml.etree.ElementTree as ET

def champsim_config_reader(champsim_config, btb_direct_predictor_sets):
    res = {}
    if isinstance(champsim_config, str):
        with open(champsim_config, "r") as json_file:
            champsim_config = json.load(json_file)
    
    if "LLC" not in champsim_config:
        pass
        
    res["L30_buffer_sizes"] = str(champsim_config["LLC"]["mshr_size"]) + ",48,32,48"
    l3_capacity = str(
        champsim_config["block_size"]
        * champsim_config["LLC"]["sets"]
        * champsim_config["LLC"]["ways"]
    )
    l3_ways = champsim_config["LLC"]["ways"]
    res["L3_config"] = l3_capacity + f",64,{l3_ways},2,8,15,64,1"

    res["L20_buffer_sizes"] = str(champsim_config["L2C"]["mshr_size"]) + ",48,32,48"
    l2_capacity = str(
        champsim_config["block_size"]
        * champsim_config["L2C"]["sets"]
        * champsim_config["L2C"]["ways"]
    )
    l2_ways = champsim_config["L2C"]["ways"]
    res["L2_config"] = l2_capacity + f",64,{l2_ways},2,8,15,64,1"

    res["dcache_buffer_sizes"] = str(champsim_config["L1D"]["mshr_size"]) + ",48,32,48"
    l1d_capacity = str(
        champsim_config["block_size"]
        * champsim_config["L1D"]["sets"]
        * champsim_config["L1D"]["ways"]
    )
    l1d_ways = champsim_config["L1D"]["ways"]
    res["dcache_config"] = l1d_capacity + f",64,{l1d_ways},2,8,5,64,1"

    res["icache_buffer_sizes"] = str(champsim_config["L1I"]["mshr_size"]) + ",64,64,32"
    l1i_capacity = str(
        champsim_config["block_size"]
        * champsim_config["L1I"]["sets"]
        * champsim_config["L1I"]["ways"]
    )
    l1i_ways = champsim_config["L1I"]["ways"]
    res["icache_config"] = l1i_capacity + f",64,{l1i_ways},1,8,4,64,1"

    btb_direct_sets_mapping = {
        "0": 512,
        "1": 1024,
        "2": 2048
    }

    btb_capacity = str(
        btb_direct_sets_mapping[str(btb_direct_predictor_sets)] * 8
    )
    res["BTB_config"] = f"{btb_capacity},8,8,1,1,3"

    if champsim_config["ooo_cpu"][0]["branch_predictor"] == "perceptron":
        res["local_predictor_size"] = "1304,8"
        res["local_predictor_entries"] = "3912"
    elif champsim_config["ooo_cpu"][0]["branch_predictor"] == "hashed_perceptron":
        res["local_predictor_size"] = "192,12"
        res["local_predictor_entries"] = "3712"
    elif champsim_config["ooo_cpu"][0]["branch_predictor"] == "bimodal":
        res["local_predictor_size"] = "16381,2"
        res["local_predictor_entries"] = "16384"
    elif champsim_config["ooo_cpu"][0]["branch_predictor"] == "gshare":
        res["local_predictor_size"] = "14,2"
        res["local_predictor_entries"] = "16384"
    elif champsim_config["ooo_cpu"][0]["branch_predictor"] == "tage_sc_l":
        res["local_predictor_size"] = "2,2"
        res["local_predictor_entries"] = "16384"
        res["global_predictor_entries"] = "16384"
        res["global_predictor_bits"] = "2"
        res["chooser_predictor_entries"] = "16384"
        res["chooser_predictor_bits"] = "2"


    res["target_core_clockrate"] = str(champsim_config["ooo_cpu"][0]["frequency"])

    res["fetch_width"] = str(champsim_config["ooo_cpu"][0]["fetch_width"])
    res["decode_width"] = str(champsim_config["ooo_cpu"][0]["decode_width"])
    res["issue_width"] = str(champsim_config["ooo_cpu"][0]["dispatch_width"])
    res["commit_width"] = str(champsim_config["ooo_cpu"][0]["retire_width"])
    res["instruction_buffer_size"] = str(
        champsim_config["ooo_cpu"][0]["ifetch_buffer_size"]
    )
    res["decoded_stream_buffer_size"] = str(
        champsim_config["ooo_cpu"][0]["decode_buffer_size"]
    )
    res["instruction_window_size"] = str(champsim_config["ooo_cpu"][0]["scheduler_size"])
    res["ROB_size"] = str(champsim_config["ooo_cpu"][0]["rob_size"])

    return res

def xml_reader(data, champsim_terminal_output):
    # Process champsim output
    res = {}
    res["mpki"] = 0 # Default value

    # L3/LLC
    res["L30_read_accesses"] = str(
        data["roi"]["LLC"]["LOAD"]["hit"][0] + data["roi"]["LLC"]["LOAD"]["miss"][0]
    )
    res["L30_read_misses"] = str(data["roi"]["LLC"]["LOAD"]["miss"][0])
    res["L30_write_accesses"] = str(
        data["roi"]["LLC"]["WRITE"]["hit"][0] + data["roi"]["LLC"]["WRITE"]["miss"][0]
    )
    res["L30_write_misses"] = str(data["roi"]["LLC"]["WRITE"]["miss"][0])

    # L2
    res["L20_read_accesses"] = str(
        data["roi"]["cpu0_L2C"]["LOAD"]["hit"][0] + data["roi"]["cpu0_L2C"]["LOAD"]["miss"][0]
    )
    res["L20_read_misses"] = str(data["roi"]["cpu0_L2C"]["LOAD"]["miss"][0])
    res["L20_write_accesses"] = str(
        data["roi"]["cpu0_L2C"]["WRITE"]["hit"][0]
        + data["roi"]["cpu0_L2C"]["WRITE"]["miss"][0]
    )
    res["L20_write_misses"] = str(data["roi"]["cpu0_L2C"]["WRITE"]["miss"][0])

    # L1D
    res["dcache_read_accesses"] = str(
        data["roi"]["cpu0_L1D"]["LOAD"]["hit"][0] + data["roi"]["cpu0_L1D"]["LOAD"]["miss"][0]
    )
    res["dcache_read_misses"] = str(data["roi"]["cpu0_L1D"]["LOAD"]["miss"][0])
    res["dcache_write_accesses"] = str(
        data["roi"]["cpu0_L1D"]["WRITE"]["hit"][0]
        + data["roi"]["cpu0_L1D"]["WRITE"]["miss"][0]
    )
    res["dcache_write_misses"] = str(data["roi"]["cpu0_L1D"]["WRITE"]["miss"][0])

    # L1I
    res["icache_read_accesses"] = str(
        data["roi"]["cpu0_L1I"]["LOAD"]["hit"][0] + data["roi"]["cpu0_L1I"]["LOAD"]["miss"][0]
    )
    res["icache_read_misses"] = str(data["roi"]["cpu0_L1I"]["LOAD"]["miss"][0])

    # dtlb
    res["dtlb_core0_total_accesses"] = str(
        data["roi"]["cpu0_DTLB"]["LOAD"]["hit"][0]
        + data["roi"]["cpu0_DTLB"]["LOAD"]["miss"][0]
    )
    res["dtlb_core0_total_misses"] = str(data["roi"]["cpu0_DTLB"]["LOAD"]["miss"][0])

    # itlb
    res["itlb_core0_total_accesses"] = str(
        data["roi"]["cpu0_ITLB"]["LOAD"]["hit"][0]
        + data["roi"]["cpu0_ITLB"]["LOAD"]["miss"][0]
    )
    res["itlb_core0_total_misses"] = str(data["roi"]["cpu0_ITLB"]["LOAD"]["miss"][0])

    keywords = [
        "CPU0BranchPredictionAccuracy:",
        "CPU0cumulativeIPC",
        "RQROW_BUFFER_HIT:",
        "WQROW_BUFFER_HIT",
        "FULL",
        "MPKI:"
    ]
    all_lines = champsim_terminal_output.splitlines()
    all_lines = [line.replace(" ", "") for line in all_lines]
    br_accuracy = 0
    RQ_ROW_BUFFER_HIT = 0
    RQ_ROW_BUFFER_MISS = 0
    WQ_ROW_BUFFER_HIT = 0
    WQ_ROW_BUFFER_MISS = 0
    FULL = 0
    for each_idx in range(len(all_lines)):
        if keywords[0] in all_lines[each_idx]:
            my_string = all_lines[each_idx]
            result = re.search("CPU0BranchPredictionAccuracy:(.*)MPKI:", my_string)
            if result:
                if isinstance(result.group(1).split("%")[0], str) and "-" in result.group(1).split("%")[0]:
                    # Probably no branch instructions
                    br_accuracy = 100.0
                else:
                    br_accuracy = float(result.group(1).split("%")[0])
            
        if keywords[1] in all_lines[each_idx]:
            my_string = all_lines[each_idx]
            result = re.search("cycles:(.*)", my_string)
            if result:
                result = int(result.group(1))
                res["total_cycles"] = str(result)
        if keywords[2] in all_lines[each_idx]:
            my_string = all_lines[each_idx]
            result = re.search("ROW_BUFFER_HIT:(.*)", my_string)
            if result:
                RQ_ROW_BUFFER_HIT = int(result.group(1))
            if each_idx + 1 < len(all_lines):
                my_string = all_lines[each_idx + 1]
                result = re.search("ROW_BUFFER_MISS:(.*)", my_string)
                if result:
                    RQ_ROW_BUFFER_MISS = int(result.group(1))
        if keywords[3] in all_lines[each_idx]:
            my_string = all_lines[each_idx]
            result = re.search("ROW_BUFFER_HIT:(.*)", my_string)
            if result:
                if result.group(1) != "Channel0":
                    WQ_ROW_BUFFER_HIT = int(result.group(1))
                else:
                    WQ_ROW_BUFFER_HIT = 0
            if each_idx + 1 < len(all_lines):
                 my_string = all_lines[each_idx + 1]
                 result = re.search("ROW_BUFFER_MISS:(.*)", my_string)
                 if result:
                     WQ_ROW_BUFFER_MISS = int(result.group(1))
        if keywords[4] in all_lines[each_idx]:
            my_string = all_lines[each_idx]
            result = re.search("FULL:(.*)", my_string)
            if result:
                FULL = int(result.group(1))
        if keywords[5] in all_lines[each_idx]:
            my_string = all_lines[each_idx]
            result = re.search("MPKI:(.*)AverageROB", my_string)
            if result:
                res["mpki"] = float(result.group(1))
    total_instructions = str(data["roi"]["cores"][0]["instructions"])
    res["total_instructions"] = total_instructions
    res["committed_instructions"] = total_instructions
    res["fp_instructions"] = "0"
    res["committed_fp_instructions"] = "0"

    res["branch_mispredictions"] = (
        int(data["roi"]["cores"][0]["mispredict"]["BRANCH_CONDITIONAL"])
        + int(data["roi"]["cores"][0]["mispredict"]["BRANCH_DIRECT_CALL"])
        + int(data["roi"]["cores"][0]["mispredict"]["BRANCH_INDIRECT"])
        + int(data["roi"]["cores"][0]["mispredict"]["BRANCH_INDIRECT_CALL"])
        + int(data["roi"]["cores"][0]["mispredict"]["BRANCH_RETURN"])
    )
    res["branch_mispredictions"] = str(res["branch_mispredictions"])
    try:
        if br_accuracy == 100:
            res["branch_instructions"] = 0
        else:
            res["branch_instructions"] = (int(res["branch_mispredictions"]) * 100) / (
                100 - float(br_accuracy)
            )
    except:
        print(f"Error in branch accuracy: {res['branch_mispredictions']} / 100 - {br_accuracy}")
        exit(1)
    res["branch_instructions"] = str(int(res["branch_instructions"]))


    res["global_predictor_entries"] = "4096"
    res["global_predictor_bits"] = "2"
    res["chooser_predictor_entries"] = "4096"
    res["chooser_predictor_bits"] = "2"

    res["mc_memory_accesses"] = (
        RQ_ROW_BUFFER_HIT
        + RQ_ROW_BUFFER_MISS
        + WQ_ROW_BUFFER_HIT
        + WQ_ROW_BUFFER_MISS
        + FULL
    )
    res["mc_memory_reads"] = RQ_ROW_BUFFER_HIT + RQ_ROW_BUFFER_MISS
    res["mc_memory_writes"] = WQ_ROW_BUFFER_HIT + WQ_ROW_BUFFER_MISS

    res["mc_memory_accesses"] = str(res["mc_memory_accesses"])
    res["mc_memory_reads"] = str(res["mc_memory_reads"])
    res["mc_memory_writes"] = str(res["mc_memory_writes"])
    return res

def set_xml(res, xml_path, output_file_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    desired_component_id = "system.core0"
    for component in root.findall(".//component[@id='" + desired_component_id + "']"):
        # Iterate through the stats under the component
        for stat in component.findall("./stat"):
            if (
                stat.get("name") == "total_instructions"
                or stat.get("name") == "int_instructions"
            ):
                stat.set("value", res["total_instructions"])
            elif stat.get("name") == "fp_instructions":
                stat.set("value", res["fp_instructions"])
            elif stat.get("name") == "branch_instructions":
                stat.set("value", res["branch_instructions"])
            elif stat.get("name") == "branch_mispredictions":
                stat.set("value", res["branch_mispredictions"])
            elif (
                stat.get("name") == "committed_instructions"
                or stat.get("name") == "committed_int_instructions"
            ):
                stat.set("value", res["committed_instructions"])
            elif stat.get("name") == "committed_fp_instructions":
                stat.set("value", res["committed_fp_instructions"])
            elif stat.get("name") == "load_instructions":
                stat.set("value", "0")
            elif stat.get("name") == "store_instructions":
                stat.set("value", "0")
            elif stat.get("name") == "total_cycles" or stat.get("name") == "busy_cycles":
                stat.set("value", res["total_cycles"])
            elif stat.get("name") == "idle_cycles":
                stat.set("value", "0")
    desired_component_id = "system"
    for component in root.findall(".//component[@id='" + desired_component_id + "']"):
        # Iterate through the stats under the component
        for stat in component.findall("./stat"):
            if stat.get("name") == "total_cycles" or stat.get("name") == "busy_cycles":
                stat.set("value", res["total_cycles"])
            if stat.get("name") == "idle_cycles":
                stat.set("value", "0")
    for param in root.iter("param"):
        if (
            param.get("name") == "target_core_clockrate"
            or param.get("name") == "clock_rate"
        ):
            param.set(
                "value", res["target_core_clockrate"]
            )
        elif param.get("name") == "fetch_width":
            param.set("value", res["fetch_width"])
        elif param.get("name") == "decode_width":
            param.set("value", res["decode_width"])
        elif param.get("name") == "issue_width" or param.get("name") == "peak_issue_width":
            param.set("value", res["issue_width"])
        elif param.get("name") == "commit_width":
            param.set("value", res["commit_width"])
        elif param.get("name") == "instruction_buffer_size":
            param.set("value", res["instruction_buffer_size"])
        elif param.get("name") == "decoded_stream_buffer_size":
            param.set("value", res["decoded_stream_buffer_size"])
        elif (
            param.get("name") == "instruction_window_size"
            or param.get("name") == "fp_instruction_window_size"
        ):
            param.set("value", res["instruction_window_size"])
        elif param.get("name") == "ROB_size":
            param.set("value", res["ROB_size"])
        elif (
            param.get("name") == "total_instructions"
            or param.get("name") == "committed_instructions"
        ):
            param.set("value", res["total_instructions"])
        elif (
            param.get("name") == "int_instructions"
            or param.get("name") == "committed_int_instructions"
        ):
            param.set("value", res["fp_instructions"])
        elif (
            param.get("name") == "fp_instructions"
            or param.get("name") == "committed_fp_instructions"
        ):
            param.set("value", res["branch_instructions"])
        elif param.get("name") == "branch_mispredictions":
            param.set("value", res["branch_mispredictions"])
        elif param.get("name") == "branch_mispredictions":
            param.set("value", res["load_instructions"])
        elif param.get("name") == "store_instructions":
            param.set("value", res["store_instructions"])
        elif param.get("name") == "local_predictor_size":
            param.set("value", res["local_predictor_size"])
        elif param.get("name") == "local_predictor_entries":
            param.set("value", res["local_predictor_entries"])

        elif param.get("name") == "global_predictor_entries":
            param.set("value", res["global_predictor_entries"])
        elif param.get("name") == "global_predictor_bits":
            param.set("value", res["global_predictor_bits"])
        elif param.get("name") == "chooser_predictor_entries":
            param.set("value", res["chooser_predictor_entries"])
        elif param.get("name") == "chooser_predictor_bits":
            param.set("value", res["chooser_predictor_bits"])
        elif param.get("name") == "BTB_config":
            param.set("value", res["BTB_config"])

        desired_component_id = "system.core0.itlb"

        for component in root.findall(".//component[@id='" + desired_component_id + "']"):
            for stat in component.findall("./stat"):
                if stat.get("name") == "total_misses":
                    stat.set("value", res["itlb_core0_total_misses"])
                elif stat.get("name") == "total_accesses":
                    stat.set("value", res["itlb_core0_total_accesses"])

        desired_component_id = "system.core0.dtlb"

        for component in root.findall(".//component[@id='" + desired_component_id + "']"):
            for stat in component.findall("./stat"):
                if stat.get("name") == "total_misses":
                    stat.set("value", res["dtlb_core0_total_misses"])
                elif stat.get("name") == "total_accesses":
                    stat.set("value", res["dtlb_core0_total_accesses"])

        desired_component_id = "system.core0.icache"

        for component in root.findall(".//component[@id='" + desired_component_id + "']"):
            for stat in component.findall("./stat"):
                if stat.get("name") == "read_accesses":
                    stat.set("value", res["icache_read_accesses"])
                elif stat.get("name") == "read_misses":
                    stat.set("value", res["icache_read_misses"])
            for param in component.findall("./param"):
                if param.get("name") == "icache_config":
                    param.set("value", res["icache_config"])
                elif param.get("name") == "icache_buffer_sizes":
                    param.set("value", res["icache_buffer_sizes"])

        desired_component_id = "system.core0.dcache"

        for component in root.findall(".//component[@id='" + desired_component_id + "']"):
            for stat in component.findall("./stat"):
                if stat.get("name") == "read_accesses":
                    stat.set("value", res["dcache_read_accesses"])
                elif stat.get("name") == "read_misses":
                    stat.set("value", res["dcache_read_misses"])
                elif stat.get("name") == "write_accesses":
                    stat.set("value", res["dcache_write_accesses"])
                elif stat.get("name") == "write_misses":
                    stat.set("value", res["dcache_write_misses"])
            for param in component.findall("./param"):
                if param.get("name") == "dcache_config":
                    param.set("value", res["dcache_config"])
                elif param.get("name") == "dcache_buffer_sizes":
                    param.set("value", res["dcache_buffer_sizes"])

        desired_component_id = "system.L20"

        for component in root.findall(".//component[@id='" + desired_component_id + "']"):
            for param in component.findall("./param"):
                if param.get("name") == "L2_config":
                    param.set("value", res["L2_config"])
                elif param.get("name") == "L20_buffer_sizes":
                    param.set("value", res["L20_buffer_sizes"])
            for stat in component.findall("./stat"):
                if stat.get("name") == "read_accesses":
                    stat.set("value", res["L20_read_accesses"])
                elif stat.get("name") == "write_accesses":
                    stat.set("value", res["L20_write_accesses"])
                elif stat.get("name") == "read_misses":
                    stat.set("value", res["L20_read_misses"])
                elif stat.get("name") == "write_misses":
                    stat.set("value", res["L20_write_misses"])

        desired_component_id = "system.L30"

        for component in root.findall(".//component[@id='" + desired_component_id + "']"):
            for param in component.findall("./param"):
                if param.get("name") == "L3_config":
                    param.set("value", res["L3_config"])
                elif param.get("name") == "L30_buffer_sizes":
                    param.set("value", res["L30_buffer_sizes"])
            for stat in component.findall("./stat"):
                if stat.get("name") == "read_accesses":
                    stat.set("value", res["L30_read_accesses"])
                elif stat.get("name") == "write_accesses":
                    stat.set("value", res["L30_write_accesses"])
                elif stat.get("name") == "read_misses":
                    stat.set("value", res["L30_read_misses"])
                elif stat.get("name") == "write_misses":
                    stat.set("value", res["L30_write_misses"])

        desired_component_id = "system.mc"

        for component in root.findall(".//component[@id='" + desired_component_id + "']"):
            for stat in component.findall("./stat"):
                if stat.get("name") == "memory_accesses":
                    stat.set("value", res["mc_memory_accesses"])
                elif stat.get("name") == "memory_reads":
                    stat.set("value", res["mc_memory_reads"])
                elif stat.get("name") == "memory_writes":
                    stat.set("value", res["mc_memory_writes"])

        tree = ET.ElementTree(root)
        tree.write(output_file_path, encoding="utf-8")
