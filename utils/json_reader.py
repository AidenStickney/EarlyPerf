from utils.xml_reader import xml_reader, champsim_config_reader, set_xml
import re
import json

def parse_output(data, champsim_terminal_output, act_decoded, idx, config_file, btb_direct_predictor_sets):
    def grab_line_with_word(text, word, after_line=None):
        lines = text.splitlines()
        found = False
        for line in lines:
            if after_line is not None:
                if after_line in line:
                    found = True
                if not found:
                    continue
            if word in line:
                return line
        return None

    def grab_lines_with_word(text, word):
        lines = text.splitlines()
        found_lines = []
        for line in lines:
            if word in line:
                found_lines.append(line)
        return found_lines

    def get_line_number_with_word(text, word):
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if word in line:
                return i
        return None

    results = []
    curr_instr = 0
    curr_cycles = 0
    init_instr = 0
    instructions = []
    cycles = []
    cum_ipc = []
    inst_ipc = []

    max_simulation_index = len(data) - 1

    total_res = []
    prev_res = {}
    champsim_info = champsim_config_reader(config_file, btb_direct_predictor_sets)
    mpki = []
    def get_text_between(text, start, end):
        if end == "":
            pattern = rf"{re.escape(start)}(.*)"
        else:
            pattern = rf"{re.escape(start)}(.*?){re.escape(end)}"
        
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            return "" # Return empty if not found
    for i in range(max_simulation_index + 1):
        if i != max_simulation_index:
            champsim_terminal_output_subset = get_text_between(
                champsim_terminal_output, f"=== Simulation{i} ===", f"=== Simulation{i + 1} ==="
            )
        else:
            # Try end time first
            champsim_terminal_output_subset = get_text_between(
                champsim_terminal_output, f"=== Simulation{i} ===", "End Time"
            )
            # If not, try till end of file
            if not champsim_terminal_output_subset:
                champsim_terminal_output_subset = get_text_between(
                    champsim_terminal_output, f"=== Simulation{i} ===", ""
                )

        spec_data = data[i]
        if spec_data["roi"]["cores"][0]["instructions"] < 100:
            continue
        res = xml_reader(spec_data, champsim_terminal_output_subset)
        mpki.append(res["mpki"])

        cycles = data[i]["roi"]["cores"][0]["cycles"]
        instr = data[i]["roi"]["cores"][0]["instructions"]
        curr_instr += instr
        curr_cycles += cycles
        cum_ipc.append(curr_instr/curr_cycles)
        inst_ipc.append(instr/cycles)

        for key in res.keys():
            res[key] = float(res[key])
        if i != 0 and i != 80:
            for key in res.keys():
                res[key] += prev_res[key]

        total_res.append(res)
        prev_res = res

    res_all = {}
    for res in total_res:
        for key in res.keys():
            value = res[key]
            if key not in res_all:
                res_all[key] = [value]
            else:
                res_all[key].append(value)

    res_all["cumulative_ipc"] = cum_ipc
    res_all["inst_ipc"] = inst_ipc
    res_all["mpki"] = mpki
    
    if isinstance(config_file, dict):
        config = config_file
    else:
        with open(config_file, 'r') as json_file:
            config = json.load(json_file)
            
    champsim_info["L1DPrefetcher"] = config["L1D"]["prefetcher"]
    champsim_info["L1IPrefetcher"] = config["L1I"]["prefetcher"]
    champsim_info["BranchPredictor"] = config["ooo_cpu"][0]["branch_predictor"]
    res_all["champsim_info"] = champsim_info

    return res_all