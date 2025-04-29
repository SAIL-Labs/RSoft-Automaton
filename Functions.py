import numpy as np

# Function to extract parameters from the .ind file
def Extract_params(param=""):
    with open("MCF_Test.ind", "r") as r:
        for line in r:
            if param in line:
                key, value = line.split("=")
                key = key.strip()
                val = float(value.strip())
                break # <-- stop reading the file once found
    return val
#################################################################################
def insert_after_match(lines, match_string, insert_lines, strip=True):
    if isinstance(insert_lines, str):
        insert_lines = [insert_lines]

    modified = []

    for line in lines:
        line_to_check = line.strip() if strip else line
        modified.append(line)
        if line_to_check.startswith(match_string):
            for new_line in insert_lines:
                modified.append(new_line if new_line.endswith("\n") else new_line + "\n")

    return modified
#################################################################################
def has_another_line(file):
    current_pos = file.tell()
    read_line = bool(file.readline())
    file.seek(current_pos)
    return read_line
#################################################################################
def AddHack(file_name, json, core_num, background_index):
    launch_array = {}
    for launch_key in json:
        launch_array[launch_key] = json[launch_key]
    
    block_text = { 
    "pathway":'''
    pathway {n}
        {n}
    end pathway
    ''',
    "monitor": '''
    monitor {n}
        pathway = {n}
        monitor_type = {monitor_type}
        monitor_tilt = {launch_tilt}
        monitor_component = {comp}
        monitor_background_index = {background_index}
        monitor_width = {monitor_width}
        monitor_height = {monitor_height} 
    end monitor
    ''',
    "launch_field":'''
    launch_field {n}
        launch_pathway = {n}
        launch_type = {launch_type}
        launch_mode = {launch_mode}
        launch_mode_radial = {launch_mode_radial}
        launch_align_file = {launch_align_file}
        launch_random_set = {launch_random_set}
        launch_tilt = {launch_tilt}
        launch_normalization = {launch_normalization}
        launch_height = {monitor_height}
        launch_width = {monitor_width}
    end launch_field
    '''
    }

    # appends the above text to the .ind file
    with open(f"{file_name}.ind", "a") as f:
        for block_type in ["pathway","monitor","launch_field"]:
            if core_num == 1:
                max_segments = int(core_num) + 1
            else:
                max_segments = int(core_num) + 2
            for i in range(1,max_segments):
                # only add 1 launch field
                if block_type == "launch_field" and i != 1:
                    continue

                text = block_text[block_type].format(
                    n=i, 
                    monitor_height = launch_array["monitor_height"],
                    monitor_width = launch_array["monitor_width"],
                    background_index = background_index,
                    monitor_type=launch_array["monitor_type"],
                    comp=launch_array["comp"],
                    launch_tilt =launch_array["launch_tilt"],
                    launch_type=launch_array["launch_type"],
                    launch_mode=launch_array["launch_mode"],
                    launch_mode_radial=launch_array["launch_mode_radial"],
                    launch_align_file=launch_array["launch_align_file"],
                    launch_random_set=launch_array["launch_random_set"],
                    launch_normalization=launch_array["launch_normalization"])
                f.write(text)
#################################################################################
# Calculate the V-number from available parameters
def calc_V(core_diam, n_core, n_cladd, wavelength):
    a = core_diam/2
    NA = np.sqrt(np.abs(n_core**2 - n_cladd**2))
    V = (2 * np.pi * a / wavelength) * NA
    return V, NA

# sample from prior space to ensure parameters that satisfy the V-number condition are passed through
def prior_sampling(param_dict, background_index, free_space_wavelength):
    while True:
        V = calc_V(param_dict["Corediam"], param_dict["Core_index"], background_index, free_space_wavelength)
        if V >= 2.405:
            return -1e6

# filter parameter space to only include priors that result in V < 2.405
def filter_parameter_space_by_v_number(para_space, background_index, wavelength, v_max=2.405, v_min=0.0, samples_per_param=30): 
    corediams = np.linspace(*para_space["Corediam"], samples_per_param)
    core_indices = np.linspace(*para_space["Core_index"], samples_per_param)

    valid_combinations = [
        (d, n) for d in corediams for n in core_indices
        if v_min < calc_V(d, n, background_index, wavelength) < v_max
    ]

    if not valid_combinations:
        return {}

    corediam_vals, coreindex_vals = zip(*valid_combinations)
    return {
        "Corediam": (min(corediam_vals), max(corediam_vals)),
        "Length": para_space["Length"],  # remains unchanged
        "Core_index": (min(coreindex_vals), max(coreindex_vals))
    }