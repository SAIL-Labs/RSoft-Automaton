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
'''
Hacking part to add pathways only
'''
def add_pathway(file_name, json, core_num):
    launch_array = {}
    for launch_key in json:
        launch_array[launch_key] = json[launch_key]

    pathway_text = '''
pathway {n}
        {n}
end pathway'''

    with open(f"{file_name}.ind", "a") as f:
        # if core_num == 1:
        #     max_segments = int(core_num) + 1
        # else:
        #     max_segments = int(core_num) + 2
        # for i in range(1,max_segments):
        pathway_text_block = pathway_text.format(n = core_num)
        f.write(pathway_text_block)
#################################################################################
'''
Hacking part to add monitors only
'''
def add_monitor(file_name, json, segment_id, mon_h = "", mon_w = ""):
    """
    Appends a monitor block for a given segment ID.

    Args:
        file_name (str): base name of the .ind file
        json (dict): launch/monitor parameters
        segment_id (int): segment/pathway index to attach monitor to
        cladd (bool): whether this monitor is for the cladding segment
    """
    # Copy JSON entries to local dict
    launch_array = {k: json[k] for k in json}

    # Pick correct monitor size
    monitor_width = mon_w
    
    monitor_height = mon_h

    monitor_text = f'''
monitor {segment_id}
    pathway = {segment_id}
    monitor_type = {launch_array.get("monitor_type")}
    monitor_tilt = {launch_array.get("launch_tilt")}
    monitor_component = {launch_array.get("comp")}
    monitor_width = {monitor_width}
    monitor_height = {monitor_height}
end monitor
    '''

    with open(f"{file_name}.ind", "a") as f:
        f.write(monitor_text)
#################################################################################
'''
Hacking part to add launch field
'''
def add_launch_field(file_name, json, core_num, how_launch = 1):
    launch_array = {}
    for launch_key in json:
        launch_array[launch_key] = json[launch_key]
    
    launch_text = '''
launch_field {how_launch}
    launch_pathway = {n}
    launch_type = {launch_type}
    launch_mode = {launch_mode}
    launch_mode_radial = {launch_mode_radial}
    launch_align_file = {launch_align_file}
    launch_random_set = {launch_random_set}
    launch_tilt = {launch_tilt}
    launch_normalization = {launch_normalization}
end launch_field
    '''
    with open(f"{file_name}.ind", "a") as f:
        # if core_num == 1:
        #     max_segments = int(core_num) + 1
        # else:
        #     max_segments = int(core_num) + 2
        # for i in range(1,how_launch + 1):
        #     # only add 1 launch field
        # if core_num != 2 :
        #     continue

        text = launch_text.format(
            n=core_num, 
            how_launch = how_launch,
            launch_field_height = launch_array["launch_field_height"],
            launch_field_width = launch_array["launch_field_width"],
            launch_tilt =launch_array["launch_tilt"],
            launch_type=launch_array["launch_type"],
            launch_mode=launch_array["launch_mode"],
            launch_mode_radial=launch_array["launch_mode_radial"],
            launch_align_file=launch_array["launch_align_file"],
            launch_random_set=launch_array["launch_random_set"],
            launch_normalization=launch_array["launch_normalization"])
        f.write(text)
#################################################################################
def AddHack(file_name, json, core_num, central_index):
    launch_array = {k: json[k] for k in json}

    block_text = { # 10.2
        "pathway": '''
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
end monitor
''',
        "launch_field": '''
launch_field {n}
    launch_pathway = {central_index}
    launch_type = {launch_type}
    launch_mode = {launch_mode}
    launch_mode_radial = {launch_mode_radial}
    launch_random_set = {launch_random_set}
end launch_field
'''
    }

    # Open file in append mode
    with open(f"{file_name}.ind", "a") as f:

        # Write all pathways
        for i in range(1, core_num + 2):  # +1 for cladding
            text = block_text["pathway"].format(n=i)
            f.write(text)

        # Write all monitors
        for i in range(1, core_num + 2):
            monitor_width = launch_array["cladd_monitor_width"] if i == 1 else launch_array["core_monitor_width"]
            monitor_height = launch_array["cladd_monitor_height"] if i == 1 else launch_array["core_monitor_height"]
            monitor_type = "MONITOR_WG_POWER" if i == 1 else launch_array["monitor_type"]

            text = block_text["monitor"].format(
                n=i,
                monitor_width=monitor_width,
                monitor_height=monitor_height,
                monitor_type=monitor_type,
                comp=launch_array["comp"],
                launch_tilt=launch_array["launch_tilt"]
            )
            f.write(text)
            # Write only one launch field (for the cladding (MMF case)/core (SMF case))
            text = block_text["launch_field"].format(
                n=1,
                central_index=1,
                launch_type=launch_array["launch_type"],
                launch_tilt=launch_array["launch_tilt"],
                launch_normalization=launch_array["launch_normalization"],
                launch_align_file = launch_array["launch_align_file"],
                launch_mode=launch_array["launch_mode"],
                launch_mode_radial=launch_array["launch_mode_radial"],
                launch_random_set=launch_array["launch_random_set"]
            )
        f.write(text)
#################################################################################
# Calculate the V-number from available parameters
def calc_V(core_diam, n_core, n_cladd, wavelength):
    a = core_diam/2
    NA = np.sqrt(np.abs(n_core**2 - n_cladd**2))
    V = (2 * np.pi * a / wavelength) * NA
    return V, NA

# sample from prior space to ensure parameters that satisfy the V-number condition are passed through
def prior_sampling(param_dict, core_index,background_index, free_space_wavelength):
    while True:
        V, NA = calc_V(param_dict["Corediam"], core_index, background_index, free_space_wavelength)
        if V >= 2.405 or V<= 1.0:
            return -1e6

# filter parameter space to only include priors that result in V < 2.405
def filter_parameter_space_by_v_number(para_space, background_index, wavelength, core_index, samples_per_param,v_max=2.405, v_min=1.0): 
    corediams = np.linspace(*para_space["Corediam"], samples_per_param)
    # core_indices = np.linspace(*para_space["Core_index"], samples_per_param)
    # core_indices = [core_index]
    valid_combinations = [
        d for d in corediams 
        if v_min < calc_V(d, core_index, background_index, wavelength)[0] < v_max
    ]

    if not valid_combinations:
        return {}

    # corediam_vals = zip(*valid_combinations)
    return {
        "Corediam": (min(valid_combinations), max(valid_combinations)),
        # "Length": para_space["Length"],  # Length remains unchanged
        # "Core_index": (min(coreindex_vals), max(coreindex_vals))
    }