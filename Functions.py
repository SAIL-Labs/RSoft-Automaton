import numpy as np, pandas as pd
import json, os, csv, ofiber, random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from template import *
#######################################################################################################################################################
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
#######################################################################################################################################################
def insert_after_match(lines, match_string, insert_lines, strip=True, segment_filter=None):
    if isinstance(insert_lines, str):
        insert_lines = [insert_lines]

    modified = []
    in_segment = False
    matches_segment = False

    for line in lines:
        line_to_check = line.strip() if strip else line
        modified.append(line)

        # Detect segment membership by comp_name
        if line_to_check.startswith("comp_name ="):
            if segment_filter is None:
                matches_segment = True
            elif segment_filter in line_to_check:
                matches_segment = True
            else:
                matches_segment = False

        # Insert only if inside the right segment
        if matches_segment and line_to_check.startswith(match_string):
            for new_line in insert_lines:
                modified.append(new_line if new_line.endswith("\n") else new_line + "\n")

    return modified
#######################################################################################################################################################
def has_another_line(file):
    current_pos = file.tell()
    read_line = bool(file.readline())
    file.seek(current_pos)
    return read_line
#######################################################################################################################################################
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
#######################################################################################################################################################
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
#######################################################################################################################################################
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
#######################################################################################################################################################

#######################################################################################################################################################
def AddHack(file_name, json_file, core_num, param_dict):
    launch_array = {k: json_file[k] for k in json_file}
    
    block_text = { 
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
    monitor_mode = {monitor_mode}
end monitor
''',
        "launch_field": '''
launch_field {n}
    launch_pathway = {n}
    launch_type = {launch_type}
    launch_mode = {launch_mode}
    launch_mode_radial = {launch_mode_radial}
    launch_random_set = {launch_random_set}
    launch_normalization = {launch_normalization}
    launch_align_file = {launch_align_file}
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
            # monitor_width = launch_array["cladd_monitor_width"] if i == 1 else launch_array["core_monitor_width"]
            # monitor_height = launch_array["cladd_monitor_height"] if i == 1 else launch_array["core_monitor_height"]
            monitor_type = launch_array["cladding_monitor_type"] if i == 1 else launch_array["monitor_type"]

            text = block_text["monitor"].format(
                n=i,
                # monitor_width=monitor_width,
                # monitor_height=monitor_height,
                monitor_type=monitor_type,
                comp=launch_array["comp"],
                launch_tilt=launch_array["launch_tilt"],
                monitor_mode = launch_array["launch_mode"] if i == (launch_array["core_to_monitor"] + 1) else 0
            )
            f.write(text)
            # Write only one launch field (for the cladding (MMF case)/core (SMF case))
            text = block_text["launch_field"].format(
                n=1,
                launch_type=launch_array["launch_type"],
                launch_tilt=launch_array["launch_tilt"],
                launch_normalization=launch_array["launch_normalization"],
                launch_align_file = launch_array["launch_align_file"],
                launch_mode=launch_array["launch_mode"],
                launch_mode_radial=launch_array["launch_mode_radial"],
                launch_random_set=launch_array["launch_random_set"],
            )
        f.write(text)

    # Open file in read mode
    with open(f"{file_name}.ind", "r") as f:
        lines = f.readlines()
    # Insert delta after core and cladding segment start

    core_name = [f"core_{n}" for n in range(1, core_num+1)]
    for core_key in core_name:
        lines = insert_after_match(lines, "begin.width =", [
            f"\tbegin.delta = {core_params[core_key]['delta']}\n",
            f"\tend.delta = {core_params[core_key]['delta']}\n"
        ], segment_filter=f"{core_key}")

    lines = insert_after_match(lines, "begin.width =", [
        f"\tbegin.delta = {launch_array['cladding_delta']}\n",
        f"\tend.delta = {launch_array['cladding_delta']}\n"
    ], segment_filter="Super Cladding")

    # Build the updated lines
    modified_lines = []
    in_core = False
    
    for line in lines:
        line_strip = line.strip()
        replaced = False

        # Detect if the segment is the core or cladding
        if line_strip.startswith("comp_name = core_0 "):
            in_core = True
            core_index += 1
        elif line_strip.startswith("comp_name = Super Cladding "):  # not a core
            in_core = False
        
        for param, val in param_dict.items():

            if line_strip.startswith(f"{param} ="):
                modified_lines.append(f"{param} = {val:.6f}\n")
                replaced = True
                break

        if not replaced:
            modified_lines.append(line)

    # Write the final .ind file with symbolic delta expression
    with open(f"{file_name}.ind", "w") as out:
        out.writelines(modified_lines)
#######################################################################################################################################################
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
############################################################################################################################################
def log_optimizer_results(x_iters, y_vals, param_batch, result_batch, param_names, iteration_start, batch_size, penalty_batch=None, csv_path="optimizer_results.csv"):
    """
    Save a batch of scikit-optimize parameter evaluations to CSV, and plot the results.

    Parameters:
        param_batch (list of lists): Parameter vectors from skopt.ask()
        result_batch (list of floats): Corresponding throughput (negated loss) results
        param_names (list of str): Names of the parameters in order
        iteration_start (int): Index of the first sample in this batch (zero-based)
        penalty_batch (list of floats, optional): Penalties applied to each evaluation
        csv_path (str): Path to CSV file (default = optimizer_results.csv)
    """

    include_penalty = penalty_batch is not None
    header = ["Iteration"] + param_names + ["Throughput"]
    if include_penalty:
        header.append("penalty")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(header)

        for j, (params, score) in enumerate(zip(param_batch, result_batch)):
            iteration_number = iteration_start + j + 1
            throughput = -score
            row = [iteration_number] + list(params) + [throughput]
            if include_penalty:
                row.append(penalty_batch[j])
            writer.writerow(row)

    # Find best result
    best_idx = np.argmax(y_vals)
    best_params = x_iters[best_idx]
    best_throughput = y_vals[best_idx]
    
    para_tag = "best_params_log.csv"
    with open(para_tag, "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow([iteration_start // batch_size + 1] + list(best_params) + [best_throughput])

def plotting_optimizer_results(df, param_names):
    """
    Plots optimizer results from a DataFrame, assuming columns:
    - 'Iteration'
    - 'Throughput'
    - One column for each parameter in param_names
    """
    x_iters = df[param_names].values.tolist()
    y_vals = df["Throughput"].values
    n_params = len(param_names)

    # Find best result
    best_idx = np.argmax(y_vals)
    best_params = x_iters[best_idx]
    best_throughput = y_vals[best_idx]
    c_num = df["Iteration"].values

    print("Best fitting values:")
    for param_name, val, in zip(param_names, best_params):
        print(f"{param_name}: {val:.3f}")
    print(f"Throughput: {best_throughput:.3f}")
    
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 4.5 + 1.5 * n_params), sharex=False)

    if n_params == 1:
        axes = [axes]

    for k, (param_name, ax) in enumerate(zip(param_names, axes)):
        x_vals = df[param_name].values

        scatter = ax.scatter(x_vals, y_vals, c=c_num, cmap='viridis_r', s=60, edgecolor='k', label="Evaluations")
        ax.scatter(best_params[k], best_throughput, c='red', s=100, label="Best", zorder=3, edgecolor='black')
        
        ax.set_ylabel("Throughput", fontsize=12)
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_title(f"{param_name} vs Throughput", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label("Iteration")

    plt.tight_layout()
    fig.savefig("Optimization_results.png", dpi=300)
    plt.show()
############################################################################################################################################
def build_fibre(circuit, path_num, core_positions, core_names, Taper_length, beginning_dims_list, final_dims_list):
    for j, (x, y) in enumerate(core_positions):
        path_num += 1
        core = circuit.add_segment(
            position=(x, y , 0),
            offset=(x, y, Taper_length),
            dimensions=beginning_dims_list[j],
            dimensions_end=final_dims_list[j]
        )
        core.set_name(core_names[j])
    return path_num

def build_PL(circuit, path_num, core_positions, core_names, taper, Taper_length,
             cladd_beginning_diam, cladd_final_diam,
             core_beginning_dims_list, core_final_dims_list):
    
    cladding = circuit.add_segment(
        position=(0, 0, 0),
        offset=(0, 0, Taper_length),
        dimensions=cladd_beginning_diam,
        dimensions_end=cladd_final_diam
    )
    cladding.set_name("Super Cladding")
    path_num += 1
    
    for j, (x, y) in enumerate(core_positions):
        path_num += 1
        core = circuit.add_segment(
            position=(x / taper, y / taper, 0),
            offset=(x - (x/taper), y - (y/taper), Taper_length),
            dimensions=core_beginning_dims_list[j],
            dimensions_end=core_final_dims_list[j]
        )
        core.set_name(core_names[j])
    return path_num
############################################################################################################################################
def throughput_metric(csv_path, fixed_length, fixed, vars, param_range, mode_selective):
    df = pd.read_csv(csv_path)
    monitor_columns = [col for col in df.columns if col.startswith("Monitor_")]
    throughput = df[monitor_columns[1:]].tail(10).mean().sum()
    
    # this needs to be a separate function
    if "Taper_L" in vars and not fixed_length:
        taper_L = vars["Taper_L"]
        hyper_param = fixed["length_hyperparam"]
        max_L = max(param_range["Taper_L"])
        penalty = (taper_L / max_L)
        adjusted_throughput = throughput - hyper_param * penalty
        return adjusted_throughput
    else:
        return throughput

def mode_selective_metric(csv_path, core_to_monitor, mode_type=""):
    df = pd.read_csv(csv_path)
    monitor_columns = [col for col in df.columns if col.startswith("Monitor_")]

    # Sanity check to prevent choosing invalid cores
    if core_to_monitor < 1 or core_to_monitor > len(monitor_columns):
        raise ValueError("Invalid core_to_monitor index.")

    # Compute average monitor power over final 10 samples
    ms_core = monitor_columns[Simulation_params["core_to_monitor"]]
    avg_throughput = df[ms_core].tail(10).mean()
    non_ms_cores = [col for col in monitor_columns if col != ms_core]

    P_ms = avg_throughput
    P_non_ms = df[non_ms_cores].tail(10).mean().sum()

    # Select metric style
    if mode_type == mode_type:
        # We expect LP01 to go into the mode-selective core only
        return P_ms / (P_non_ms + 1e-12)  # avoid divide-by-zero
    else:
        # For higher-order modes, we want leakage into the MS core to be small
        return P_non_ms / (P_ms + 1e-12) # avoid divide-by-zero

#######################################################################################################################################################
def overwrite_template_val():
    with open("launch_config.json", "r") as launch_config:
        simulation_val = json.load(launch_config)

    sim_keys = [keys for keys,_ in Simulation_params.items()] 
    core_keys = [key for key,_ in core_params.items()]

    for key, val in simulation_val.items():
        # replace simulation parameters
        if key in sim_keys:
            Simulation_params[key] = val
        # replace mode-selective core parameters
        if key in core_keys:
            core_params[key] = val
#######################################################################################################################################################
def diameter_limits(mode_num_low, mode_num_high, λ_low, λ_high, background_index, cladding_delta, core_num):
    # r_core = np.linspace(r_low, r_high)  # in µm
    λ = np.linspace(λ_low, λ_high) # in µm
    mode_num_range = np.linspace(mode_num_low, mode_num_high)

    # Meshgrid
    num, wave = np.meshgrid(mode_num_range, λ, indexing='ij') 

    # Material parameters
    n_clad = background_index
    n_core = background_index + cladding_delta # since we want a multimode diameter to support 7 modes
    numerical_ap = ofiber.numerical_aperture(n_core, n_clad)
    # mode_number = core_num # want to match the number of modes present to the number of cores

    # Compute V-number and number of modes
    v_num = np.sqrt(4 * mode_num_range)
    diameter = (v_num * wave) / (np.pi * numerical_ap)
    # V = 2 * np.pi / (LAMBDA * R *NA)
    # num_modes = V**2 / 4

    # num_modes = np.round(num_modes).astype(int)

    # Plot
    plt.figure(figsize=(8, 5))
    c = plt.pcolormesh(wave, num, diameter, cmap='gist_ncar')
    plt.colorbar(c, label="Estimated Mode Diameter")
    # plt.contour(wave, num, diameter, levels=[mode_number], colors='k', linewidths=1.5, linestyles='--')
    # contour_proxy = mlines.Line2D([], [], color='k', linestyle='--',label=f'Core to support {mode_number} modes')

    # plt.legend(handles=[contour_proxy], loc='upper right')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Number of Modes")
    plt.title(f"MutliMode Core Radius to Support Modes")

    plt.tight_layout()
    plt.savefig(f"MutliMode Core Radius to Support Modes.png", dpi=300)

    # Binary mask where number of modes is close to mode_number (within a tolerance)
    # mask = np.isclose(num_modes, mode_number, atol=0.1)

    # # Get all core radii where this condition is true
    # radii_matching_modes = R[mask]
    # min_radius = np.min(radii_matching_modes)
    # max_radius = np.max(radii_matching_modes)
    return #2*min_radius, 2*max_radius