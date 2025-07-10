import numpy as np, pandas as pd, math
import json, os, csv, ofiber, random
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from template import *
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib import colors
import glob
import ehtplot.color
import cmocean
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
def create_folders(folder_name):
    '''
    Creates folder to be placed within the 'Results' folder on the desktop.

    Arguments:
        - folder_name: string entry that will become the name of the folder
    
    Returns:
        - pathway to results folder
    '''

    user_home = os.path.expanduser("~")
    desktop_path = os.path.join(user_home, "Desktop")
    results_root = os.path.join(desktop_path, "Results")
    results_folder = os.path.join(results_root, folder_name)
    os.makedirs(results_folder, exist_ok=True)
    return results_folder
#######################################################################################################################################################
def AddHack(file_name, json_file, core_num, param_dict, mon_type = ""):
    '''
    Hacking function to add text that will import segments to RSoft that the Python API does not currently handle.

    file_name: name tag for the ind file to be hacked
    json_file: contains a dictionary of parameters to be used by the launch field and pathway monitors
    core_num: number of cores in ind file
    param_dict: json file contain the names and values of all the parameters to be modified during simulations
    mon_type: specifies the type of monitor to use. 
        "pathway_mon" records the throughput at each iteration making the simulation time scale with Z and grid spacing,
        "port_mon" (default) records only the throughput at the end of the fibre, or the position at which the monitor is placed.
    '''
    launch_array = {k: json_file[k] for k in json_file}
    if mon_type == "pathway_mon":
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
    monitor_normalization = {monitor_normalization}
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
    launch_phase = {launch_phase}
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
                monitor_type = launch_array["cladding_monitor_type"] if i == 1 else launch_array["monitor_type"]

                text = block_text["monitor"].format(
                    n=i,
                    # monitor_width=monitor_width,
                    # monitor_height=monitor_height,
                    monitor_type=monitor_type,
                    comp=launch_array["comp"],
                    launch_tilt=launch_array["launch_tilt"],
                    monitor_mode = 0, #launch_array["launch_mode"] if i == (launch_array["core_to_monitor"] + 1) else 0
                    monitor_normalization = launch_array["monitor_normalization"]
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
                    launch_phase = launch_array["launch_phase"]
                )
            f.write(text)
    elif mon_type == "port_mon":
        block_text = { 
#         "monitor": '''
# time_monitor {n}
#     profile_type = PROF_INACTIVE
#     color = 2
#     type = TIMEMON_EXTENDED
#     timeaverage = 2
#     monitoroutputmask = 1024
# 	monitoroutputformat = OUTPUT_AMP_PHASE
# ''',
        "pathway": '''
pathway {n}
    {n}
end pathway
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
    launch_phase = {launch_phase}
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
            # for i in range(1, core_num + 2):
                # monitor_width = launch_array["cladd_monitor_width"] if i == 1 else launch_array["core_monitor_width"]
                # monitor_height = launch_array["cladd_monitor_height"] if i == 1 else launch_array["core_monitor_height"]
                # monitor_type = launch_array["cladding_monitor_type"] if i == 1 else launch_array["monitor_type"]

                # text = block_text["monitor"].format(
                #     n=i,
                #     # monitor_width=monitor_width,
                #     # monitor_height=monitor_height,
                #     monitor_type=monitor_type,
                #     comp=launch_array["comp"],
                #     launch_tilt=launch_array["launch_tilt"],
                #     monitor_mode = 0, #launch_array["launch_mode"] if i == (launch_array["core_to_monitor"] + 1) else 0
                #     monitor_normalization = launch_array["monitor_normalization"]
                # )
                # f.write(text)
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
                launch_phase = launch_array["launch_phase"]
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
    in_segment_header = False
    current_segment_is_super_cladding = False
    core_index = 0
    
    for line in lines:
        line_strip = line.strip()
        replaced = False

        if line_strip.startswith("comp_name = Super Cladding"):
            current_segment_is_super_cladding = True
            in_segment_header = True
            modified_lines.append(line)
            continue
        elif line_strip.startswith("comp_name = core_"):
            current_segment_is_super_cladding = False
            in_segment_header = True
            modified_lines.append(line)
            continue
        
        if line_strip.startswith("extended ="):
            if not current_segment_is_super_cladding:
                continue

        if in_segment_header and not line_strip.startswith("begin."):
            modified_lines.append("\twidth_taper = TAPER_LINEAR\n")
            modified_lines.append("\theight_taper = TAPER_LINEAR\n")
            modified_lines.append("\tposition_taper = TAPER_LINEAR\n")
            modified_lines.append("\tposition_y_taper = TAPER_LINEAR\n")
            in_segment_header = False

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

    # hack in the port monitor stuff to monitor femSIM files rather than the launch field
    if mon_type == "port_mon":
        final_lines = []
        in_time_monitor = False
        inserted = False

        for line in modified_lines:
            final_lines.append(line)
            line_strip = line.strip()

            if line_strip.startswith("time_monitor"):
                in_time_monitor = True
                inserted = False  # reset insertion flag for each time_monitor

            if in_time_monitor and line_strip.startswith("monitoroutputmask") and not inserted:
                final_lines.append("\toverlap_type = 1\n")
                final_lines.append("\tmonitor_file = LP01_19cPL.m00\n")
                inserted = True

            if in_time_monitor and line_strip.startswith("end monitor"):
                in_time_monitor = False

        with open(f"{file_name}.ind", "w") as out:
            out.writelines(final_lines)
        
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
def log_optimizer_results(x_iters, y_vals, param_batch, result_batch, param_names,
                          iteration_start, batch_size,
                          penalty_batch=None, transfer_vector_batch=None,
                          csv_path="optimizer_results.csv"):
    """
    Save a batch of scikit-optimize parameter evaluations to CSV, and plot the results.

    Parameters:
        param_batch (list of lists): Parameter vectors from skopt.ask()
        result_batch (list of floats): Corresponding throughput (negated loss) results
        param_names (list of str): Names of the parameters in order
        iteration_start (int): Index of the first sample in this batch (zero-based)
        penalty_batch (list of floats, optional): Penalties applied to each evaluation
        transfer_vector_batch (list of np.array, optional): Transfer vectors to log
        csv_path (str): Path to CSV file
    """

    include_penalty = penalty_batch is not None
    include_tf = transfer_vector_batch is not None and transfer_vector_batch[0] is not None

    # Setup dynamic header
    header = ["Iteration"] + param_names + ["Throughput"]
    if include_penalty:
        header.append("penalty")
    if include_tf:
        tf_len = len(transfer_vector_batch[0])
        header += [f"TF_{k+1}" for k in range(tf_len)]

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
            if include_tf:
                row.extend(transfer_vector_batch[j])
            writer.writerow(row)

    # Log best point
    best_idx = np.argmax(y_vals)
    best_params = x_iters[best_idx]
    best_throughput = y_vals[best_idx]
    best_tf = transfer_vector_batch[best_idx] if include_tf else []

    para_tag = "best_params_log.csv"
    with open(para_tag, "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["Iteration"] + param_names + ["Throughput"] +
                        ([f"TF_{i+1}" for i in range(len(best_tf))] if include_tf else []))
        writer.writerow([iteration_start // batch_size + 1] + list(best_params) + [best_throughput] +
                        (list(best_tf) if include_tf else []))

def plotting_optimizer_results(df, param_names, tf = None, plot = True, csv_path = ""):
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
    if tf is not None:
        best_transfer_vector = tf[best_idx]
        print("Best transfer vector:", best_transfer_vector)
    c_num = df["Iteration"].values

    if plot:
        print("Best fitting values:")
        for param_name, val, in zip(param_names, best_params):
            print(f"{param_name}: {val:.3f}")
        print(f"Throughput: {best_throughput:.3f}")

        interim_data = pd.read_csv(csv_path)
        interim_x_iters = interim_data[param_names].values.tolist()
        interim_y_vals = interim_data["Throughput"].values
        interim_idx = np.argmax(interim_y_vals)
        interim_c_num = interim_data["Iteration"].values

        interim_best_params = interim_x_iters[interim_idx]
        interim_best_throughput = interim_y_vals[interim_idx] 

        fig, axes = plt.subplots(n_params, 1, figsize=(8, 4.5 + 1.5 * n_params), sharex=False)

        if n_params == 1:
            axes = [axes]

        for k, (param_name, ax) in enumerate(zip(param_names, axes)):
            x_vals = interim_data[param_name].values

            scatter = ax.scatter(x_vals, interim_y_vals, c=interim_c_num, cmap='viridis_r', s=60, edgecolor='k', label="Evaluations")
            ax.scatter(interim_best_params[k], interim_best_throughput, c='red', s=100, label="Best", zorder=3, edgecolor='black')
            
            ax.set_ylabel("Throughput", fontsize=12)
            if param_name == "core_diam":
                ax.set_xlabel(param_name + r" ($\mu$m)", fontsize = 12)
            elif param_name == "core_delta":
                ax.set_xlabel(param_name + r" ($n_{\mathrm{eff}}$)", fontsize = 12)
            elif param_name == "taper":
                ax.set_xlabel(param_name + " ratio (MCF Diam/ MMF Diam)", fontsize = 12)
            elif param_name == "Taper_L":
                ax.set_xlabel(param_name + r" ($\mu$m)", fontsize = 12)
            else:
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

    # Store segments and monitors for attachment
    core_segments = []
    port_monitors = []

    for j, (x, y) in enumerate(core_positions):
        path_num += 1
        core = circuit.add_segment(
            position=(x / taper, y / taper, 0),
            offset=(x, y, Taper_length),
            # offset=(x - (x / taper), y - (y / taper), Taper_length),
            dimensions=core_beginning_dims_list[j],
            dimensions_end=core_final_dims_list[j]
        )
        core.set_name(core_names[j])
        # circuit.attach(core, cladding, 0, 0, 0)
        core_segments.append(core)

        # if Launch_params["mon_type"] == "port_mon":
        #     port = circuit.add_portmonitor(position=(x, y, Taper_length))
        #     port_monitors.append(port)
        #     circuit.attach(port, core, 1, 1, 1) 

    if Launch_params["mon_type"] == "port_mon":
        for j, (x, y) in enumerate(core_positions):
            # Place monitor at the end of the segment, note that these are not offset from anything and so need the extra distance to line up with the segments
            port = circuit.add_portmonitor(position=(x + (x / taper), y + (y / taper), Taper_length), dimensions = (6.5,6.5))
            port_monitors.append(port)

    # Attach port monitors to core segments
    if Launch_params["mon_type"] == "port_mon":
        for core_seg, port_mon in zip(core_segments, port_monitors):
            # Attach monitor to the *output end* of the segment
            circuit.attach(port_mon, core_seg, 1, 1, 1) 
    return path_num
############################################################################################################################################
def throughput_metric(csv_path, fixed_length, fixed, vars, param_range, mode_selective):
    df = pd.read_csv(csv_path)
    monitor_columns = [col for col in df.columns if col.startswith("Monitor_")]
    throughput = df[monitor_columns[1:]].tail(10).mean().sum()
    
    if "Taper_L" in vars and not fixed_length:
        taper_L = vars["Taper_L"]
        hyper_param = fixed["length_hyperparam"]
        max_L = max(param_range["Taper_L"])
        penalty = (taper_L / max_L)
        adjusted_throughput = throughput - hyper_param * penalty
        return adjusted_throughput
    else:
        return throughput
    
def transfer_matrix_component(csv_path, row):
    df = pd.read_csv(csv_path)
    transfer_vector = []

    monitor_columns = [col for col in df.columns if col.startswith("Monitor_")]        
    ms_col = f"Monitor_{Simulation_params['core_to_monitor']}"
    
    if Launch_params["mon_type"] == "pathway_mon":
        for cl in monitor_columns:
            transfer_vector.append(df[cl].tail(10).mean())
        throughput = df[ms_col].tail(10).mean()
        return np.array(transfer_vector), throughput
    
    elif Launch_params["mon_type"] == "port_mon":
        throughput = df[ms_col].tail(10).mean()
        return row, throughput

            
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

def read_port_mon_file(filepath = ""):
    dat = pd.read_csv(filepath, skiprows = 3, sep=r'\s+', header = None)
    all_vals = dat.values.flatten()
    filtered = all_vals[ all_vals < 1]
    return filtered
#######################################################################################################################################################
def overwrite_template_val():
    with open("launch_config.json", "r") as launch_config:
        simulation_val = json.load(launch_config)
    for k,_ in simulation_val.items():
        if k in fixed_params.keys():
            raise Warning(f"Cannot change {k} using simulation_val. Change directly within template.py instead")

    sim_keys = [keys for keys,_ in Simulation_params.items()] 
    core_keys = [key for key,_ in core_params.items()]

    for key, val in simulation_val.items():
        # replace simulation parameters
        if key in sim_keys:
            Simulation_params[key] = val
        # replace mode-selective core parameters
        if key in core_keys:
            core_params[key] = val
    
    # generate core property dictionaries
    core_params.clear()
    for i in range(1, Simulation_params["core_num"] + 1):
        core_key = f"core_{i}"
        if core_key in simulation_val:
            core_params[core_key] = simulation_val[core_key]
        elif "core_diam" in fixed_params and "core_delta" in fixed_params:
            core_params[core_key] = {
                "core_diam": fixed_params["core_diam"],
                "delta": fixed_params["core_delta"]
            }
        elif "core_diam" in variable_params and "core_delta" in variable_params:
            core_params[core_key] = {
                "core_diam": variable_params["core_diam"],
                "delta": variable_params["core_delta"]
            }
#######################################################################################################################################################
def plot_lp_modes(V):
    r_over_a = np.linspace(0, 1.5, 50) # changes the position of the maxima points
    phi = np.linspace(0, 2 * np.pi, 40) 
    clevs = np.linspace(-4, 4, 9)
    
    fig, axs = plt.subplots(3, 3, figsize=(8, 8), subplot_kw={'polar': True})  # Use subplot_kw to specify polar

    axs = axs.flatten()  # Flatten the array to simplify index access

    for idx, ax in enumerate(axs):
        ell = idx // 3
        em = idx % 3 + 1

        b = ofiber.LP_mode_value(V, ell, em)
        if b is None:
            ax.set_title(r"LP$_{%d%d}$ Field" % (ell, em))
            ax.axis('off')  # Turn off axis if no mode exists
            ax.annotate('No such mode', xy=(0.5, 0.5), ha='center', va='center')  # Use transAxes for positioning
            continue

        r_field = ofiber.LP_radial_field(V, b, ell, r_over_a)
        phi_field = np.cos(ell * phi)
        R, PHI = np.meshgrid(r_over_a, phi)
        R_FIELD, PHI_FIELD = np.meshgrid(phi_field, r_field)
        Z = R_FIELD * PHI_FIELD

        cax = ax.contourf(phi, r_over_a, Z, levels=clevs)
        ax.set_xticklabels([])  # Remove x tick labels
        ax.set_yticklabels([])  # Remove y tick labels
        ax.set_title(r"LP$_{%d%d}$ Field" % (ell, em))
        ax.grid(False)

        fig.colorbar(cax, ax=ax)

    plt.tight_layout()  # Adjust layout
    plt.savefig(f"Plotted_LP_Modes_for_V_{V:.3f}.png", dpi = 300)

def plot_available_modes(diam, wave, ell_num, NA):
    for d in diam:
        r = d / 2
        V = ofiber.V_parameter(r, NA, wave)

        m0_list = []
        m1_list = []
        ell_list = []

        # Number of mode orders you want to plot
        ell_range = range(ell_num)  
        n_plots = len(ell_range)

        # Determine plot size 
        n_cols = 2
        n_rows = math.ceil(n_plots / n_cols)

        plt.figure(figsize=(8,5))

        ell_list = []
        m0_list = []
        m1_list = []

        for i, ell in enumerate(ell_range):

            plt.subplot(n_rows, n_cols, i + 1)
            
            aplt = ofiber.plot_LP_modes(V, ell)
            b_vals = ofiber.LP_mode_values(V, ell)

            ell_list.append(ell)
            m0_list.append(b_vals[0] if len(b_vals) > 0 else np.nan)
            m1_list.append(b_vals[1] if len(b_vals) > 1 else np.nan)
        
        # Construct DataFrame
        title = f"Propagation constants (d = {2*r:.3f} µm, λ = {wave:.3f} µm)"
        columns = pd.MultiIndex.from_product([[title], ["l", "m=1", "m=2"]])
        data = list(zip(ell_list, m0_list, m1_list))
        b_df = pd.DataFrame(data, columns=columns)
        
        plt.suptitle(title)
        print(b_df.to_string(index=False))
        plt.tight_layout()
        plt.savefig(f"Available_LP_Modes_for_V_{V:.3f}.png", dpi = 300)
        plot_lp_modes(V)

def print_paras(radii, wavelengths, mode_count, mode_desired, l_modes_to_consider, NA):
    rad_range = []
    for i, r in enumerate(radii):
        for j, wl in enumerate(wavelengths):
            if mode_count[i, j] == mode_desired:
                rad_range.append((r, wl))

    df_range = pd.DataFrame(rad_range, columns = ["Core Radius (µm)", "Wavelength (µm)"])
    df_filtered = df_range[(df_range["Wavelength (µm)"] >= 1.55) & (df_range["Wavelength (µm)"] < 1.56)]

    min_diam = 2 * min(df_filtered["Core Radius (µm)"])
    max_diam = 2 * max(df_filtered["Core Radius (µm)"])
    diam = [min_diam, max_diam]

    taper_max = fixed_params["MCFCladd"] / min_diam
    taper_min = fixed_params["MCFCladd"] / max_diam

    wave = df_filtered["Wavelength (µm)"].iloc[np.argmin(df_filtered["Core Radius (µm)"])]                 
    plot_available_modes(diam, wave, l_modes_to_consider, NA)

    return df_filtered, taper_min, taper_max

LP_mode_dict = {
    "LP01": 1,
    "LP11": 2,
    "LP21": 2,
    "LP02": 1,
    "LP31": 2,
    "LP12": 2,
    "LP41": 2,
    "LP22": 2,
    "LP03": 1,
    "LP51": 2,
    "LP32": 2,
    "LP13": 2,
    "LP61": 2,
    "LP42": 2,
    "LP23": 2,
    "LP04": 1,
    "LP71": 2
}

def mode_wanted_considering_mode_orientations(LP_mode_dict, mode_desired):
    '''
    LP_mode_dict: dictionary where values indicate how many degenerate orientations a mode has
    mode_desired: total mode index counting orientations 
    
    Returns:
        index (1-based) of the LP mode group containing the desired mode
    '''
    mode_number = 0
    for i, (_, val) in enumerate(LP_mode_dict.items()):
        mode_number += val
        if mode_desired <= mode_number:
            return i + 1 

    raise ValueError(f"Desired mode {mode_desired} exceeds total number of available mode orientations ({mode_number}).")

def extract_portmon_amp_phase(tf_list, grid_size_range = None):
    """
    Function used to comb through the complete list of transfer vectors from BeamPROP to extract the amplitude and phase values recorded by each 
    port monitor.

    Arguments:
        - tf_list: list of transfer vectors from RSoft
        - grid_size_range: range of grid sizes to test RSoft simulations on
    Returns:
        - arrays for the amplitude, phase and list of transfer vectors, as well as grid sizes if specified.
    """
    tf_result = []
    amp = []
    phase = []

    for tf in tf_list:
        arrs = np.array(tf).flatten()
        tf_result.append(arrs)
        amp.append(arrs[1::2])
        phase.append(np.deg2rad(arrs[2::2])) 

    if grid_size_range is not None:
        grid_size = list(grid_size_range)
        return amp, phase, grid_size, tf_result
    else:
        return amp, phase, tf_result

def reorder_tf_vectors(tf_vector, simulation_val):
    """
    Reorder tf_vector index to match central core being #1, increasing in an anticlockwise fashion
    
    Arguments:
        - tf_vector: list of tf_vectors resulting from BeamPROP
        - simulaiton_val: disctionary of values used to initialise RSoft
    
    Returns:
        - reordered list of transfer vectors
    """
    tf_matrix = []
    labels = []
    core_num = simulation_val.get("core_num", Simulation_params["core_num"])
    geo = simulation_val.get("grid_type", Simulation_params["grid_type"])

    for label, vec in tf_vector:
        labels.append(label)
        
        if geo == "Hex":
            if core_num == 19:
                reorder_indices = [9, 10, 14, 13, 8, 4, 5, 11, 15, 18, 17, 16, 12, 7, 3, 0, 1, 2, 6]
            elif core_num == 7:
                reorder_indices = [3, 4, 6, 5, 2, 0, 1]
        vec = np.array(vec)[reorder_indices]

        tf_matrix.append(vec)
    return labels, tf_matrix

def phase_to_pixel(phase_val, phase_min, phase_max, resolution):
    # Maps phase_val to a pixel index for a colorbar image
    return int(round((phase_val - phase_min) / (phase_max - phase_min) * (resolution - 1)))

def assign_17modes_to_tflist(tf_list):
    tf_vectors_phase = [
    ("LP01", tf_list[0]),
    ("LP02", tf_list[5]),
    ("LP03", tf_list[14]),
    ("LP11a", tf_list[1]),
    ("LP11b", tf_list[2]),
    ("LP12a", tf_list[8]),
    ("LP12b", tf_list[9]),
    ("LP21a", tf_list[3]),
    ("LP21b", tf_list[4]),
    ("LP22a", tf_list[12]),
    ("LP22b", tf_list[13]),
    ("LP31a", tf_list[6]),
    ("LP31b", tf_list[7]),
    ("LP41a", tf_list[10]),
    ("LP41b", tf_list[11]),
    ("LP51a", tf_list[15]),
    ("LP51b", tf_list[16])
    ]
    return tf_vectors_phase

def plot_tf_matrix(tf_vectors, simulation_val, matrix_type="", ax=None, cbar=True, reorder = False, phase = False):
    '''
    Plot the transfer matrix for a given number of cores in some geometry AFTER running RSoftSimulation.py

    Parameters:
        tf_vectors: array of transfer vectors and their labels, organised as (label, vector), produced by RSoftSimulation.py. Each vector is a 1D array
        simulation_val: dictionary of values set to overwrite preset definitions in RSoftSimulation.py. Must contain core_num and optionally grid_type
    Returns:
        Plot of the transfer matrix
    '''
    core_num = simulation_val["core_num"]
    geo = simulation_val.get("grid_type", Simulation_params["grid_type"]) 
    labels = []
    tf_matrix = []

    for label, vec in tf_vectors:
        labels.append(label)
        
        if reorder:
            """
            Reorder index to match central core being #1, increasing in an anticlockwise fashion
            """
            if core_num == 19:
                reorder_indices = [9, 10, 14, 13, 8, 4, 5, 11, 15, 18, 17, 16, 12, 7, 3, 0, 1, 2, 6]
            elif core_num == 7:
                reorder_indices = [3, 4, 6, 5, 2, 0, 1]
            vec = np.array(vec)[reorder_indices]

        tf_matrix.append(vec)

    tf_matrix = np.array(tf_matrix)  # ensure 2D shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if phase:
        im = ax.imshow(tf_matrix.T, cmap = 'twilight_shifted')
    else:
        im = ax.imshow(tf_matrix.T, cmap='viridis')
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)  
        fig = ax.get_figure()
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(f"{matrix_type}")

    ax.set_ylabel("Core No.")
    ax.set_xlabel("Excited Mode")
    ax.set_yticks(ticks=np.arange(core_num), labels=np.arange(1, core_num + 1))
    ax.set_xticks(ticks=np.arange(len(tf_vectors)), labels=labels, rotation=90)
    ax.set_title(f"{matrix_type}")
    ax.tick_params(axis='both', which='major', labelsize=14)

    return im

def plot_combined_tf_matrix(amp, phase, core_num, phase_max = 2*np.pi, amp_max = 1.0, dir = "", name = ""):
    """
    Function used to combine both amplitude and phase transfer matrices into one joined matrix.

    Arguments:
        - amp: numpy array containing amplitude values
        - phase: numpy array containing phase values
        - core_num: number of cores specified in either simulation_val or Simulation_Params
        - dir: string pointing to the save directory
        - name: name of the image to save
    
    Return:
        - Transfer matrix containing the phase and amplitude for each core and mode
    """
    resolution = 500
    phase_min = 0
    phase_max = phase_max
    amp_min = 0
    amp_max = amp_max

    amp_matrix = np.vstack(amp)
    phase_matrix = np.vstack(phase)

    comp_matrix = amp_matrix * np.exp(1j * phase_matrix)
    comp_tf_vector = assign_17modes_to_tflist(comp_matrix)

    label, tf_matrix = reorder_tf_vectors(comp_tf_vector)

    amp_phase_img = np.transpose(apply_complex_map(tf_matrix, cmocean.cm.phase), (1, 0, 2))
    amp_phase_colorbar = generate_complex_colorbar(resolution = resolution)

    fig, ax = plt.subplots(figsize=(14,8))
    im = ax.imshow(amp_phase_img, aspect='auto', origin='lower')
    plt.gca().invert_yaxis()

    # Set mode and core labels
    ax.set_yticks(np.arange(comp_matrix.shape[1]))
    ax.set_xticks(np.arange(comp_matrix.shape[0]))

    ax.set_ylabel("Core No.", fontsize = 14, labelpad = 10)
    ax.set_xlabel("Excited Mode", fontsize = 14, labelpad = 10)
    ax.set_yticks(ticks=np.arange(core_num), labels=np.arange(1, core_num + 1))
    ax.set_xticks(ticks=np.arange(len(amp)), labels=(labels for labels, _ in comp_tf_vector), rotation=90)

    ax.set_title("Complex Transfer Matrix (Amplitude+Phase)", fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    cb_ax = fig.add_axes([1, 0.15, 0.03, 0.8])  
    cb_ax.imshow(amp_phase_colorbar, aspect='auto', origin='lower')

    phase_tick_vals = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    phase_tick_labels = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    amp_tick_vals = [0, amp_max]
    if amp_max < 0.5:
        amp_tick_labels = ["0", "0.5"]
    elif amp_max >= 0.5:
        amp_tick_labels = ["0", "1"]

    yticks = [phase_to_pixel(val, phase_min, phase_max, resolution) for val in phase_tick_vals]
    xticks = [phase_to_pixel(val, amp_min, amp_max, resolution) for val in amp_tick_vals]

    cb_ax.set_yticks(yticks)
    cb_ax.set_xticks(xticks)
    cb_ax.set_yticklabels(phase_tick_labels)
    cb_ax.set_xticklabels(amp_tick_labels)
    cb_ax.set_ylabel("$\phi$ [rad]")
    cb_ax.set_xlabel("$|E|$")
    cb_ax.tick_params(axis='y', right=True, labelright=True, left=False, labelleft=False)
    cb_ax.yaxis.set_label_position("right")


    plt.tight_layout()
    save_dir = dir + "\\" + name
    plt.savefig(save_dir, bbox_inches="tight", dpi = 300)
    plt.show()

def print_max_amp_or_phase_value(array):
        """
        Function that loops through each element in the array and prints out the maximum value

        Arguments:
            - array: 1D array of values
        Returns:
            - maximum value in the array
        """
        max_val = []
        for i in array:
            max_val.append(max(i))
        max_value = max(max_val)
        return max_value
#######################################################################################################################################################
def assign_core_properties(simulation_val):
    '''
    Function that takes in the dictionary simulation_val and assigns template parameters such as 
    core diameter and core delta prior to dumping json. If mode selective (=1) then the monitored core
    will be set to None so that skopt may optimise its parameters alone.
    '''
    if "core_num" in simulation_val and "core_delta" in simulation_val:
        ms_diam = [variable_params["core_diam"]] * simulation_val["core_num"]
        ms_delta = [simulation_val["core_delta"]] * simulation_val["core_num"]

        if len(ms_diam) != simulation_val["core_num"] or len(ms_delta) != simulation_val["core_num"]:
            raise Exception("Number of specified core properties does not match the number of modelled cores")

        for i, (diam, delta) in enumerate(zip(ms_diam, ms_delta), start=1):
            if simulation_val["mode_selective"] == 1 and i == simulation_val["core_to_monitor"]:
                simulation_val[f"core_{i}"] = {
                    "core_diam": None,
                    "delta": None
                }
            else:
                simulation_val[f"core_{i}"] = {
                    "core_diam": diam,
                    "delta": delta
                }
#######################################################################################################################################################
def apply_complex_map(field, cmap, power=1.0, normalise=True, shift=0.0):
    """
    Maps a complex-valued 2D array to an RGB image using a colormap.

    The hue is determined by the phase (angle) of each complex value, and the brightness
    is scaled by the magnitude (absolute value) raised to the given power. The result is
    normalized to the [0, 1] range for display.

    Parameters
    ----------
    field : np.ndarray
        2D array of complex values to visualize.
    cmap : callable
        A matplotlib colormap function (e.g., plt.cm.hsv).
    power : float, optional
        Exponent to apply to the magnitude for brightness scaling/gamma (default is 1.0).

    Returns
    -------
    np.ndarray
        3D array representing the RGB image (shape: field.shape + (3,)).
    """
    angles = np.angle(field)
    angles_norm = np.mod(angles + shift, 2 * np.pi) / (2 * np.pi)

    if normalise:
        amp = np.abs(field) ** power / np.max(np.abs(field) ** power)
    else:
        amp = np.abs(field) ** power

    img = cmap(angles_norm) * (amp)[:, :, None]

    if normalise:
        img = img[..., :3] / np.max(img[..., 0:3])
    else:
        img = img[..., :3]

    return img

def generate_complex_colorbar(amp_max = 1, amp_min = 0, phase_max = 2*np.pi,phase_min = 0,power=1.0, shift=0.0, cmap=cmocean.cm.phase, resolution=300):
    """
    Create an RGB image for a hue-brightness colorbar: hue = phase, brightness = amplitude.
    """
    phase_vals = np.linspace(phase_min, phase_max, resolution)
    amp_vals = np.linspace(amp_min, amp_max, resolution)
    phase_grid, amp_grid = np.meshgrid(phase_vals, amp_vals)

    field = amp_grid * np.exp(1j * (phase_grid + shift))

    rgb_img = cmap((np.angle(field) + np.pi) / (2 * np.pi))[:, :, :3]
    brightness = (np.abs(field) ** power)[:, :, None]
    rgb_img *= brightness

    # Transpose it to rotate for vertical display
    colorbar_img_vertical = np.transpose(rgb_img, (1, 0, 2))  # shape becomes (phase, amp, 3)

    return colorbar_img_vertical

def plot_rsoft_femsim_output(num_modes, results_folder = "", name = "", title = "", polarization = "", save = True, false_mode = True):
    '''
    Function to search for femsim mode files in the ex polarization.

    num_modes: number of modes accounting for orientations but NOT polarization
    results_folder: string locating the directory of the FemSIM results
    name: prefix of FemSIM files
    title: name of the saved image
    save: Determines whether the combined mode image is saved or not with 'title' as the filename
    false_mode: if true plots a mode with no zoom where in most cases is a false solution determined by FemSIM

    Returns:
        collated image of all the modes supported by the fibre calculated by FemSIM
    '''
    
    mode_desired = mode_wanted_considering_mode_orientations(LP_mode_dict, num_modes) 
    print(f"{num_modes} ({mode_desired} distinct) modes with their orientations and polarizations have been plotted")
    if false_mode:
        total_subplots = mode_desired * 3 + 1 # total number of modes including orientations of polarisation, and the false mode
    else:
        total_subplots = mode_desired * 3
    cols = 6
    rows = total_subplots // cols

    # check if additional row is needed
    if total_subplots % cols !=0:
        rows +=1

    fig, axes = plt.subplots(rows, cols, figsize=(20,15), constrained_layout = True)

    im_list = []

    for i, ax in enumerate(axes.flat):
        if i >= total_subplots:
            ax.axis('off')  # Hide extra axes if grid is larger than data
            continue
        
        if i < 10:
            field = f"{name}_{polarization}.m0{i}"
        else:
            field = f"{name}_{polarization}.m{i}"

        dat = pd.read_csv(results_folder + "\\" + field, skiprows = 4, sep=r'\s+', header = None)
        dat = np.asarray(dat)
        Nx, Ny = dat.shape
        new_x = np.linspace(-Nx//2, Nx//2, Nx)
        new_y = np.linspace(-Ny//2, Ny//2, Ny)
        amp, phase = dat[:, ::2], dat[:, 1::2]
        im = ax.imshow(amp,
                    extent=[new_x[0], new_x[-1], new_y[0], new_y[-1]],
                    aspect='auto', cmap='afmhot_10u')

        im_list.append(im)

        ax.set_xlabel("X ($\mu m$)")
        ax.set_ylabel("Y ($\mu m$)")

        if false_mode and i != total_subplots-1:
            ax.set_xlim(-100,100)
            ax.set_ylim(-200,200)
        else:
            continue
            
    plt.suptitle(title)
    cbar = fig.colorbar(im_list[-1], ax=axes.ravel().tolist(), location='right')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label='Amplitude', size=14, weight='bold')

    if save:
        plt.savefig(results_folder + "\\" + title + ".png", dpi=1000)
    plt.show()


def make_animation(data_folder="", file_pattern="", output_gif="", interval=100):
    '''
    Function that collates individual field files into a single GIF animation.
    data_folder: location of the individual field files. NOTE: this must only contain the field files to animate and nothing else
    file_pattern: prefix of the field files
    output_gif: string determining the output filename of the GIF
    interval: time in milliseconds between each frame
    plot: string to determined what part of the field values to plot. Use either "amp" (default) or "ph"

    Returns:
        GIF animation of the evolution of field files
    '''
    file_list = sorted(glob.glob(os.path.join(data_folder, file_pattern)))

    print(f"Found {len(file_list)} files.")

    # Load first frame
    first_file = file_list[0]
    dat = pd.read_csv(first_file, skiprows=4, sep=r'\s+', header=None)
    dat = np.asarray(dat)
    amp, phase = dat[:, ::2], dat[:, 1::2]
    Z = amp*np.exp(1j * phase)

    Nx, Ny = dat.shape
    new_x = np.linspace(-Nx//2, Nx//2, Nx)
    new_y = np.linspace(-Ny//2, Ny//2, Ny)

    fig, (ax, ax_cb) = plt.subplots(1,2, figsize=(10, 6), width_ratios=[4, 1])

    im = ax.imshow(apply_complex_map(Z, cmocean.cm.phase), extent=[new_x[0], new_x[-1], new_y[0], new_y[-1]], aspect='auto')

    ax.set_ylabel("X ($\mu m$)")
    ax.set_xlabel("Y ($\mu m$)")

    ax_cb.imshow(generate_complex_colorbar(resolution=Z.shape[0]), extent=[0, 1, -np.pi, np.pi], origin='lower')
    ax_cb.set_xlabel("$|E|$")
    ax_cb.set_ylabel("$\phi$ [rad]")
    ax_cb.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax_cb.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])
    ax_cb.tick_params(axis='y', right=True, labelright=True, left=False, labelleft=False)
    ax_cb.yaxis.set_label_position("right")
    ax_cb.set_xticks([0, 1])
    
    def update(frame_idx):
        filename = file_list[frame_idx]
        dat = pd.read_csv(filename, skiprows=4, sep=r'\s+', header=None)
        dat = np.asarray(dat)
        amp, phase = dat[:, ::2], dat[:, 1::2]
        Z = amp*np.exp(1j * phase)

        im.set_data(apply_complex_map(Z, cmocean.cm.phase))
        ax.set_title(f"Frame {frame_idx}")
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(file_list), blit=True, interval=interval, repeat=True #(plotting_phase,) is a 1-element tuple, required by FuncAnimation
    )
    matplotlib.rcParams['animation.ffmpeg_path'] = r"C:\Users\justinvella\Desktop\Git_Repos\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
    writer = animation.FFMpegWriter(fps = 30, metadata=dict(artist = "Justin Vella"))
    # ani.save(output_gif, writer='pillow')
    ani.save(f"{output_gif}.mp4", writer = writer)
    print(f"Saved animation to {output_gif}")

    plt.close(fig) 
#######################################################################################################################################################
"""
This function block is taken from Barnaby's lanternfiber_minimal.py

The fiber-mode relevant parts from lanternfiber.py

This class uses
ofiber https://ofiber.readthedocs.io
polarTransform https://polartransform.readthedocs.io/en/latest/getting-started.html
"""
import numpy as np
import ofiber
import matplotlib.pyplot as plt
import polarTransform
from scipy import ndimage

class lanternfiber:
    def __init__(self, n_core=None, n_cladding=None, core_radius=None, wavelength=None, nmodes=19, nwgs=19,
                 datadir='./'):
        self.n_core = n_core
        self.n_cladding = n_cladding
        self.core_radius = core_radius
        self.wavelength = wavelength
        self.allmodes_b = None
        self.allmodes_l = None
        self.allmodes_m = None
        self.nmodes = nmodes
        self.nwgs = nwgs
        self.max_r = None
        self.datadir = datadir

        self.all_smpowers = []
        self.all_mmpowers = []
        self.all_mmphases = []
        self.all_smphases = []
        self.Cmat = None # Transfer matrix
        self.Imat = None # Intensity-output matrix
        self.out_field_ampl = []
        self.out_field_phase = []
        self.wg_posns = None
        self.microns_per_pixel = None
        self.npix = None
        self.input_field = None
        self.allmodefields_rsoftorder = None
        self.all_runallbats = []
        self.all_hyperbats = []
        self.all_indiv_commands = []
        self.all_wls = None
        self.allBatfileNames = []

        if n_core is not None:
            self.NA = self.calc_numerical_aperture(n_core, n_cladding)
            self.V = self.calc_V_parameter(core_radius, self.NA, wavelength)

        # If the order of Rsoft monitor objects does not match the conventional waveguide order,
        # specify order here. Using rsoft numbering (so starts at 1).
        self.monitor_order = [10, 9, 14, 15, 11, 6, 5, 4, 3, 8, 13, 18, 19, 16, 17, 12, 7, 2, 1]

        # Specify mode indices
        self.LP_modes = np.array([[0,1],
                             [0,2],
                             [0,3],
                             [1,1],
                             [-1,1],
                             [1,2],
                             [-1,2],
                             [2,1],
                             [-2,1],
                             [2,2],
                             [-2,2],
                             [3,1],
                             [-3,1],
                             [3,2],
                             [-3,2],
                             [4,1],
                             [-4,1],
                             [5,1],
                             [-5,1]
                             ])

        # Make text mode labels
        modelabels = []
        for k in range(self.nmodes):
            if k < 3:  # Assumes first 3 modes are LP0x modes
                label = 'LP%d%d' % (self.LP_modes[k, 0], self.LP_modes[k, 1])
            else:
                if self.LP_modes[k, 0] > 0:
                    suf = 'a'
                else:
                    suf = 'b'
                label = 'LP%d%d' % (np.abs(self.LP_modes[k, 0]), self.LP_modes[k, 1]) + suf
            modelabels.append(label)
        self.modelabels = modelabels


    def calc_numerical_aperture(self, n_core, n_cladding):
        return np.sqrt(n_core ** 2 - n_cladding ** 2)


    def calc_V_parameter(self, core_radius, NA, wavelength):
        V = 2 * np.pi / wavelength * core_radius * NA
        return V


    def find_fiber_modes(self, max_l=100, return_n_unique=False, verbose=True):
        """
        Finds LP modes for the specified fiber.

        Parameters
        ----------
        max_l
            Maximum number of l modes to find (can be arbitrarily large)
        """
        self.NA = self.calc_numerical_aperture(self.n_core, self.n_cladding)
        self.V = self.calc_V_parameter(self.core_radius, self.NA, self.wavelength)

        allmodes_b = []
        allmodes_l = []
        allmodes_m = []
        for l in range(max_l):
            cur_b = ofiber.LP_mode_values(self.V, l)
            if len(cur_b) == 0:
                break
            else:
                allmodes_b.extend(cur_b)
                ls = (np.ones_like(cur_b)) * l
                allmodes_l.extend(ls.astype(int))
                ms = np.arange(len(cur_b))+1
                allmodes_m.extend(ms)

        allmodes_b = np.asarray(allmodes_b)
        nLPmodes = len(allmodes_b)
        # print('Total number of LP modes found: %d' % nLPmodes)
        l = np.asarray(allmodes_l)
        total_unique_modes = len(np.where(l == 0)[0]) + len(np.where(l > 0)[0])*2
        if verbose:
            print('Total number of unique modes found: %d' % total_unique_modes)
        self.allmodes_b = allmodes_b
        self.allmodes_l = allmodes_l
        self.allmodes_m = allmodes_m
        self.nLPmodes = nLPmodes

        # ADDED - HACK?
        self.nmodes = total_unique_modes
        if return_n_unique:
            return total_unique_modes


    def make_fiber_modes(self, max_r=2, npix=100, zlim=0.04, show_plots=False,
                         normtosum=True, rotate_mode_angle=None):
        """
        Calculate the LP mode fields, and store as polar and cartesian amplitude maps

        Parameters
        ----------
        max_r
            Maximum radius to calculate mode field, where r=1 is the core diameter
        npix
            Half-width of mode field calculation in pixels
        zlim
            Maximum value to plot
        show_plots : bool
            Whether to produce a plot for each mode
        normtosum : bool
            If True, normalise each mode field so summed power = 1
        """

        r = np.linspace(0, max_r, npix) # Radial positions, normalised so core_radius = 1
        self.max_r = max_r
        self.npix = npix
        self.allmodefields_cos_polar = []
        self.allmodefields_cos_cart = []
        self.allmodefields_sin_polar = []
        self.allmodefields_sin_cart = []
        self.allmodefields_rsoftorder = []

        array_size_microns = self.max_r * self.core_radius * 2
        self.microns_per_pixel = array_size_microns / (npix*2)

        for mode_to_calc in range(self.nLPmodes):
            field_1d = ofiber.LP_radial_field(self.V, self.allmodes_b[mode_to_calc],
                                              self.allmodes_l[mode_to_calc], r)

            phivals = np.linspace(0, 2*np.pi, npix)
            phi_cos = np.cos(self.allmodes_l[mode_to_calc] * phivals)
            phi_sin = np.sin(self.allmodes_l[mode_to_calc] * phivals)

            rgrid, phigrid = np.meshgrid(r, phivals)
            field_r_cos, field_phi = np.meshgrid(phi_cos, field_1d)
            field_r_sin, field_phi = np.meshgrid(phi_sin, field_1d)
            field_cos = field_r_cos * field_phi
            field_sin = field_r_sin * field_phi

            # Normalise each field so its total intensity is 1
            field_cos = field_cos / np.sqrt(np.sum(field_cos**2))
            field_sin = field_sin / np.sqrt(np.sum(field_sin**2))
            field_cos = np.nan_to_num(field_cos)
            field_sin = np.nan_to_num(field_sin)

            field_cos_cart, d = polarTransform.convertToCartesianImage(field_cos.T)
            field_sin_cart, d = polarTransform.convertToCartesianImage(field_sin.T)

            if rotate_mode_angle is not None:
                print('Warning: rotating mode fields by %f degrees. ONLY APPLIES TO CARTESIAN FIELDS!' %
                      rotate_mode_angle)
                field_cos_cart = ndimage.rotate(field_cos_cart, rotate_mode_angle, reshape=False)
                field_sin_cart = ndimage.rotate(field_sin_cart, rotate_mode_angle, reshape=False)

            if normtosum:
                field_cos = field_cos / np.sqrt(np.sum(field_cos**2))
                field_sin = field_sin / np.sqrt(np.sum(field_sin**2))
                field_cos_cart = field_cos_cart / np.sqrt(np.sum(field_cos_cart**2))
                field_sin_cart = field_sin_cart / np.sqrt(np.sum(field_sin_cart**2))

            self.allmodefields_cos_polar.append(field_cos)
            self.allmodefields_cos_cart.append(field_cos_cart)
            self.allmodefields_sin_polar.append(field_sin)
            self.allmodefields_sin_cart.append(field_sin_cart)
            self.allmodefields_rsoftorder.append(field_cos_cart)
            if self.allmodes_l[mode_to_calc] > 0:
                self.allmodefields_rsoftorder.append(field_sin_cart)

            if show_plots:
                self.plot_fiber_modes(mode_to_calc, zlim)
                plt.pause(0.5)


    def plot_fiber_modes(self, mode_to_plot, zlim=0.04, fignum=1):
        """
        Make a plot of the cos and sin amplitudes of a given mode

        Parameters
        ----------
        mode_to_plot
            Number of mode to plot
        zlim
            Maximum value to plot
        """
        plt.figure(fignum)
        plt.clf()
        plt.subplot(121)
        sz = self.max_r * self.core_radius
        plt.imshow(self.allmodefields_cos_cart[mode_to_plot], extent=(-sz, sz, -sz, sz), cmap='bwr',
                   vmin=-zlim, vmax=zlim)
        plt.xlabel('Position ($\mu$m)')
        plt.ylabel('Position ($\mu$m)')
        plt.title('Mode l=%d, m=%d (cos)' % (self.allmodes_l[mode_to_plot], self.allmodes_m[mode_to_plot]))
        core_circle = plt.Circle((0,0), self.core_radius, color='k', fill=False, linestyle='--', alpha=0.2)
        plt.gca().add_patch(core_circle)
        plt.subplot(122)
        sz = self.max_r * self.core_radius
        plt.imshow(self.allmodefields_sin_cart[mode_to_plot], extent=(-sz, sz, -sz, sz), cmap='bwr',
                   vmin=-zlim, vmax=zlim)
        plt.xlabel('Position ($\mu$m)')
        plt.title('Mode l=%d, m=%d (sin)' % (self.allmodes_l[mode_to_plot], self.allmodes_m[mode_to_plot]))
        core_circle = plt.Circle((0,0), self.core_radius, color='k', fill=False, linestyle='--', alpha=0.2)
        plt.gca().add_patch(core_circle)
        plt.pause(0.001)
        print('LP mode %d, %d' % (self.allmodes_l[mode_to_plot], self.allmodes_m[mode_to_plot]))