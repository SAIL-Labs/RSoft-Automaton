import numpy as np, matplotlib.pyplot as plt, pandas as pd
import subprocess, os, shutil, warnings, csv, json
from skopt import gp_minimize, Optimizer
from skopt.space import Real
from skopt.utils import dump
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Circuit_Properties import *
from Functions import *
from HexProperties import *
from rstools import RSoftUserFunction, RSoftCircuit

with open("variable_paras.json", "r") as f:
    dat = json.load(f)
    name = dat["Name"]
    background_index = dat["background_index"]
    core_num = dat["core_num"]
    delta_expr = dat["Core_delta"]
    delta = dat["delta"]

with open("prior_space.json", "r") as read:
    param_range = json.load(read)

para_space = [Real(low, high, name=prior_name) for prior_name, (low, high) in param_range.items()]

with open("fibre_prop.json", "r") as mp_para:
    mp_data = json.load(mp_para)
    num_para = mp_data["num_paras"]
    batch_num = mp_data["batch_number"]
    free_space_wavelength = mp_data["free_space_wavelength"]

def RunRsoft(params): 
    param_dict = {dim.name: val for dim, val in zip(para_space, params)}
    ##########################################################################################################################################################
    '''
    This block of code will test the incoming prior values and filter them such that the 
    V-number satisfies the single mode fibre condition.
    '''
    # V, NA = calc_V(param_dict["Corediam"], delta_expr, background_index, free_space_wavelength)
    # if V >= 2.405 or V <= 1.0:
    #     return -1e6
    # prior_sampling(param_dict,delta_expr,background_index, free_space_wavelength)
    # filter_parameter_space_by_v_number(param_dict,background_index, free_space_wavelength,delta_expr,288)
    ##########################################################################################################################################################
    '''
    Manual setup to loop through a list of values
    Runs the terminal line that will initiate RSoft. 
    All output files will appear in a subfolder on the Desktop (windows)
    '''
    name_tag = "_".join(f"{key}_{val:.6f}" for key, val in param_dict.items())

    filename = f"MCF_{name_tag}.ind"
    prefix   = f"prefix={name_tag}"
    folder   = f"Sim_{name_tag}"

    # Read original template
    with open(f"{name}.ind", "r") as r:
        lines = r.readlines()

    # Construct symbolic delta expression
    # delta_expr = param_dict["Core_index"] 
    # delta_expr = param_dict["Core_index"] - background_index

    # Insert delta after core segment start
    lines = insert_after_match(lines, "comp_name = core", 
                        [f"\tbegin.delta = {delta_expr}\n",
                        f"\tend.delta = {delta_expr}\n" ])
    # lines = insert_after_match(lines, "comp_name = Super Cladding", 
    #                     [f"\tbegin.delta = {delta}\n",
    #                     f"\tend.delta = {delta}\n" ])

    # Insert delta after monitor segment starts
    # lines = insert_after_match(lines, "monitor ",f"\tmonitor_delta = {delta_expr}\n")

    # Build the updated lines
    modified_lines = []
    in_core = False
    with open("core_positions.json", "r") as g:
        core_positions = json.load(g)
        core_index = -1

    for line in lines:
        line_strip = line.strip()
        replaced = False
        '''
        Might want to change this to self.core_position((0,0)) in the future
        '''
        # Detect start of a monitor block for the core (monitor 1)
        if line_strip.startswith("monitor ") and "1" in line_strip:
            in_core_monitor = True
        elif line_strip.startswith("monitor "):  # any other monitor
            in_core_monitor = False

        # Detect if the segment is the core or cladding
        if line_strip.startswith("comp_name =") and "core_" in line_strip:
            in_core = True
            core_index += 1
        elif line_strip.startswith("comp_name ="):  # not a core
            in_core = False
        
        # Insert delta only inside core segments
        if in_core and line_strip.startswith("begin.width ="):
            modified_lines.append(line)
            modified_lines.append(f"\tbegin.delta = {delta_expr}\n")
            continue  # skip to next line
        elif in_core and line_strip.startswith("end.width ="):
            modified_lines.append(line)
            modified_lines.append(f"\tend.delta = {delta_expr}\n")
            continue

        for param, val in param_dict.items():
            if param == "Core_index":
                continue
            # dynamically replace the monitor/launch field dimensions
            # if param == "Corediam":
            #     if line_strip.startswith("monitor_height =") and in_core_monitor:
            #         modified_lines.append(f"\tmonitor_height = {(val * 1.1):.6f}\n")
            #         replaced = True
            #         break
            #     if line_strip.startswith("monitor_width =") and in_core_monitor:
            #         modified_lines.append(f"\tmonitor_width = {(val * 1.1):.6f}\n")
            #         replaced = True
            #         break
            #     if line_strip.startswith("launch_height ="):
            #         modified_lines.append(f"\tlaunch_height = {(val):.6f}\n")
            #         replaced = True
            #         break
            #     if line_strip.startswith("launch_width ="):
            #         modified_lines.append(f"\tlaunch_width = {(val):.6f}\n")
            #         replaced = True
            #         break
            if line_strip.startswith(f"{param} ="):
                modified_lines.append(f"{param} = {val:.6f}\n")
                replaced = True
                break

            if param == "acore_taper_ratio" and in_core:
                if line_strip.startswith("begin.x ="):
                    x, _ = core_positions[core_index]
                    modified_lines.append(f"\tbegin.x = {x / val:.6f}\n")
                    replaced = True
                    break
                if line_strip.startswith("begin.y =") and in_core:
                    _, y = core_positions[core_index]
                    modified_lines.append(f"\tbegin.y = {y / val:.6f}\n")
                    replaced = True
                    break

        if not replaced:
            modified_lines.append(line)

    # Write the final .ind file with symbolic delta expression
    with open(filename, "w") as out:
        out.writelines(modified_lines)
    
    # Set up folders
    user_home = os.path.expanduser("~")
    desktop_path = os.path.join(user_home, "Desktop")
    results_root = os.path.join(desktop_path, "Results")
    results_folder = os.path.join(results_root, folder)
    os.makedirs(results_folder, exist_ok=True)

    # Run simulation, if multiprocessing takes too long time it out and raise error
    # try:
        # os.environ["OMP_NUM_THREADS"] = "2" # my attempt at programmatically setting the number of threads BeamProp uses
        # there should be a way to make the outputs automatically be placed in a certain directory
    subprocess.run(["bsimw32", filename, prefix, "wait=0"], check=True) # put this else where
    # except subprocess.TimeoutExpired:
    #     print("RSoft timed out")

    # Read .mon file
    uf = RSoftUserFunction()
    uf.read(f"{name_tag}.mon")
    x_all, y_all = uf.get_arrays()
    x = x_all
    y = np.real(y_all)
    # try:
    #     x = x_all[0]
    #     y = y_all[0]
    #     # Ensure both are iterable
    #     if not hasattr(x, '__len__') or not hasattr(y, '__len__'):
    #         raise ValueError("Monitor data is not iterable.")
    # except Exception as e:
    #     print(f"[Error] Invalid monitor data in {name_tag}.mon: {e}")
    #     return 1e6

    # Save results, make a separate function here
    results = {0: y}  # just one run per call
    csv_tag = f"Throughput_{name_tag}.csv"
    with open(csv_tag, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", name_tag])
        for i in range(len(x)):
            writer.writerow([x[i], y[i]])

    # Move files to results folder
    for file in os.listdir():
        if file.startswith(name_tag) or file == filename:
            shutil.move(file, os.path.join(results_folder, file))

    df = pd.read_csv(csv_tag)
    throughput_difference = abs(df[name_tag].iloc[0] - df[name_tag].iloc[-1])/df[name_tag].iloc[0]
    average = df[name_tag].mean()

    for file in os.listdir():
        if file.startswith(csv_tag):
            shutil.move(file, os.path.join(results_folder, file))
    return -average # throughput_difference #negative since skopt minimizes the function

# initialises multiprocessing function that runs the RSoft simulation
def mp_eval(params):
    # just for tracking purposes if running from the terminal
    print(f"[PID {os.getpid()}] Starting RunRsoft with {params}")
    return RunRsoft(params)

# backend of skopt.gp_minimize that can handle multiprocessing
opt = Optimizer(
    dimensions = para_space,
    base_estimator = "GP",
    acq_func = "EI",
    random_state=42
)

# the multiprocessing in all its glory. The first if __name__ == "__main__" is required 
if __name__ == "__main__":    
    # how many values in each parameter space to run simulation with
    total_calls = num_para
    # this is the number of points to sample simultaneously. Increase to cycle through prior space quicker at the cost of CPU computation
    batch_size = batch_num
    all_results = []

    for i in range(0, total_calls, batch_size):
        
        # Suggest next batch of points
        param_batch = opt.ask(batch_size)

        # Evaluate in parallel
        with mp.Pool(batch_size) as pool:
            result_batch = pool.map(mp_eval, param_batch)

        # Feed results back to optimizer
        opt.tell(param_batch, result_batch)
        
        all_results.extend(zip(param_batch, result_batch))
        # save results for plotting/analysis
        dump(opt, "rsoft_opt_checkpoint.pkl", store_objective=False)

        # Unpack results
        x_iters = [r[0] for r in all_results]  # parameter sets
        y_vals = [-r[1] for r in all_results]  # throughput values

        c_num = np.arange(len(x_iters))  # iteration counter
        param_names = [dim.name for dim in opt.space.dimensions]
        n_params = len(param_names)

        # Find best result
        # best_idx = np.argmin(y_vals)
        best_idx = np.argmax(y_vals)
        best_params = x_iters[best_idx]
        best_throughput = y_vals[best_idx]

        # Create plots
        fig, axes = plt.subplots(n_params, 1, figsize=(8, 4.5 + 1.5 * n_params), sharex=False)

        if n_params == 1:
            axes = [axes]

        for k, (param_name, ax) in enumerate(zip(param_names, axes)):
            x_vals = [x[k] for x in x_iters]

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
        plt.savefig(f"plot_iteration_{i//batch_size + 1}.png", dpi=300)

        # move images to prevent clumping
        user_home = os.path.expanduser("~")
        desktop_path = os.path.join(user_home, "Desktop")
        results_root = os.path.join(desktop_path, "Results")
        images_dir = os.path.join(results_root, "Images")

        os.makedirs(images_dir, exist_ok = True)
        for file in os.listdir():
            if file.startswith("plot_iteration_"):
                shutil.move(file, os.path.join(images_dir, file))
        
        para_tag = "best_params_log.csv"
        with open(para_tag, "w", newline="") as log:
            writer = csv.writer(log)
            writer.writerow([i // batch_size + 1] + list(best_params) + [best_throughput])