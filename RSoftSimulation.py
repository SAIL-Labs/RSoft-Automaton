import numpy as np, os, shutil, csv
import subprocess, json, time
from pathlib import Path
from skopt import Optimizer
from skopt.space import Real
from skopt.utils import dump
import multiprocessing as mp


from Circuit_Properties import *
from Functions import *
from HexProperties import *
from template import * 
from rstools import RSoftUserFunction, RSoftCircuit # type:ignore

class RSoftSim:
    def __init__(self):
        self.sym = {}

    def init_priors(self , prior_space_pid, build_tf, custom = None):
        base_priors = {}
        if custom:
            base_priors.update(custom)
        self.prior_space = base_priors

        if build_tf:
            with open(prior_space_pid, "w") as write:
                json.dump(self.prior_space, write)
        else:
            with open("prior_space.json", "w") as write:
                json.dump(self.prior_space, write)
        return base_priors

    def generate_core_positions(self):
        SimParam = Simulation_params
        core_sep = fixed_params["core_sep"]
        grid_type = SimParam["grid_type"]
        core_num = SimParam["core_num"]
        if grid_type == "Hex":
            """
            Generate hexagonal core coordinates and store internally.
            """
            
            if core_num % 2 == 0:
                raise ValueError(f"The number of cores must be odd to perfectly fit inside the hex grid. Received: {core_num}")

            # row_numbers = [number_rows(core_num)]
            # for row_num in row_numbers:
                # hcoord, vcoord = generate_hex_grid(row_num, core_sep, include_centre = SimParam["plot_centre_core"])
                # self.core_positions = list(zip(hcoord, vcoord))
            hcoord, vcoord = generate_hex_ring_grid(fixed_params["MCFCladd"]/2, core_sep, SimParam["core_num"], include_center = SimParam["plot_centre_core"])
            self.core_positions = list(zip(hcoord, vcoord))
            with open("core_positions.json", "w") as g:
                json.dump(self.core_positions, g)

        if grid_type == "Pent":
            """
            Generate pentagon core coordinates and store internally.
            """
            # estimated_radius = estimate_pentagon_radius(core_num,core_sep)
            hcoord, vcoord = generate_pentagon_grid(fixed_params["MCFCladd"] / 2, core_sep, Simulation_params["core_num"], include_centre = SimParam["plot_centre_core"])
            self.core_positions = list(zip(hcoord, vcoord))
            with open("core_positions.json", "w") as g:
                json.dump(self.core_positions, g)

        if grid_type == "Circ":
            """
            Generate circular core coordinates and store internally.
            """
            estimated_radius = estimate_circle_radius_with_autofit(core_num,core_sep)
            hcoord, vcoord = generate_filled_circle_grid(estimated_radius,core_sep)
            self.core_positions = list(zip(hcoord, vcoord))
            with open("core_positions.json", "w") as g:
                json.dump(self.core_positions, g)
        
        if grid_type == "Square":
            '''
            Generate square cre coordinates and store internally.
            '''
            hcoord, vcoord = generate_square_grid(core_num, core_sep)
            self.core_positions = list(zip(hcoord, vcoord))
            with open("core_positions.json", "w") as g:
                json.dump(self.core_positions, g)

    def RunRSoftSim(self, name_tag, fixed, vars, fixed_length, param_range, simulation_val, csv_path, json_config, prior_space_pid):
        filename = f"{name_tag}.ind"
        sim_tool = simulation_val.get("sim_tool", RSoft_params["sim_tool"])
        # Run RSoft simulation
        if sim_tool == "ST_BEAMPROP":
            prefix   = f"prefix={name_tag}"
            folder   = f"BP_{name_tag}"

            results_folder = create_folders(folder)

            try:
                subprocess.run(
                    [r"C:\Synopsys\PhotonicSolutions\2024.09-SP2-1\RSoft\bin\bsimw32.exe", filename, prefix, "wait=0"],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Command failed with code {e.returncode}")
                print("Command:", e.cmd)
                print("stdout:\n", e.stdout)
                print("stderr:\n", e.stderr)
                return -1e6
        elif sim_tool == "ST_FEMSIM":
            prefix   = f"prefix=FS_{name_tag}"
            folder   = f"FS_{name_tag}"

            results_folder = create_folders(folder)

            try:
                subprocess.run(
                    ["femsim", filename, prefix, "wait=0"],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Command failed with code {e.returncode}")
                print("Command:", e.cmd)
                print("stdout:\n", e.stdout)
                print("stderr:\n", e.stderr)
                return -1e6

        # Move all output files immediately after simulation
        for file in os.listdir():
            if file.startswith(name_tag) or file == filename or file == csv_path or file == json_config or file == prior_space_pid:
                shutil.move(file, os.path.join(results_folder, file))

        # Safely access the .mon file in its new location
        if Launch_params["mon_type"] == "pathway_mon" and sim_tool != "ST_FEMSIM":
            mon_path = Path(results_folder) / f"{name_tag}.mon"
            timeout = 10
            t_start = time.time()
            while not mon_path.exists():
                if time.time() - t_start > timeout:
                    raise FileNotFoundError(f"{mon_path} not found within {timeout} seconds after simulation.")
                time.sleep(0.1)

        elif Launch_params["mon_type"] == "port_mon" and sim_tool != "ST_FEMSIM":
            mon_path = Path(results_folder) / f"{name_tag}_mon.dat"
            timeout = 10
            t_start = time.time()
            while not mon_path.exists():
                if time.time() - t_start > timeout:
                    raise FileNotFoundError(f"{mon_path} not found within {timeout} seconds after simulation.")
                time.sleep(0.1)

        # Read .mon file from moved location
        uf = RSoftUserFunction()
        uf.read(str(mon_path))
        if Launch_params["mon_type"] == "pathway_mon":
            x_all, y_all, z_all = uf.get_arrays()
            num_monitors = z_all.shape[1]

            # Write throughput CSV to same folder
            csv_tag = f"Throughput_{name_tag}.csv"
            csv_pathway = Path(results_folder) / csv_tag

            with open(csv_pathway, mode="w", newline="") as file:
                writer = csv.writer(file)
                header = ["x"] + [f"Monitor_{i}" for i in range(num_monitors)]
                writer.writerow(header)
                for i in range(z_all.shape[0]):
                    row = [x_all[i]] + [np.real(z_all[i, j]) for j in range(z_all.shape[1])]
                    writer.writerow(row)
        
        elif Launch_params["mon_type"] == "port_mon":
            x_all, y_all, z_all = uf.get_arrays()
            num_monitors = (z_all.shape[1] - Simulation_params["core_num"])

            # Write throughput CSV to same folder
            csv_tag = f"Throughput_{name_tag}.csv"
            csv_pathway = Path(results_folder) / csv_tag

            with open(csv_pathway, mode="w", newline="") as file:
                writer = csv.writer(file)
                header = ["x"]
                for i in range(num_monitors):
                    header.append(f"Monitor_{i+1}_Amplitude")
                    header.append(f"Monitor_{i+1}_Phase")
                writer.writerow(header)
                for i in range(z_all.shape[0]):
                 #NOTE: by default row contains amplitude and phase values, should expect 2*core_num entries
                    row = [x_all[i]] + list(z_all[i])
                    writer.writerow(row)

        if Simulation_params["metric"] == 'TH':
            return -throughput_metric(csv_pathway, fixed_length,
                                       fixed, vars, 
                                       param_range, Simulation_params["mode_selective"])
        # if Simulation_params['metric'] == 'MS':
        #     return - mode_selective_metric(csv_pathway, Simulation_params["core_to_monitor"], 
        #                                    f"LP{Launch_params['launch_mode']}{Launch_params['launch_mode_radial']}")
        if Simulation_params["metric"] == "TF":
            if Launch_params["mon_type"] == "port_mon":
                transfer_vector, throughput = transfer_matrix_component(csv_pathway, row, port_mon = True)
                return transfer_vector, -throughput
            else:
                transfer_vector, throughput = transfer_matrix_component(csv_pathway, row)
                return transfer_vector, -throughput

    def build_circuit(self, params, build_tf, json_config, csv_path, simulation_val, prior_space_pid): # maybe put this into its own function. Make it universal.
        """
        Create the design file using template.py and 
        write to separate .ind file. Also contains function to run BeamProp
        and scikit Optimize
        """
        if build_tf:
            with open(json_config, "r") as launch_config:
                sim_val = json.load(launch_config)
        else:
            with open("launch_config.json", "r") as launch_config:
                sim_val = json.load(launch_config)
        
        for key, val in sim_val.items():
            Launch_params[key] = val

        # load prior space
        if build_tf:
            for attempt in range(10):
                try:
                    with open(prior_space_pid, "r") as read:
                        param_range = json.load(read)
                    break
                except json.decoder.JSONDecodeError:
                    time.sleep(0.2)
            else:
                raise RuntimeError(f"Failed to load {prior_space_pid} after retries.")
        else:
            for attempt in range(10):
                try:
                    with open("prior_space.json", "r") as read:
                        param_range = json.load(read)
                    break
                except json.decoder.JSONDecodeError:
                    time.sleep(0.2)
            else:
                raise RuntimeError("Failed to load prior_space.json after retries.")
            
        para_space = [Real(low, high, name=prior_name) for prior_name, (low, high) in param_range.items()]
        param_dict = {dim.name: val for dim, val in zip(para_space, params)}
        
        # update template file with chosen values from scikit.Optimize()
        variable_params.update(param_dict)

        self.circuit = RSoftCircuit()

        """
        Load variable, fixed, and launch parameters from template.py
        and build symbol dictionary containing ONLY the parameters that
        RSoft needs.
        """
        if sim_val["launch_type"] == "LAUNCH_MULTIMODE":
            # Remove problematic launch symbols temporarily
            launch_skip_keys = {"launch_mode", "launch_mode_radial"}
            self.sym = {**RSoft_params,
                        **{k: v for k, v in Launch_params.items() if k not in launch_skip_keys}}
        # elif RSoft_params["sim_tool"] == "ST_BEAMPROP" and simulation_val["mon_type"] == "port_mon":
        #     RSoft_skip_keys = {"fem_iterations", "fem_nev"}
        #     launch_skip_keys = {"monitor_type", "comp"}
        #     self.sym = {**{k: v for k, v in RSoft_params.items() if k not in RSoft_skip_keys},
        #                 **{l: m for l, m in Launch_params.items() if l not in launch_skip_keys}}
        else:
            self.sym = {**RSoft_params,
                        **Launch_params}

        for key, val in self.sym.items():
            # mode selective properties defined as ('string', ('string', float)) <- will throw C++ error
            if not isinstance(val, (int, float, str)):
                continue
            self.circuit.set_symbol(key, val)

        # extract taper ratio, length, core number, 
        # core and cladding diameter
        fixed = fixed_params
        vars = variable_params
        sim_param = Simulation_params
        launch = Launch_params

        fixed_length = False
        if "Taper_L" not in fixed and "Taper_L" not in vars:
            raise Exception("Taper Length defined as neither being fixed nor variable. " \
            "Please specify 'Taper_L' in template.fixed_params or template.variable_params.")

        if "Taper_L" in fixed:
            Taper_L = fixed["Taper_L"]
            fixed_length = True
        else:
            Taper_L = vars["Taper_L"]
        
        if "taper" in fixed:
            taper = fixed["taper"]
        else:
            taper = vars["taper"]
        MMF_Taper = fixed["MMF_Taper"]
        core_num = sim_param["core_num"]
        core_name = [f"core_{n}" for n in range(1, core_num + 1)]
        structure = Simulation_params["Structure"]

        cladd_diam = fixed["MCFCladd"]
        cladding_beg_dims = (cladd_diam / MMF_Taper, cladd_diam / MMF_Taper)
        cladding_end_dims = (cladd_diam , cladd_diam)

        core_beg_dims_list = []
        core_end_dims_list = []

        if sim_param["mode_selective"] == 1:
            for j, core_key in enumerate(core_name, start=1):
                if j == Simulation_params["core_to_monitor"]:
                    # core to be optimized by skopt
                    core_diam = variable_params["core_diam"]
                    core_delta = variable_params["core_delta"]
                else:
                    # pass
                    # use preconfigured values to specify core parameters
                    core_diam = 6.5 #core_params[core_key]["core_diam"]
                    core_delta = simulation_val["core_delta"]

                # Store dimensions for this core
                core_beg_dims_list.append((core_diam / taper, core_diam / taper))
                core_end_dims_list.append((core_diam, core_diam))

                core_params[core_key]["core_diam"] = core_diam
                core_params[core_key]["delta"] = core_delta
        else: 
            for j, core_key in enumerate(core_name, start=1):
                core_diam = core_params[core_key]["core_diam"]
                core_delta = core_params[core_key]["delta"]

                # Store dimensions for this core
                core_beg_dims_list.append((core_diam / taper, core_diam / taper))
                core_end_dims_list.append((core_diam, core_diam))

                core_params[core_key]["core_diam"] = core_diam
                core_params[core_key]["delta"] = core_delta

        # functions to generate the core layout, either a standard fibre or a complicated photonic lantern setup (either in hex, pent or circular geometry)
        path_num = 0
        if structure == "Fibre":
            path_num = build_fibre(self.circuit, path_num, self.core_positions, 
                        core_name, Taper_L, tuple(core * taper for core in core_beg_dims_list), core_end_dims_list,
                        simulation_val)

        elif structure == "PL":
            path_num = build_PL(self.circuit, path_num, self.core_positions, 
                    core_name, taper, Taper_L,
                    cladding_beg_dims, cladding_end_dims,
                    core_beg_dims_list, core_end_dims_list,
                    simulation_val)
        # elif structure == "pigtail":
        #     core_cladd = fixed["core_claddings"]
        #     core_cladding_beg_dims = (core_cladd / taper, core_cladd / taper)
        #     core_cladding_end_dims = (core_cladd , core_cladd)

        #     cen_core_cladd = fixed["cen_core_cladding"]
        #     cen_core_cladding_beg_dims = (cen_core_cladd / taper, cen_core_cladd / taper)
        #     cen_core_cladding_end_dims = (cen_core_cladd , cen_core_cladd)
        #     path_num = build_pigtail(self.circuit, path_num, self.core_positions, 
        #             core_name, taper, Taper_L,
        #             cladding_beg_dims, cladding_end_dims,
        #             core_beg_dims_list, core_end_dims_list,
        #             simulation_val, core_cladding_beg_dims, core_cladding_end_dims,
        #             cen_core_cladding_beg_dims, cen_core_cladding_end_dims)
            
        if simulation_val["launch_type"] == LaunchType.SM:
            launch_mode = simulation_val["launch_mode"]
            launch_mode_radial = simulation_val["launch_mode_radial"]
            name_tag = f"_LP{launch_mode}{launch_mode_radial}_".join(f"{key}_{val:.6f}" for key, val in param_dict.items())
        else:
            grid = simulation_val["grid_size"] # use only when trying to find the optimal gridding to run BeamPROP in.
            name_tag = f"_Grid{grid}".join(f"{key}_{val:.6f}" for key, val in param_dict.items())
        self.sym["Name"] = name_tag
        self.circuit.write(f"{name_tag}.ind")
        """
        Append all pathway, monitor, and launch field blocks based 
        on launch parameters. 
        """ 
        
        AddHack(name_tag, launch, path_num - 1, param_dict, simulation_val["core_to_monitor"], simulation_val.get("mon_type", Launch_params["mon_type"]))
        '''
        Manual setup to loop through a list of values. Runs the terminal line that will initiate RSoft and will calculate the 
        metric to test.
        All output files will appear in a subfolder on the Desktop (windows)
        '''

        if Simulation_params['metric'] != 'TF':
            average_throughput = self.RunRSoftSim(name_tag, fixed, 
                                              vars, fixed_length, 
                                              param_range, simulation_val, csv_path, json_config, prior_space_pid)
            return average_throughput
        else: 
            transfer_vector, average_throughput = self.RunRSoftSim(name_tag, fixed, 
                                              vars, fixed_length, 
                                              param_range, simulation_val, csv_path, json_config, prior_space_pid)
            return transfer_vector, average_throughput

    def MultProc(self, build_tf, json_config, csv_path, simulation_val, prior_space_pid): #csv_path
        images_dir = Path(os.path.expanduser("~/Desktop/Results/Images"))
        images_dir.mkdir(parents=True, exist_ok=True)

        # load prior space
        if build_tf:
            for attempt in range(10):
                try:
                    with open(prior_space_pid, "r") as read:
                        param_range = json.load(read)
                    break
                except json.decoder.JSONDecodeError:
                    time.sleep(0.2)
            else:
                raise RuntimeError(f"Failed to load {prior_space_pid} after retries.")
        else:
            for attempt in range(10):
                try:
                    with open("prior_space.json", "r") as read:
                        param_range = json.load(read)
                    break
                except json.decoder.JSONDecodeError:
                    time.sleep(0.2)
            else:
                raise RuntimeError("Failed to load prior_space.json after retries.")
            
        para_space = [Real(low, high, name=prior_name) for prior_name, (low, high) in param_range.items()]
        
        # backend of skopt.gp_minimize that can handle multiprocessing
        opt = Optimizer(
            dimensions = para_space,
            base_estimator = "GP",
            acq_func = "EI",
            random_state=42
        )

        sim_param = Simulation_params
        # how many values in each parameter space to run simulation with
        total_calls = sim_param["num_paras"]

        # this is the number of points to sample simultaneously. 
        # Increase to cycle through prior space quicker at the cost of CPU computation
        batch_size = sim_param["batch_num"]
        all_results = []
        # initialise the optimizer with template solutions
        param_names = [dim.name for dim in para_space]
        seed_params = [variable_params[k] for k in param_names]

        # self.sym["Name"] = "MCF_Test"
        if Simulation_params['metric'] != 'TF':
            seed_result = self.build_circuit(seed_params, build_tf, json_config, csv_path, simulation_val, prior_space_pid)
            opt.tell(seed_params, seed_result)
            tf_vector = None
        else:
            tf_vector, seed_result = self.build_circuit(seed_params, build_tf, json_config, csv_path, simulation_val, prior_space_pid)
            # seed_result = mode_selective_tf_matrix_metric(tf_vector, simulation_val["core_to_monitor"], simulation_val["modes_to_monitor"])
            opt.tell(seed_params, seed_result)

        all_results.append({
            "params": seed_params,
            "result": seed_result,
            "transfer_vector": tf_vector
        })

        log_optimizer_results(
            x_iters=[seed_params],
            y_vals=[-seed_result],  
            param_batch=[seed_params],
            result_batch=[seed_result],
            param_names=param_names,
            iteration_start=0,
            batch_size=1,
            penalty_batch=None,
            transfer_vector_batch=[tf_vector]
        )
        # run optimizer as normal
        for i in range(0, total_calls, batch_size):
            
            # Suggest next batch of points
            param_batch = opt.ask(batch_size)

            # Evaluate in parallel
            ctx = mp.get_context("spawn")
            args_list = [(params, build_tf, json_config, csv_path, simulation_val, prior_space_pid) for params in param_batch]
            with ctx.Pool(batch_size) as pool:
                result_batch = pool.map(run_rsoft_sim, args_list) 

            # Feed results back to optimizer
            # opt.tell(param_batch, result_batch)
            if Simulation_params['metric'] == 'TF':
                scores_only = [score for _, score in result_batch]
                opt.tell(param_batch, scores_only)
            else:
                opt.tell(param_batch, result_batch)
            
            for p, r in zip(param_batch, result_batch):
                if Simulation_params['metric'] == 'TF':
                    tf_vector, score = r
                else:
                    tf_vector = None
                    score = r
                all_results.append({
                    "params": p,
                    "result": score,
                    "transfer_vector": tf_vector
                })
                
            # all_results.extend(zip(param_batch, result_batch))

            '''
            save results for plotting/analysis
            '''
            
            # Unpack results
            x_iters = [r["params"] for r in all_results]  # parameter sets
            y_vals = [-r["result"] for r in all_results]  # throughput values
            tf_vector_val = [r["transfer_vector"] for r in all_results]

            param_names = [dim.name for dim in opt.space.dimensions]
            # log chosen values and penalties
            log_optimizer_results(x_iters, y_vals,
                                  param_batch, 
                                  [r["result"] for r in all_results[-batch_size:]],
                                  param_names, iteration_start=i,
                                  batch_size = batch_size,
                                  penalty_batch= None,
                                  transfer_vector_batch=tf_vector_val)

    def RunRSoft(self, simulation_val, prior_space_pid, csv_path, json_config, simulate=False, simulate_tf = False, build_tf = True): #csv_path, 
        '''
        Multiprocessing must to be run outside of a Jupyter cell or it will silently 
        fail/infinitely loop on the first batch
        '''

        # write in the values within simulation_val
        overwrite_template_val(json_config)
        
        # generate the positions of the cores. 
        self.generate_core_positions()

        # remove old results
        # csv_path = "optimizer_results.csv"
        if os.path.exists(csv_path):
            os.remove(csv_path)

        if simulate:
            self.MultProc(build_tf, json_config, csv_path, simulation_val, prior_space_pid) #csv_path
            return  
        # elif simulate_tf:
        #     tf_MultProc(simulation_val, prior_space_pid)
        #     return

        # -- BELOW: for "simulate=False" only --
        # Always load template/seed params first
        
        if build_tf:
            for attempt in range(10):
                try:
                    with open(prior_space_pid, "r") as read:
                        param_range = json.load(read)
                    break
                except json.decoder.JSONDecodeError:
                    time.sleep(0.2)
            else:
                raise RuntimeError(f"Failed to load {prior_space_pid} after retries.")
        else:
            for attempt in range(10):
                try:
                    with open("prior_space.json", "r") as read:
                        param_range = json.load(read)
                    break
                except json.decoder.JSONDecodeError:
                    time.sleep(0.2)
            else:
                raise RuntimeError("Failed to load prior_space.json after retries.")
        
        para_space = [Real(low, high, name=prior_name) for prior_name, (low, high) in param_range.items()]
        param_names = [dim.name for dim in para_space]
        seed_params = [variable_params[k] for k in param_names]

        # -- If not multi-mode: do just the single template simulation
        if Simulation_params['metric'] != 'TF':
            seed_result = self.build_circuit(seed_params, build_tf, json_config, csv_path, simulation_val, prior_space_pid)
            tf_vector = None
        else:
            tf_vector, seed_result = self.build_circuit(seed_params, build_tf, json_config, csv_path, simulation_val, prior_space_pid)

        log_optimizer_results(
            x_iters=[seed_params],
            y_vals=[-seed_result],
            param_batch=[seed_params],
            result_batch=[seed_result],
            param_names=param_names,
            iteration_start=0,
            batch_size=1,
            penalty_batch=None,
            transfer_vector_batch=[tf_vector],
            csv_path = csv_path
        )

def run_rsoft_sim(args):
    from RSoftSimulation import RSoftSim  
    from Functions import overwrite_template_val

    params, build_tf, json_config, csv_path, simulation_val, prior_space_pid = args
    # this needs to be defined here as well or 
    # else some paras won't be updated for some reason???
    overwrite_template_val(json_config)
    sim = RSoftSim()
    sim.generate_core_positions()
    return sim.build_circuit(params, build_tf, json_config, csv_path, simulation_val, prior_space_pid)

#############################################################################################################################################################################
"""
Below is the current workflow for building the transfer matrix for a PL. Above is blindly optimizing a single core in the PL using 
it's throughput as the metric.

CURRENT: simultaneously chooses pairs of modes to run RSoft with to develop the transfer matrix faster than running individually.
Multiprocessing to occur after the parameter is chosen that sequentially injects individual modes to build the transfer matrix. 
Multiprocessing that picks a new set of parameters, injects individual LP modes to build the transfer matrix, 
and then suggest new parameters to test
"""
import copy

def multiple_mode_tf(arg_list):

    sim_val, m, rm, params, gridding = arg_list
    '''
    TO DO: fix up the gridding part of this code.
    '''
    if gridding:
        # sim_val = copy.deepcopy(simulation_val_list)
        # sim_val["grid_size"] = gr
        # sim_val["grid_size_y"] = gr
        # sim_val["launch_type"] = LaunchType.MM
        # sim_val["launch_tilt"] = 0
        # sim_val["launch_mode_radial"] = "*"

        # CONTINUE HERE

        # specify MS core properties
        # assign_core_properties(sim_val)

        # dump configuration paras into json for use later
        pid = os.getpid()
        # prior_space_pid = f"Grid_{gr}_prior_space_{pid}.json"
        # code_config = f"Grid_{gr}_launch_config.json"
        # optimizer_result = f"Grid_{gr}_Optimizer_Result.csv"
        # param_num = f"Grid {gr}"
    else:
        sim_val = copy.deepcopy(sim_val)
        sim_val["launch_mode"] = m
        sim_val["launch_mode_radial"] = rm

        # write core diameter properties to simulation_val and set special core properties to None for SKOPT to overwrite
        assign_core_properties(sim_val)
        core_to_monitor = sim_val["core_to_monitor"]

        # dynamically load parameters to vary
        param_names = list(variable_params.keys())

        # Assign the new params directly to the special core
        for pname, pval in zip(param_names, params):
            sim_val[f"core_{core_to_monitor}"][pname] = pval
            variable_params[pname] = pval

        # dump configuration paras into json for use later
        # pid is needed to avoid cross-talking between files created/used in multiprocessing tasks
        pid = os.getpid()
        # prior space identifier
        prior_space_pid = f"LP_{m}{rm}_prior_space_{pid}.json"
        # launch config identifier
        code_config = f"LP_{m}{rm}_launch_config_{pid}.json"
        # csv to store results later
        optimizer_result = f"LP_{m}{rm}_Optimizer_Result_{pid}.csv"
        param_num = f"LP{m}{rm}"
    
    # dump launch config profile into simulation_val
    with open(code_config, "w") as launch_config:
        json.dump(sim_val, launch_config, indent = 2)

    custom_priors = {
                "core_delta": (0.0, 0.02),
                "core_diam": (1.0, 20.0)
                }

    build_tf = True

    sim = RSoftSim()
    sim.init_priors(prior_space_pid, build_tf, custom_priors)

    # run simulation
    sim.RunRSoft(sim_val, prior_space_pid, csv_path = optimizer_result, json_config = code_config, build_tf = build_tf)

    # Load results
    best_para_log = f"best_params_log_{pid}.csv"
    data = pd.read_csv(best_para_log)

    # Extract parameter names
    param_names = list(custom_priors.keys())

    # Identify and extract TF columns
    tf_columns = [col for col in data.columns if col.startswith("TF_")]
    if tf_columns:
        tf_vectors = data[tf_columns].values.tolist()  
    else:
        tf_vectors = None

    # Call plotting function
    plotting_optimizer_results(data, param_names, tf=tf_vectors, plot= False)
    return (param_num, tf_vectors)

def run_tf_multproc(params, simulation_val, mode_vals, radial_mode_vals, gridding = False,): 

    tf_list = []
    if not gridding:
        # initialise global parent argument list. This creates multiple instances of args_list based on the length of
        # mode_vals or radial_mode_vals. i.e. 
        # [(simulation_val, 0, 1, params, False),(simulation_val, 1, 1, params, False),(simulation_val, -1, 1, params, False), .....]
        args_list = [(simulation_val, m, rm, params, gridding) for m, rm in zip(mode_vals, radial_mode_vals)] 
    else:
        # run gridding determination
        grid_size_list = np.arange(0.1, 2.1, 0.1)
        grid_size_range = np.linspace(0.1, 2.0, len(grid_size_list))

        if simulation_val["launch_type"] != "LAUNCH_MULTIMODE" and Launch_params["launch_random_set"] != 0:
            raise Exception("Launch type must be multimode with a fixed random set when determining optimal grid sizes!")
        
        # initialise global parent argument list
        args_list = [(simulation_val, gr, params, gridding) for gr in grid_size_range] 

    # note 'spawn' means Python will start n separate processes each with their own memory of global variables. 
    # You need to define any changes within their own instance or else NOTHING changes.
    with mp.get_context("spawn").Pool(processes=6) as pool:
        # supply multiple_mode_tf with the arguments required to run. 
        # Since the number of processes match the length of the mode_vals/radial_mode_vals
        # then each worker gets an instance of arg_list. i.e. arg_list[i]
        results = pool.map(multiple_mode_tf, args_list)

    for param, result in results:
        print(f"Param: {param}, Result: {result}")
        tf_list.append(results)
        
    if gridding:
        return grid_size_range, tf_list
    else:
        return tf_list

def run_all_modes_for_params(params, simulation_val, mode_vals, radial_mode_vals, gridding=False):
    """
    For a single param vector, run all modes and aggregate result.
    params: list of optimized parameter values (from skopt)
    simulation_val: base simulation config
    mode_list: list of (m, rm) tuples
    gridding: bool, determines whether to run gridding determination or not
    """

    if gridding:
        # run gridding determination
        grid_size_range, tf_list = run_tf_multproc(params, simulation_val, mode_vals, radial_mode_vals, params, gridding)
        return grid_size_range, tf_list
    else:
        # multiprocessing in here, we only want multiprocessing in this function and not in main_optimizer!!!!
        tf_list = run_tf_multproc(params, simulation_val, mode_vals, radial_mode_vals, gridding)
    
    # Aggregate result (compute scalar loss/metric for this parameter vector)
    core_to_monitor = simulation_val["core_to_monitor"]
    modes_to_monitor = ["LP01"]  

    loss = mode_selective_tf_matrix_metric(
        tf_list,
        core_to_monitor=core_to_monitor,
        modes_to_monitor=modes_to_monitor
    )
    return loss

def main_optimizer(prior_space_pid, simulation_val, mode_vals, radial_mode_vals, gridding, total_calls, simulate_tf_metric = True):
    # create image folder in results location
    images_dir = Path(os.path.expanduser("~/Desktop/Results/Images"))
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load prior space
    for attempt in range(10):
        try:
            with open(prior_space_pid, "r") as read:
                param_range = json.load(read)
            break
        except json.decoder.JSONDecodeError:
            time.sleep(0.2)
    else:
        raise RuntimeError(f"Failed to load {prior_space_pid} after retries.")
    
    # read in prior space to inform optimiser of the dimensions
    para_space = [Real(low, high, name=prior_name) for prior_name, (low, high) in param_range.items()]
    
    # initialise optimiser
    opt = Optimizer(
        dimensions=para_space,
        base_estimator="GP",
        acq_func="EI",
        random_state=42
    )
    # if true, run optimisation testing the loss metric
    if simulate_tf_metric:
        all_results = []
        for batch_idx in range(total_calls):
            # ask for 1 set of parameter vectors only to prevent daemonic process having children 
            param_batch = opt.ask()
            # run simulation
            result_batch = run_all_modes_for_params(param_batch, simulation_val, mode_vals, radial_mode_vals, gridding = gridding)
            # tell optimiser the performance of the chosen parameters
            opt.tell(param_batch, result_batch)
            # log iteration of parameters, store for later use
            print(f"Iteration {batch_idx + 1}: {result_batch}")
            all_results.append({'params': param_batch, 'result': result_batch, 'Iteration': batch_idx + 1})
        
        return all_results
    # if false, run tf code for the template parameters
    else:
        param_names = ["core_delta", "core_diam"]
        params = [variable_params[k] for k in param_names]
        tf_list = run_tf_multproc(params, simulation_val, mode_vals, radial_mode_vals, gridding)
        return tf_list