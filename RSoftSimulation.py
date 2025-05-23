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

    def init_priors(self, custom = None):
        base_priors = {}
        if custom:
            base_priors.update(custom)
        self.prior_space = base_priors
        with open("prior_space.json", "w") as write:
            json.dump(self.prior_space, write)
        return base_priors

    def generate_core_positions(self):
        SimParam = Simulation_params
        core_sep = fixed_params["core_sep"]
        grid_type = SimParam["grid_type"]
        if grid_type == "Hex":
            """
            Generate hexagonal core coordinates and store internally.
            """
            core_num = SimParam["core_num"]
            if core_num % 2 == 0:
                raise ValueError(f"The number of cores must be odd to perfectly fit inside the hex grid. Received: {core_num}")

            row_numbers = [number_rows(core_num)]
            for row_num in row_numbers:
                hcoord, vcoord = generate_hex_grid(row_num, core_sep)
                self.core_positions = list(zip(hcoord, vcoord))
            with open("core_positions.json", "w") as g:
                json.dump(self.core_positions, g)

        if grid_type == "Pent":
            """
            Generate pentagon core coordinates and store internally.
            """
            estimated_radius = estimate_pentagon_radius(core_num,core_sep)
            hcoord, vcoord= generate_filled_pentagon_grid(estimated_radius, core_sep)
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

    def RunRSoftSim(self, name_tag, fixed, vars, fixed_length, param_range):
        filename = f"{name_tag}.ind"
        prefix   = f"prefix={name_tag}"
        folder   = f"Sim_{name_tag}"

        # Set up folders
        user_home = os.path.expanduser("~")
        desktop_path = os.path.join(user_home, "Desktop")
        results_root = os.path.join(desktop_path, "Results")
        results_folder = os.path.join(results_root, folder)
        os.makedirs(results_folder, exist_ok=True)

        # Run RSoft simulation
        try:
            subprocess.run(
                ["bsimw32", filename, prefix, "wait=0"],
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
            if file.startswith(name_tag) or file == filename:
                shutil.move(file, os.path.join(results_folder, file))

        # Safely access the .mon file in its new location
        mon_path = Path(results_folder) / f"{name_tag}.mon"
        timeout = 10
        t_start = time.time()
        while not mon_path.exists():
            if time.time() - t_start > timeout:
                raise FileNotFoundError(f"{mon_path} not found within {timeout} seconds after simulation.")
            time.sleep(0.1)

        # Read .mon file from moved location
        uf = RSoftUserFunction()
        uf.read(str(mon_path))
        x_all, y_all, z_all = uf.get_arrays()
        num_monitors = z_all.shape[1]

        # Write throughput CSV to same folder
        csv_tag = f"Throughput_{name_tag}.csv"
        csv_path = Path(results_folder) / csv_tag

        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            header = ["x"] + [f"Monitor_{i}" for i in range(num_monitors)]
            writer.writerow(header)
            for i in range(z_all.shape[0]):
                row = [x_all[i]] + [np.real(z_all[i, j]) for j in range(z_all.shape[1])]
                writer.writerow(row)

        if Simulation_params["metric"] == 'TH':
            return -throughput_metric(csv_path, fixed_length,
                                       fixed, vars, 
                                       param_range, Simulation_params["mode_selective"])
        if Simulation_params['metric'] == 'MS':
            return - mode_selective_metric(csv_path, Simulation_params["core_to_monitor"], 
                                           f"LP{Launch_params['launch_mode']}{Launch_params['launch_mode_radial']}")

    def build_circuit(self, params): # maybe put this into its own function. Make it universal.
        """
        Create the design file using template.py and 
        write to separate .ind file. Also contains function to run BeamProp
        and scikit Optimize
        """
        with open("launch_config.json", "r") as launch_config:
            simulation_val = json.load(launch_config)
        
        for key, val in simulation_val.items():
            Launch_params[key] = val

        # load prior space
        with open("prior_space.json", "r") as read:
            param_range = json.load(read)
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
            raise Exception("Taper Length defined as neither being fixed nor variable. Please specify 'Taper_L' in template.fixed_params or template.variable_params.")

        if "Taper_L" in fixed:
            Taper_L = fixed["Taper_L"]
            fixed_length = True
        else:
            Taper_L = vars["Taper_L"]
        
        taper = vars["taper"]
        core_num = sim_param["core_num"]
        core_name = [f"core_{n}" for n in range(1, core_num+1)]
        structure = Simulation_params["Structure"]

        cladd_diam = fixed["MCFCladd"]
        cladding_beg_dims = (cladd_diam / taper, cladd_diam / taper)
        cladding_end_dims = (cladd_diam , cladd_diam)

        core_beg_dims_list = []
        core_end_dims_list = []

        for core_key in core_name:
            if core_key == f"core_{Simulation_params['core_to_monitor']}":
                core_diam = variable_params["core_diam"]
            else:
                core_diam = core_params[core_key]["core_diam"]
            
            core_beg_dims_list.append((core_diam / taper, core_diam / taper))
            core_end_dims_list.append((core_diam , core_diam))

        path_num = 0
        if structure == "Fibre":
            path_num = build_fibre(self.circuit, path_num, self.core_positions, 
                        core_name, Taper_L, tuple(core * taper for core in core_beg_dims_list), core_end_dims_list)

        elif structure == "PL":
            path_num = build_PL(self.circuit, path_num, self.core_positions, 
                    core_name, taper, Taper_L,
                    cladding_beg_dims, cladding_end_dims,
                    core_beg_dims_list, core_end_dims_list)

        name_tag = "_".join(f"{key}_{val:.6f}" for key, val in param_dict.items())
        self.sym["Name"] = name_tag
        self.circuit.write(f"{name_tag}.ind")
        """
        Append all pathway, monitor, and launch field blocks based 
        on launch parameters. 
        """ 
        
        AddHack(name_tag, launch, path_num -1, param_dict)
        '''
        Manual setup to loop through a list of values. Runs the terminal line that will initiate RSoft and will calculate the 
        metric to test.
        All output files will appear in a subfolder on the Desktop (windows)
        '''
        average_throughput = self.RunRSoftSim(name_tag, fixed, 
                                              vars, fixed_length, 
                                              param_range)

        return average_throughput
    
    def MultProc(self):
        images_dir = Path(os.path.expanduser("~/Desktop/Results/Images"))
        images_dir.mkdir(parents=True, exist_ok=True)

        # load prior space
        with open("prior_space.json", "r") as read:
            param_range = json.load(read)
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

        self.sym["Name"] = "MCF_Test"
        seed_result = self.build_circuit(seed_params)
        opt.tell(seed_params, seed_result)
        all_results.append((seed_params, seed_result))
        
        log_optimizer_results(
            x_iters=[seed_params],
            y_vals=[-seed_result],  
            param_batch=[seed_params],
            result_batch=[seed_result],
            param_names=param_names,
            iteration_start=0,
            batch_size=1,
            penalty_batch=None
        )       
        # run optimizer as normal
        for i in range(0, total_calls, batch_size):
            
            # Suggest next batch of points
            param_batch = opt.ask(batch_size)

            # Evaluate in parallel
            ctx = mp.get_context("spawn")
            with ctx.Pool(batch_size) as pool:
                result_batch = pool.map(run_rsoft_sim, param_batch)

            # Feed results back to optimizer
            opt.tell(param_batch, result_batch)
            
            all_results.extend(zip(param_batch, result_batch))

            '''
            save results for plotting/analysis
            '''
            
            # Unpack results
            x_iters = [r[0] for r in all_results]  # parameter sets
            y_vals = [-r[1] for r in all_results]  # throughput values

            param_names = [dim.name for dim in opt.space.dimensions]
            # log chosen values and penalties
            log_optimizer_results(x_iters, y_vals,
                                  param_batch, result_batch,
                                  param_names, iteration_start=i,
                                  batch_size = batch_size)

    def RunRSoft(self, simulate = True):
        '''
        Multiprocessing must to be run outside of a Jupyter cell or it will silently 
        fail/infinitely loop on the first batch
        '''

        # write in the values within simulation_val
        overwrite_template_val()
        
        # generate the positions of the cores. 
        self.generate_core_positions()

        # remove old results
        csv_path = "optimizer_results.csv"
        if os.path.exists(csv_path):
            os.remove(csv_path)

        if simulate:
            self.MultProc()
        else:
            # load prior space
            with open("prior_space.json", "r") as read:
                param_range = json.load(read)
            para_space = [Real(low, high, name=prior_name) for prior_name, (low, high) in param_range.items()]
            param_names = [dim.name for dim in para_space]
            seed_params = [variable_params[k] for k in param_names]

            # build circuit with template/overwritten values and run a single simulation
            self.build_circuit(seed_params)

def run_rsoft_sim(params):
    from RSoftSimulation import RSoftSim  
    from Functions import overwrite_template_val

    # this needs to be defined here as well or else some paras won't be updated for some reason???
    overwrite_template_val()
    sim = RSoftSim()
    sim.generate_core_positions()
    return sim.build_circuit(params)