import numpy as np, os, shutil, csv, matplotlib.pyplot as plt
import subprocess, json, Template
from skopt import gp_minimize, Optimizer
from skopt.space import Real
from skopt.utils import dump
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Circuit_Properties import *
from Functions import *
from HexProperties import *
from rstools import RSoftUserFunction, RSoftCircuit

class RSoftSim:
    def __init__(self):
        # self.name = name
        self.sym = {}
        # self.core_log = {}

        # # Generate Fibre Parameters
        # self.length = 1000
        # self.Corediam = 8.2
        # self.Claddiam = 150
        # self.acore_taper_ratio = 10
        # self.Core_sep = 100
        # self.core_num = 1
        # self.Core_delta = 0.01
        # self.background_index = 1.456 
        # self.delta =  0.012 # 0.0036
        # self.core_beg_diam = 'Corediam / acore_taper_ratio'
        # self.cladding_beg_diam = 'Claddiam / acore_taper_ratio'

        # # Cire geometry
        # self.grid_type = "Hex"
        # self.core_positions = [] # store tuples of (x, y)

        # # Multiprocessing Parameters
        # self.num_paras = 96
        # self.batch_number = 12

        # # Simulation Setup
        # self.Dx = 0.1
        # self.Dy = self.Dx
        # self.Dz = 0.2
        # self.Phase = 0
        # self.boundary_gap_x = 10
        # self.boundary_gap_y = self.boundary_gap_x
        # self.boundary_gap_z = 0
        # self.bpm_pathway = 1
        # self.bpm_pathway_monitor = self.bpm_pathway
        # self.sim_tool = Sim_tool.BP
        # self.width = 5
        # self.height = self.width
        # self.H = self.height
        # self.grid_uniform = 0
        # self.eim = 0
        # self.polarization = 0
        # self.free_space_wavelength = 1.55
        # self.k0 = 2 * np.pi / self.free_space_wavelength
        # self.slice_display_mode = "DISPLAY_CONTOURMAPXZ"
        # self.slice_position_z = 100
        
        # # CAD Stuff if you like the GUI or just want to check ind files
        # self.cad_aspectratio_z = -1
        # self.cad_aspectratio_y = -1
        # self.cad_aspectratio_x = -1

        # # Launch Properties
        # self.monitor_type = Monitor_Prop.FIBRE_MODE_POWER 
        # self.comp = Monitor_comp.BOTH
        # self.launch_port = 0
        # self.launch_type = ""
        # self.launch_file = ""
        # self.launch_random_set = 0 # if self.launch_type == LaunchType.SM else 1
        # self.launch_tilt = 1 # if self.launch_type == LaunchType.SM else 0
        # self.launch_align_file = 1
        # self.launch_mode = 0 # if self.launch_type == LaunchType.SM else "*"
        # self.launch_mode_radial = 1 # if self.launch_type == LaunchType.SM else "*"
        # self.launch_normalization = 1
        # self.grid_size = self.Dx
        # self.grid_size_y = self.Dy
        # self.step_size = self.Dz
        # self.core_monitor_width = self.Corediam * 1.1
        # self.core_monitor_height = self.core_monitor_width 
        # self.cladd_monitor_width = self.Claddiam * 1.1
        # self.cladd_monitor_height = self.cladd_monitor_width 
        # self.launch_field_height = self.core_beg_diam
        # self.launch_field_width = self.core_beg_diam

        # self.structure = Struct_type.FIBRE
    
    # def taper_pos(self, x):
    #     taper = f"{x} / acore_taper_ratio"
    #     return taper

    # def variable_parameters_dict(self):
    #     return {
    #         "Name": self.name,
    #         "Length": self.length,
    #         "Corediam": self.Corediam,
    #         "Claddiam": self.Claddiam,
    #         "Core_sep": self.Core_sep,
    #         "acore_taper_ratio": self.acore_taper_ratio,
    #         "core_beg_diam": self.core_beg_diam,
    #         "cladding_beg_diam": self.cladding_beg_diam,
    #         "core_num": self.core_num,
    #         "Core_delta": self.Core_delta,
    #         "background_index": self.background_index,
    #         "delta": self.delta,
    #         "core_monitor_width": self.core_monitor_width,
    #         "core_monitor_height": self.core_monitor_height,
    #         "cladd_monitor_width": self.cladd_monitor_width,
    #         "cladd_monitor_height": self.cladd_monitor_height,
    #         "launch_field_height": self.launch_field_height,
    #         "launch_field_width": self.launch_field_width
    #     }

    # def fixed_parameters_dict(self):
    #     return {
    #         "Dx": self.Dx,
    #         "Dy": self.Dy,
    #         "Dz": self.Dz,
    #         "Phase": self.Phase,
    #         "boundary_gap_x": self.boundary_gap_x,
    #         "boundary_gap_y": self.boundary_gap_y,
    #         "boundary_gap_z": self.boundary_gap_z,
    #         "bpm_pathway": self.bpm_pathway,
    #         "bpm_pathway_monitor": self.bpm_pathway_monitor,
    #         "sym_tool": self.sim_tool,
    #         "width": self.width,
    #         "height": self.height,
    #         "H": self.H,
    #         "grid_uniform": self.grid_uniform,
    #         "eim": self.eim,
    #         "polarization": self.polarization,
    #         "free_space_wavelength": self.free_space_wavelength,
    #         "k0": self.k0,
    #         "slice_display_mode": self.slice_display_mode,
    #         "slice_position_z": self.slice_position_z,
    #         "cad_aspectratio_z": self.cad_aspectratio_z,
    #         "cad_aspectratio_y": self.cad_aspectratio_y,
    #         "cad_aspectratio_x": self.cad_aspectratio_x,
    #         "grid_size": self.grid_size,
    #         "grid_size_y": self.grid_size_y,
    #         "step_size": self.step_size,
    #         "structure": self.structure,
    #         "num_paras": self.num_paras,
    #         "batch_number": self.batch_number
    #     }

    # def launch_parameters_dict(self):
    #     return {
    #         "monitor_type": self.monitor_type,
    #         "comp": self.comp,
    #         "launch_tilt": self.launch_tilt,
    #         "launch_port": self.launch_port,
    #         "launch_align_file": self.launch_align_file,
    #         "launch_mode": self.launch_mode,
    #         "launch_file": self.launch_file,
    #         "launch_type": self.launch_type,
    #         "launch_random_set": self.launch_random_set,
    #         "launch_mode_radial": self.launch_mode_radial,
    #         "launch_normalization": self.launch_normalization,
    #         "core_monitor_width": self.core_monitor_width,
    #         "core_monitor_height": self.core_monitor_height,
    #         "cladd_monitor_width": self.cladd_monitor_width,
    #         "cladd_monitor_height": self.cladd_monitor_height,
    #         "launch_field_height": self.launch_field_height,
    #         "launch_field_width": self.launch_field_width
    #     }

    def init_priors(self, custom = None):
        base_priors = {}
        if custom:
            base_priors.update(custom)
        self.prior_space = base_priors
        with open("prior_space.json", "w") as write:
            json.dump(self.prior_space, write)
        return base_priors
    
    # def load_parameters(self):
    #     """
    #     Load variable, fixed, and launch parameters from Template.py
    #     and build symbol dictionary containing ONLY the parameters that
    #     RSoft needs.
    #     """
    #     self.sym = {**Template.RSoft_params, 
    #                 **Template.Launch_params}

    def generate_core_positions(self):
        SimParam = Template.Simulation_params
        core_sep = Template.fixed_params["core_sep"]
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

    def RunRSoftSim(self, name_tag):
        filename = f"{name_tag}.ind"
        prefix   = f"prefix={name_tag}"
        folder   = f"Sim_{name_tag}"

        # Set up folders
        user_home = os.path.expanduser("~")
        desktop_path = os.path.join(user_home, "Desktop")
        results_root = os.path.join(desktop_path, "Results")
        results_folder = os.path.join(results_root, folder)
        os.makedirs(results_folder, exist_ok=True)
        # Run RSoft simulation, if multiprocessing takes too long time it out and raise error
            # there should be a way to make the outputs automatically be placed in a certain directory
        subprocess.run(["bsimw32", filename, prefix, "wait=0"], check=True) # put this else where

        # Read .mon files
        uf = RSoftUserFunction()
        uf.read(f"{name_tag}.mon")
        x_all, y_all, z_all = uf.get_arrays()
        num_monitors = z_all.shape[1]
        # x_all: x-coordinates of the sampled data
        # y_all: y-coordinates of the sampled data
        # z_all: field data sampled at each point for each monitor (len(x), num_monitors)

        csv_tag = f"Throughput_{name_tag}.csv"
        with open(csv_tag, mode="w", newline="") as file:
            writer = csv.writer(file)
            header = ["x"] + [f"Monitor_{i}" for i in range(num_monitors)]
            writer.writerow(header)
            for i in range(z_all.shape[0]):
                row = [x_all[i]] + [np.real(z_all[i, j]) for j in range(z_all.shape[1])]
                writer.writerow(row)

        df = pd.read_csv(csv_tag)
        monitor_columns = [col for col in df.columns if col.startswith("Monitor_")]
        average = (df[monitor_columns].mean().sum())/Template.Simulation_params["core_num"]

        # Move all matching files to results folder
        for file in os.listdir():
            if file.startswith(name_tag) or file == filename or file == csv_tag:
                shutil.move(file, os.path.join(results_folder, file))
        return -average

    def build_circuit(self, params):
        """
        Create the design file using Template.py and 
        write to separate .ind file. Also contains function to run BeamProp
        and scikit Optimize
        """
        # write in the values within simulation_val
        with open("launch_config.json", "r") as launch_config:
            simulation_val = json.load(launch_config)
        for key, val in simulation_val.items():
            Template.Launch_params[key] = val

        # load prior space
        with open("prior_space.json", "r") as read:
            param_range = json.load(read)
        para_space = [Real(low, high, name=prior_name) for prior_name, (low, high) in param_range.items()]
        param_dict = {dim.name: val for dim, val in zip(para_space, params)}
        # update template file with chosen values from scikit.Optimize()
        Template.variable_params.update(param_dict)

        self.circuit = RSoftCircuit()

        """
        Load variable, fixed, and launch parameters from Template.py
        and build symbol dictionary containing ONLY the parameters that
        RSoft needs.
        """
        self.sym = {**Template.RSoft_params, 
                    **Template.Launch_params}
        for key, val in self.sym.items():
            self.circuit.set_symbol(key, val)
        
        # extract taper ratio, length, core number, 
        # core and cladding diameter
        fixed = Template.fixed_params
        vars = Template.variable_params
        sim_param = Template.Simulation_params
        launch = Template.Launch_params

        taper = vars["taper"]
        Taper_L = vars["Taper_L"]
        core_diam = fixed["core_diam"]
        cladd_diam = fixed["MCFCladd"]
        core_num = sim_param["core_num"]

        name = self.sym["Name"]
        core_beg_dims = (core_diam / taper, core_diam / taper)
        core_end_dims = (core_diam , core_diam)
        cladding_beg_dims = (cladd_diam / taper, cladd_diam / taper)
        cladding_end_dims = (cladd_diam , cladd_diam)
        core_name = [f"core_{n+1:02}" for n in range(core_num)]

        path_num = 0

        if core_num == 1:
            for j, (x, y) in enumerate(self.core_positions):
                path_num += 1
                core = self.circuit.add_segment(
                    position=(x / taper, y / taper, 0),
                    offset=(x, y, Taper_L),
                    dimensions= core_beg_dims,
                    dimensions_end= core_end_dims
                )
                core.set_name(core_name[j])

        else:   
            cladding = self.circuit.add_segment(
                    position=(0, 0, 0),
                    offset=(0, 0, Taper_L),
                    dimensions = cladding_beg_dims,
                    dimensions_end = cladding_end_dims
                    )
            cladding.set_name("Super Cladding")
            path_num += 1
            
            for j, (x, y) in enumerate(self.core_positions):
                path_num += 1
                core = self.circuit.add_segment(
                    position=(x / taper, y / taper, 0),
                    offset=(x, y, Taper_L),
                    dimensions = core_beg_dims,
                    dimensions_end = core_end_dims
                )
                core.set_name(core_name[j])
        
        name_tag = "_".join(f"{key}_{val:.6f}" for key, val in param_dict.items())
        self.sym["Name"] = name_tag
        self.circuit.write(f"{name_tag}.ind")
        """
        Append all pathway, monitor, and launch field blocks based 
        on launch parameters. 

        TO DO: include stray hacking parts that appear in MultiProcessing.py
        """ 
        
        AddHack(name_tag, launch, path_num -1, param_dict)
        '''
        Manual setup to loop through a list of values
        Runs the terminal line that will initiate RSoft. 
        All output files will appear in a subfolder on the Desktop (windows)
        '''
        average_throughput = self.RunRSoftSim(name_tag)
        return average_throughput

    # initialises multiprocessing function that runs the RSoft simulation
    def mp_eval(self, params):
        # just for tracking purposes if running from the terminal
        print(f"[PID {os.getpid()}] Starting Rsoft with {params}")
        return self.build_circuit(params)
    
    def MultProc(self):
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
        sim_param = Template.Simulation_params

        # the multiprocessing in all its glory. 
        # The first if __name__ == "__main__" is required 
         
        # how many values in each parameter space to run simulation with
        total_calls = sim_param["num_paras"]
        # this is the number of points to sample simultaneously. 
        # Increase to cycle through prior space quicker at the cost of CPU computation
        batch_size = sim_param["batch_num"]
        all_results = []

        for i in range(0, total_calls, batch_size):
            
            # Suggest next batch of points
            param_batch = opt.ask(batch_size)

            # Evaluate in parallel
            ctx = mp.get_context("spawn")
            with ctx.Pool(batch_size) as pool:
                result_batch = pool.map(self.mp_eval, param_batch)

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
    
    # def print_core_log(self):
    #     for core_name, core_info in self.core_log.items():
    #         print(f"{core_name}")
    #         for key, val in core_info.items():
    #             print(f"{key}: {val}")
    #         print("") 

    def RunRSoft(self, simulate = True):
        '''
        Multiprocessing must to be run outside of a Jupyter cell or it will silently 
        fail/infinitely loop on the first batch
        '''

        # # initialise prior space
        # self.init_priors(custom_priors)
        # # save all parameters to JSON files and load them
        # self.save_all_json()
        # # load prior spaces in json format
        # self.load_json_parameters()

        # generate the positions of the cores. 
        self.generate_core_positions()
        
        # self.build_circuit()
        if simulate:
            self.MultProc()
            # subprocess.run(["python", "Multiprocessing.py"], check=True)