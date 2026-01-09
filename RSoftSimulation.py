import numpy as np, os, shutil, csv
import subprocess, json, time
from pathlib import Path
from skopt import Optimizer
from skopt.space import Real, Categorical
from skopt.utils import dump
import multiprocessing as mp
import matplotlib.gridspec as gridspec
import seaborn as sns


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
            
            # if core_num % 2 == 0:
            #     raise ValueError(f"The number of cores must be odd to perfectly fit inside the hex grid. Received: {core_num}")

            # # row_numbers = [number_rows(core_num)]
            # # for row_num in row_numbers:
            #     # hcoord, vcoord = generate_hex_grid(row_num, core_sep, include_centre = SimParam["plot_centre_core"])
            #     # self.core_positions = list(zip(hcoord, vcoord))
            # hcoord, vcoord = generate_hex_ring_grid(fixed_params["MCFCladd"]/2, core_sep, SimParam["core_num"], include_center = SimParam["plot_centre_core"])
            row_num, excess = number_rows(core_num)
            hcoord, vcoord = old_generate_hex_grid(row_num, fixed_params["core_sep"], include_centre = Simulation_params["plot_centre_core"])
            
            # if core_num == 19:
            #     reorder_indices = [9, 10, 14, 13, 8, 4, 5, 11, 15, 18, 17, 16, 12, 7, 3, 0, 1, 2, 6]
            # elif core_num == 7:
            #     if Simulation_params["plot_centre_core"]:
            #         reorder_indices = [3,4,6,5,2,0,1]
            #         x, y = np.array(hcoord)[reorder_indices], np.array(vcoord)[reorder_indices]
            #     else:
            #         reorder_indices = [0,1,2,3,4,5]
            #         x, y = np.array(hcoord), np.array(vcoord)
            if Simulation_params["plot_centre_core"]:
                if 19 < core_num <= 37:
                    reorder_index = [18, 19, 25, 24, 17, 11, 12,
                                    20, 26, 31, 30, 29, 23, 16, 10, 5, 6, 7, 13,
                                    21, 27, 32, 36, 35, 34, 33, 28, 22, 15, 9, 4, 0, 1, 2, 3, 8, 14]
                elif 7 < core_num <= 19:
                    reorder_index = [9, 10, 14, 13, 8, 4, 5,
                                11, 15, 18, 17, 16, 12, 7, 3, 0, 1, 2, 6]
                elif core_num <= 7:
                    reorder_index = [3,4,6,5,2,0,1]
            else:
                if 19 < core_num <= 37:
                    reorder_index = [18, 19, 25, 24, 17, 11, 12,
                                    20, 26, 31, 30, 29, 23, 16, 10, 5, 6, 7, 13,
                                    21, 27, 32, 35, 34, 33, 28, 22, 15, 9, 4, 0, 1, 2, 3, 8, 14]
                elif 7 < core_num <= 19:
                    reorder_index = [9, 10, 14, 13, 8, 4, 5,
                            11, 15, 17, 16, 12, 7, 3, 0, 1, 2, 6]
                elif core_num <= 7:
                    reorder_index = [3,5,4,2,0,1]

            xcoord_og, ycoord_og, xcoord_relist, ycoord_relist = plot_excess(excess, hcoord, vcoord, reorder_index)
            # # code to skip cores if desired
            if Simulation_params["skip_core"] is not None:
                x_val, y_val = [], []
                for i, (xval, yval) in enumerate(zip(xcoord_relist, ycoord_relist)):
                    for j in Simulation_params["skip_core"]:
                        if i != j:
                            x_val.append(xval)
                            y_val.append(yval)
                self.core_positions = list(zip(x_val, y_val))
                self.cladd_positions = list(zip(xcoord_relist, ycoord_relist))
                with open("cladding_position.json", "w") as cladd_positioning:
                    json.dump(self.cladd_positions, cladd_positioning)
            else:
                self.core_positions = list(zip(xcoord_relist, ycoord_relist))
                self.cladd_positions = None
            
            with open("core_positions.json", "w") as g:
                json.dump(self.core_positions, g)

        if grid_type == "Pent":
            """
            Generate pentagon core coordinates and store internally.
            """
            row_num, excess = number_rows(core_num, grid_type="pent")
            hcoord, vcoord= generate_pent_grid(row_num, grid_spacing=fixed_params["core_sep"], include_centre=Simulation_params["plot_centre_core"])
            if Simulation_params["plot_centre_core"]:
                if core_num <= 6:
                    reorder_index_pent = [0, 5, 1, 2, 3, 4]
                elif 6 < core_num <= 16:
                    reorder_index_pent = [0, 1, 2, 3, 4, 5,
                                        14, 15, 6, 7, 8, 9, 10, 11, 12, 13]
                elif 16 < core_num <= 31:
                    reorder_index_pent = [0, 1, 2, 3, 4, 5,
                                        14, 15, 6, 7, 8, 9, 10, 11, 12, 13,
                                        27, 28, 29, 30, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26]
            else:
                if core_num <= 6:
                    reorder_index_pent = [0, 1, 2, 3, 4]
                elif 6 < core_num <= 16:
                    reorder_index_pent = [4, 0, 1, 2, 3,
                                        13, 14, 5, 6, 7, 8, 9, 10, 11, 12]
                elif 16< core_num <= 31:
                    reorder_index_pent = [4, 0, 1, 2, 3,
                                        13, 14, 15, 5, 6, 7, 8, 9, 10, 11, 12,
                                        27, 28, 29, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

            xcoord_og, ycoord_og, xcoord_relist, ycoord_relist = plot_excess(excess, hcoord, vcoord, reorder_index_pent)
            self.core_positions = list(zip(xcoord_relist, ycoord_relist))
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

    def RunRSoftSim(self, name_tag, femsim_name_tag, fixed, vars, fixed_length, param_range, simulation_val, csv_path, json_config, prior_space_pid):
        filename = f"{name_tag}.ind"
        filename_FS = f"{femsim_name_tag}.ind"
        sim_tool = simulation_val.get("sim_tool", RSoft_params["sim_tool"])
        iter_number = simulation_val["iter_num"]
        # Run RSoft simulation
        if sim_tool == "ST_BEAMPROP":
            prefix_BP   = f"prefix={name_tag}"
            folder_BP   = f"BP_SimulationNum_{iter_number}"
            prefix_FS = f"prefix={femsim_name_tag}"

            results_folder = create_folders(folder_BP, "Desktop")
            # self.sym["Result_directory"] = results_folder

            # we want to copy the important files in Onedrive for backup purposes
            onedrive_results_folder = create_folders(folder_BP, "Onedrive")

            try:
                subprocess.run(
                    [r"C:\Synopsys\PhotonicSolutions\2024.09-SP2-3\RSoft\bin\femsim.exe", filename_FS, prefix_FS, "wait=0"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                subprocess.run(
                    [r"C:\Synopsys\PhotonicSolutions\2024.09-SP2-3\RSoft\bin\bsimw32.exe", filename, prefix_BP, "wait=0"],
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
        
        # invoke special identifier for csv files to prevent multiprocessing from overwriting the same file
        pid_csv = os.getpid()
        # Move all output files immediately after simulation
        
        # files to copy to Onedrive
        onedrive_filename = name_tag + ".ind"
        onedrive_filename_results = name_tag + "_mon.dat"
        onedrive_femsim_filename_results = femsim_name_tag + ".ind"
        onedrive_neff_csv_path = Path(onedrive_results_folder) / f"Guided Modes_{pid_csv}.csv"
        neff_csv_path = Path(results_folder) / f"Guided Modes_{pid_csv}.csv"

        file_extensions_to_copy = [
           onedrive_filename, onedrive_filename_results, onedrive_femsim_filename_results,onedrive_neff_csv_path
        ]

        files_to_move = [
            filename,
            filename_FS,
            femsim_name_tag,
            csv_path,
            json_config,
            prior_space_pid,
            neff_csv_path]
        
        # Normalize to basenames in case paths are used
        files_to_move = [os.path.basename(f) for f in files_to_move]
        file_extensions_to_move = [os.path.basename(k) for k in file_extensions_to_copy]

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
            # mon_path = Path(results_folder) / f"{name_tag}_mon.dat"
            # nef_path = Path(results_folder) / f"{femsim_name_tag}.nef"
            mon_path = Path(f"{name_tag}_mon.dat")
            nef_path = Path(f"{femsim_name_tag}.nef")

            timeout = 10
            t_start = time.time()
            while not mon_path.exists():
                if time.time() - t_start > timeout:
                    raise FileNotFoundError(f"{mon_path} not found within {timeout} seconds after simulation.")
                time.sleep(0.1)
            while not nef_path.exists():
                if time.time() - t_start > timeout:
                    raise FileNotFoundError(f"{nef_path} not found within {timeout} seconds after simulation.")
                time.sleep(0.1)

        # Read .mon file from moved location
        uf = RSoftUserFunction()
        uf.read(str(mon_path))

        # make guided modes save to output csv
        uf_neff = RSoftUserFunction()
        uf_neff.read(str(nef_path))
        x_neff, y_neff = uf_neff.get_arrays()

        guided_mask = y_neff > fixed_params["cladding_neff"]
        guided_neff = np.array(y_neff[guided_mask])

        with open(neff_csv_path, mode="w", newline="") as f_neff:
            writer = csv.writer(f_neff)
            writer.writerow(["Mode_Index", "n_eff"])
            for idx, nval in enumerate(guided_neff, start=1):
                writer.writerow([idx, nval])

        with open(onedrive_neff_csv_path, mode="w", newline="") as f_neff_od:
            writer = csv.writer(f_neff_od)
            writer.writerow(["Mode_Index", "n_eff"])
            for idx, nval in enumerate(guided_neff, start=1):
                writer.writerow([idx, nval])

        # move files AFTER they have been read
        for file in os.listdir():
            # copy important files to Onedrive
            for l in file_extensions_to_move:
                if file == l:
                    os.makedirs(onedrive_results_folder, exist_ok=True)
                    shutil.copy(file, os.path.join(onedrive_results_folder,file))
                    
            # move everything to the desktop
            if (file in files_to_move or file.startswith(name_tag) or file.startswith(femsim_name_tag)):
                shutil.move(file, os.path.join(results_folder, file))

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
            if Simulation_params["skip_core"] is not None:
                num_monitors = (z_all.shape[1] - (Simulation_params["core_num"] - len(Simulation_params["skip_core"])))
            else:
                num_monitors = (z_all.shape[1] - Simulation_params["core_num"])

            # Write throughput CSV to same folder
            csv_tag = f"Throughput_{name_tag}.csv"
            csv_pathway = Path(results_folder) / csv_tag
            csv_pathway_onedrive = Path(onedrive_results_folder) / csv_tag

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

            with open(csv_pathway_onedrive, mode="w", newline="") as file_onedrvie:
                writer = csv.writer(file_onedrvie)
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
                return transfer_vector, -throughput, results_folder
            else:
                transfer_vector, throughput = transfer_matrix_component(csv_pathway, row)
                return transfer_vector, -throughput, results_folder

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
            
        if Simulation_params["industry_neff_values"] == True:
            neff_val = read_neff_values(Simulation_params["industry_neff_file"])
            para_space = []
            for prior_name, (low, high) in param_range.items():
                if prior_name == "core_neff":
                    para_space.append(Categorical(neff_val, name=prior_name))
                else:
                    para_space.append(Real(low, high, name=prior_name))
        else:
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
            taper = param_dict.get("taper", vars["taper"])
        # MMF_Taper = fixed["MMF_Taper"]
        core_num = sim_param["core_num"]
        core_name = [f"core_{n}" for n in range(1, core_num + 1)]
        structure = Simulation_params["Structure"]

        cladd_diam = fixed["MCFCladd"]
        cladding_beg_dims = (cladd_diam / taper, cladd_diam / taper)
        cladding_end_dims = (cladd_diam , cladd_diam)

        core_beg_dims_list = []
        core_end_dims_list = []

        if sim_param["mode_selective"] == 1:
            for j, core_key in enumerate(core_name, start=1):
                if j == Simulation_params["core_to_monitor"]:
                    # core to be optimized by skopt
                    core_diam = variable_params.get("core_diam")
                    core_neff = variable_params.get("core_neff", fixed_params.get("core_neff"))
                    core_taper = param_dict.get("taper", fixed_params.get("taper"))
                else:
                    # pass
                    # use preconfigured values to specify core parameters
                    core_diam = 6.5 #core_params[core_key]["core_diam"]
                    core_neff = simulation_val.get("core_neff", fixed_params.get("core_neff"))
                    core_taper = param_dict.get("taper", fixed_params.get("taper"))

                # Store dimensions for this core
                core_beg_dims_list.append((core_diam / taper, core_diam / taper))
                core_end_dims_list.append((core_diam, core_diam))

                core_params[core_key]["core_diam"] = core_diam
                core_params[core_key]["neff"] = core_neff
                core_params[core_key]["taper"] = core_taper
        else: 
            for j, core_key in enumerate(core_name, start=1):
                core_diam = core_params[core_key]["core_diam"]
                core_neff = core_params[core_key]["neff"]

                # Store dimensions for this core
                core_beg_dims_list.append((core_diam / taper, core_diam / taper))
                core_end_dims_list.append((core_diam, core_diam))

                core_params[core_key]["core_diam"] = core_diam
                core_params[core_key]["neff"] = core_neff

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
                    simulation_val,self.cladd_positions)
            
        if simulation_val["launch_type"] == LaunchType.SM:
            launch_mode = simulation_val["launch_mode"]
            launch_mode_radial = simulation_val["launch_mode_radial"]
            param_string = "_".join(f"{key}_{val:.6f}" for key, val in param_dict.items())
            name_tag = f"LP{launch_mode}{launch_mode_radial}_{param_string}"
        else:
            grid = simulation_val["grid_size"] # use only when trying to find the optimal gridding to run BeamPROP in.
            name_tag = f"_Grid{grid}".join(f"{key}_{val:.6f}" for key, val in param_dict.items())
        self.sym["Name"] = name_tag
        self.circuit.write(f"{name_tag}.ind")

        # create separate file for FemSIM field determination
        femsim_name_tag = f"FS_{name_tag}"
        self.circuit.write(f"{femsim_name_tag}.ind")

        """
        Append all pathway, monitor, and launch field blocks based 
        on launch parameters. 
        """ 
        
        AddHack(name_tag, femsim_name_tag, launch, 
                path_num - 1, param_dict, simulation_val)
        '''
        Manual setup to loop through a list of values. Runs the terminal line that will initiate RSoft and will calculate the 
        metric to test.
        All output files will appear in a subfolder on the Desktop (windows)
        '''

        if Simulation_params['metric'] != 'TF':
            average_throughput, res_folder = self.RunRSoftSim(name_tag, femsim_name_tag, fixed, 
                                                  vars, fixed_length, param_range, 
                                                  simulation_val, csv_path, json_config, 
                                                  prior_space_pid)
            return average_throughput, res_folder
        else: 
            transfer_vector, average_throughput, res_folder = self.RunRSoftSim(name_tag, femsim_name_tag, fixed, 
                                                                   vars, fixed_length, param_range, 
                                                                   simulation_val, csv_path, json_config, 
                                                                   prior_space_pid)
            return transfer_vector, average_throughput, res_folder

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
            transfer_vector_batch=[tf_vector],
            csv_path = csv_path,
            name_tag = self.sym["Name"]
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
                                  transfer_vector_batch=tf_vector_val, 
                                  csv_path = csv_path, 
                                  name_tag = self.sym["Name"])

    def RunRSoft(self, simulation_val, prior_space_pid, csv_path, json_config, pid, simulate=False, build_tf = True): #csv_path, 
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
            seed_result, res_folder = self.build_circuit(seed_params, build_tf, json_config, csv_path, simulation_val, prior_space_pid)
            tf_vector = None
        else:
            tf_vector, seed_result, res_folder = self.build_circuit(seed_params, build_tf, json_config, csv_path, simulation_val, prior_space_pid)

        results_folder = log_optimizer_results(
            x_iters=[seed_params],
            y_vals=[-seed_result],
            param_batch=[seed_params],
            result_batch=[seed_result],
            param_names=param_names,
            iteration_start=0,
            batch_size=1,
            penalty_batch=None,
            transfer_vector_batch=[tf_vector],
            csv_path = csv_path,
            name_tag = self.sym["Name"]
        )
        return results_folder, res_folder

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
and then suggest new parameters to test.

WIP: gridding portion needs to be fixed
"""
import copy

def multiple_mode_tf(arg_list):
    '''
    TO DO: fix up the gridding part of this code.
    '''
    if len(arg_list) == 9:
        sim_val, custom_priors, m, rm, params, taper_min, taper_max, gridding, iteration_num = arg_list
    if len(arg_list) == 8:
        sim_val, custom_priors, gr, params, taper_min, taper_max, gridding, iteration_num = arg_list
    if gridding:
        sim_val, custom_priors, gr, params, taper_min, taper_max, gridding, iteration_num = arg_list
        sim_val = copy.deepcopy(sim_val)
        sim_val["grid_size"] = gr
        sim_val["grid_size_y"] = gr
        sim_val["launch_type"] = LaunchType.MM
        sim_val["launch_tilt"] = 0
        sim_val["launch_mode_radial"] = "*"
        sim_val["iter_num"] = iteration_num

        # CONTINUE HERE

        # specify MS core properties
        assign_core_properties(sim_val)

        # dump configuration paras into json for use later
        pid = os.getpid()
        prior_space_pid = f"Grid_{gr}_prior_space_{pid}.json"
        code_config = f"Grid_{gr}_launch_config.json"
        optimizer_result = f"Grid_{gr}_Optimizer_Result.csv"
        param_num = f"Grid {gr}"
    else:
        sim_val, custom_priors, m, rm, params, taper_min, taper_max, gridding, iteration_num = arg_list
        sim_val = copy.deepcopy(sim_val)
        sim_val["launch_mode"] = m
        sim_val["launch_mode_radial"] = rm
        sim_val["iter_num"] = iteration_num

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

    build_tf = True

    sim = RSoftSim()
    sim.init_priors(prior_space_pid, build_tf, custom_priors)

    # run simulation
    results_folder, res_folder = sim.RunRSoft(sim_val, prior_space_pid, csv_path = optimizer_result, json_config = code_config, pid = pid, build_tf = build_tf)

    # Load results
    best_para_log = os.path.join(results_folder, f"best_params_log_{pid}.csv")    
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
    return (param_num, tf_vectors, res_folder)

def run_tf_multproc(params, iteration_num, simulation_val, custom_priors, mode_vals, radial_mode_vals, taper_min, taper_max, gridding = False): 

    tf_list = []
    if not gridding:
        # initialise global parent argument list. This creates multiple instances of args_list based on the length of
        # mode_vals or radial_mode_vals. i.e. 
        # [(simulation_val, 0, 1, params, False),(simulation_val, 1, 1, params, False),(simulation_val, -1, 1, params, False), .....]
        args_list = [(simulation_val, custom_priors,m, rm, params, taper_min, taper_max, gridding, iteration_num) for m, rm in zip(mode_vals, radial_mode_vals)] 
    else:
        # run gridding determination
        grid_size_list = np.arange(0.1, 2.1, 0.1)
        grid_size_range = np.linspace(0.1, 2.0, len(grid_size_list))

        if simulation_val["launch_type"] != "LAUNCH_MULTIMODE" and Launch_params["launch_random_set"] != 0:
            raise Exception("Launch type must be multimode with a fixed random set when determining optimal grid sizes!")
        
        # initialise global parent argument list
        args_list = [(simulation_val, custom_priors, gr, params, taper_min, taper_max, gridding, iteration_num) for gr in grid_size_range] 

    # note 'spawn' means Python will start n separate processes each with their own memory of global variables. 
    # You need to define any changes within their own instance or else NOTHING changes.
    with mp.get_context("spawn").Pool(processes=6 if not gridding else 20) as pool:
        # supply multiple_mode_tf with the arguments required to run. 
        # Since the number of processes match the length of the mode_vals/radial_mode_vals
        # then each worker gets an instance of arg_list. i.e. arg_list[i]
        results = pool.map(multiple_mode_tf, args_list)
    
    res_folder = results[-1][2]
    for param, result, _ in results:
        # print(f"Param: {param}, Result: {result}")
        # tf_list.append(results)
        tf_list.append((param, result))
        
    if gridding:
        return grid_size_range, tf_list, res_folder
    else:
        return tf_list, res_folder

def run_all_modes_for_params(params, iteration_num, simulation_val, custom_priors, mode_vals, radial_mode_vals, taper_min, taper_max, gridding=False):
    # from RSoftSimulation import RSoftSim
    """
    For a single param vector, run all modes and aggregate result.
    params: list of optimized parameter values (from skopt)
    simulation_val: base simulation config
    mode_list: list of (m, rm) tuples
    gridding: bool, determines whether to run gridding determination or not
    """

    if gridding:
        # run gridding determination
        grid_size_range, tf_list, res_folder = run_tf_multproc(params, iteration_num, simulation_val, custom_priors, mode_vals, radial_mode_vals, params, taper_min, taper_max, gridding)
        return grid_size_range, tf_list
    else:
        # multiprocessing in here, we only want multiprocessing in this function and not in main_optimizer!!!!
        tf_list, res_folder = run_tf_multproc(params, iteration_num, simulation_val, custom_priors, mode_vals, radial_mode_vals, taper_min, taper_max, gridding)
    
    # Aggregate result (compute scalar loss/metric for this parameter vector)
    """
    put code to build the tf matrix here!!!!!
    use only 1 fixed name since the multiprocessing is in run_tf_multproc, not here; oNLY 1 MATRIX IS MADE PER ITERATION
    """
    core_to_monitor = simulation_val["core_to_monitor"] - 1
    core_number = simulation_val["core_num"]
    modes_to_monitor = ["LP01"]  
    amp = []
    phase = []
    tf_vals = []

    if gridding:
        for i in range(len(grid_size_range)):
            tf_vals.append(tf_list[0][i][1]) # if plotting grid sizes use 1, else 2 for transfer vectors

        tf_vals = np.array(tf_vals)

        amp, phase, grid_size, tf_result = extract_portmon_amp_phase(tf_vals, core_number,grid_size_range)
        phase = np.unwrap(phase)
        
        amp_data = []
        phase_data = []
        int_dat = []

        for idx, (gsize, a_vals, p_vals) in enumerate(zip(grid_size, amp, phase)):
            for core_idx, val in enumerate(a_vals):
                amp_data.append({"Grid": gsize, "Core": core_idx + 1, "Amplitude": val})
                int_dat.append({"Grid": gsize, "Core": core_idx + 1, "Log Intensity": np.log(val**2)})
            for core_idx, val in enumerate(p_vals):
                phase_data.append({"Grid": gsize, "Core": core_idx + 1, "Phase": val})

        amp_df = pd.DataFrame(amp_data)
        phase_df = pd.DataFrame(phase_data)
        phase_df['Phase_norm'] = phase_df.groupby('Core')['Phase'].transform(lambda x: x - x.iloc[0])
        int_df = pd.DataFrame(int_dat)

        fig_grid_sizes, axes = plt.subplots(2,1, figsize=(10, 8), sharex= True)

        ax = axes[0]
        sns.lineplot(data= amp_df, x="Grid", y="Amplitude", hue="Core", marker = "o", ax=ax)
        ax.set_title("Effect of grid size on BeamPROP Results")
        ax.set_ylabel("Amplitude")
        ax.legend(loc='upper right', title = "Core")

        ax = axes[1]
        sns.lineplot(data=phase_df, x="Grid", y="Phase_norm", hue="Core", marker="o", ax=ax)
        ax.set_ylabel("Normalised Phase")
        ax.legend(loc='lower right', title = "Core")
        ax.set_xlabel(r"Grid Size ($\mu m$)")

        plt.tight_layout()
        plt.savefig(f"{core_number}c{simulation_val['grid_type']}PL Grid Survey.png", dpi=100)

    else:
        for i in range(len(mode_vals)):
            tf_vals.append(tf_list[i][1])

        tf_vals = np.array(tf_vals)

        amp, phase, ex_amp, ex_phase, tf_result = extract_portmon_amp_phase(tf_vals,core_number)
        mode_reorder = [0, 5, 1, 2, 3, 4]
        max_value = print_max_amp_or_phase_value(amp)

        reorder = True
        tf_labels = ["Amplitude", "Phase"]

        tf_to_plot = [amp, phase]
        ex_amp = [ex_amp[k] for k in mode_reorder]
        ex_phase = [ex_phase[k] for k in mode_reorder]
        tf_to_plot_ex = [ex_amp, ex_phase]

        tf_figure = plt.figure(figsize = (20,12))
        param_str = ", ".join(f"{p:.3f}" for p in params)
        tf_figure.suptitle(
            f"Iteration {iteration_num}\nParameters: [{param_str}]",
            y=0.7, x=0.24
        )        
        gs = gridspec.GridSpec(1,3, width_ratios=[1,1,1])
        
        # ensure each plot is the same height
        ax0 = tf_figure.add_subplot(gs[0])
        ax1 = tf_figure.add_subplot(gs[1])
        ax2 = tf_figure.add_subplot(gs[2], sharey = ax1)

        # remove sharey for the first axis
        plt.setp(ax0.get_yticklabels(), visible=True)
        plt.setp(ax1.get_yticklabels(), visible=True)
        plt.setp(ax2.get_yticklabels(), visible=False)

        # plot core skeleton to highlight the special core
        if simulation_val["grid_type"] == "Pent":
            x, y = old_generate_pentagon_grid(fixed_params["MCFCladd"] / 2, fixed_params["core_sep"], simulation_val["core_num"])
            ax0.scatter(x,y)
            ax0.scatter(x[simulation_val["core_to_monitor"] - 1], y[simulation_val["core_to_monitor"] - 1], color="r", label = "H-Core")
            ax0.set_xlabel(r"x ($\mu m$)")
            ax0.set_ylabel(r"y ($\mu m$)")
            ax0.set_aspect('equal')
            ax0.legend(loc="upper right")
            
        elif simulation_val["grid_type"] == "Hex":
            row_num, excess = number_rows(simulation_val["core_num"])
            hcoord, vcoord = old_generate_hex_grid(row_num, fixed_params["core_sep"], include_centre = simulation_val["plot_centre_core"])
            
            if simulation_val["plot_centre_core"]:
                if 19 < simulation_val["core_num"] <= 37:
                    reorder_index = [18, 19, 25, 24, 17, 11, 12,
                                    20, 26, 31, 30, 29, 23, 16, 10, 5, 6, 7, 13,
                                    21, 27, 32, 36, 35, 34, 33, 28, 22, 15, 9, 4, 0, 1, 2, 3, 8, 14]
                elif 7 < simulation_val["core_num"] <= 19:
                    reorder_index = [9, 10, 14, 13, 8, 4, 5,
                                11, 15, 18, 17, 16, 12, 7, 3, 0, 1, 2, 6]
                elif simulation_val["core_num"] <= 7:
                    reorder_index = [3,4,6,5,2,0,1]
            else:
                if 19 < simulation_val["core_num"] <= 37:
                    reorder_index = [18, 19, 25, 24, 17, 11, 12,
                                    20, 26, 31, 30, 29, 23, 16, 10, 5, 6, 7, 13,
                                    21, 27, 32, 35, 34, 33, 28, 22, 15, 9, 4, 0, 1, 2, 3, 8, 14]
                elif 7 < simulation_val["core_num"] <= 19:
                    reorder_index = [9, 10, 14, 13, 8, 4, 5,
                            11, 15, 17, 16, 12, 7, 3, 0, 1, 2, 6]
                elif simulation_val["core_num"] <= 7:
                    reorder_index = [3,4,6,5,2,0,1]

            xcoord_og, ycoord_og, xcoord_relist, ycoord_relist = plot_excess(excess, hcoord, vcoord, reorder_index)
            for i, (xval, yval) in enumerate(zip(xcoord_relist, ycoord_relist)):
                if yval > 0:
                    ax0.annotate(f"{i+1}", (xval -1, yval - 5))
                else:
                    ax0.annotate(f"{i+1}", (xval -1, yval + 2))
            ax0.scatter(xcoord_relist, ycoord_relist)
            ax0.scatter(xcoord_relist[simulation_val["core_to_monitor"]-1], 
                        ycoord_relist[simulation_val["core_to_monitor"]-1], 
                        color="r", label = "H-Core")
            ax0.set_xlabel(r"x ($\mu m$)")
            ax0.set_ylabel(r"y ($\mu m$)")
            ax0.set_aspect('equal')
            ax0.legend(loc="upper right")
            
        # plot the individual amplitude and phase matrix
        axes = [ax1, ax2]
        for i, (lab, tf_type, ex_type) in enumerate(zip(tf_labels, tf_to_plot, tf_to_plot_ex)):
            plot_phase = False

            if lab == "Phase":
                plot_phase = True
            plot_tf_matrix(tf_type, simulation_val, ex_type, matrix_type=f"{lab}", ax=axes[i], cbar=True, reorder = reorder, phase = plot_phase)
            ax2.set_ylabel(None)
            plt.close()
            plt.close()

        # tf_figure.savefig("Grif+Amplitude+Phase TF Matrix.png", dpi=300)
        # plot the combined amplitude and phase matrix
        plot_combined_tf_matrix(simulation_val, amp, phase, ex_amp, ex_phase, core_number, amp_max = max_value,
                                dir = r"C:\Users\RSoft Things\Desktop\RSoft-Automaton", 
                                name = f"TF_Combined_{simulation_val['core_num']}c{simulation_val['grid_type']}PL.png")
        
        image_dir = r"C:\Users\RSoft Things\Desktop\Results\Images"
        monitored_mode = modes_to_monitor[0]
        filename = f"transfer_matrix_{monitored_mode}_iteration_{iteration_num}.png"
        save_path = os.path.join(image_dir, filename)
        tf_figure.savefig(save_path, dpi=300)
        plt.close()

    hyp_param_b = simulation_val.get("hyp_param_b", Simulation_params["hyp_param_b"])
    hyp_param_c = simulation_val.get("hyp_param_c", Simulation_params["hyp_param_c"])
    # sim = RSoftSim()
    loss, arr_results = mode_selective_tf_matrix_metric(
        tf_list, res_folder,
        hyp_param_b,
        hyp_param_c,
        core_to_monitor=core_to_monitor,
        modes_to_monitor=modes_to_monitor,
        simulation_val = simulation_val
    )
        
    return loss, arr_results, amp, phase, ex_amp, ex_phase

def main_optimizer(prior_space_pid, simulation_val, custom_priors, mode_vals, radial_mode_vals, taper_min, taper_max, gridding, total_calls, simulate_tf_metric = True):
    
    array_of_results = []
    
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
    if simulation_val["industry_neff_values"] == True:
        neff_val = read_neff_values(simulation_val["industry_neff_file"])
        para_space = []
        for prior_name, (low, high) in param_range.items():
            if prior_name == "core_neff":
                para_space.append(Categorical(neff_val, name=prior_name))
            else:
                para_space.append(Real(low, high, name=prior_name))
    elif bestvals:
        para_space = []
        sampling_regions = sample_range(param_range, bestvals, shrink=15.0)
        for prior_name, (low, high) in sampling_regions.items():
            para_space.append(Real(low, high, name=prior_name))
    else:
        para_space = []
        for prior_name, (low, high) in param_range.items():
            if prior_name == "core_neff":
                para_space.append(Real(low, high, name=prior_name))
            else:
                para_space.append(Real(low, high, name=prior_name))
        # para_space = [Real(low, high, name=prior_name) for prior_name, (low, high) in param_range.items()]
    
    # initialise optimiser
    opt = Optimizer(
        dimensions=para_space,
        base_estimator="GP",
        acq_func="EI", #LCB
        acq_func_kwargs={"xi": 0.8}, #{"kappa": 2.5}
        acq_optimizer = "sampling",
        random_state=None,
        n_initial_points=50
    )
    # if true, run optimisation testing the loss metric
    if simulate_tf_metric:
        all_results = []
        for batch_idx in range(total_calls):
            # ask for 1 set of parameter vectors only to prevent daemonic process having children 
            param_batch = opt.ask()

            print("Trying " + ", ".join(
                f"Core neff: {param_batch[l]:.3f}" if text == "core_neff"
                else f"{text}: {param_batch[l]:.3f}"
                for l, text in enumerate(variable_params.keys())
            ))         
            

            result_batch, arr_results, amp, phase, ex_amp, ex_phase = run_all_modes_for_params(param_batch, batch_idx + 1, 
                                                                      simulation_val, custom_priors, mode_vals, 
                                                                      radial_mode_vals, taper_min, 
                                                                      taper_max, gridding = gridding)
             
            # tell optimiser the performance of the chosen parameters
            opt.tell(param_batch, result_batch)
            # log iteration of parameters, store for later use
            array_of_results.append(arr_results)
            print(f"Iteration {batch_idx + 1}: {result_batch}\n"
                  f"Loss metric iteration {batch_idx + 1} (a, b, c, c): {array_of_results[batch_idx]}")
            
            # for j, variable_text in enumerate(variable_params.keys()):
            #     if variable_text == "core_neff":
            #         param_batch[j] = param_batch[j] + RSoft_params["background_index"]   

            all_results.append({'params': param_batch, 'result': result_batch, 
                                'Iteration': batch_idx + 1, "Loss Metric": array_of_results, 
                                "Core Amplitudes": amp, "Core Phases": phase,
                                "Extra Core Amplitudes": ex_amp, "Extra Core Phases": ex_phase})
        return all_results
    # if false, run tf code for the template parameters
    else:
        param_names = ["core_diam", "core_neff"]
        params = [variable_params[k] for k in param_names]
        if gridding:
            grid_size_range, tf_list, _ = run_tf_multproc(params, 1, simulation_val, custom_priors, mode_vals, radial_mode_vals,taper_min, taper_max, gridding)
            return grid_size_range, tf_list, _
        else:
            tf_list = run_tf_multproc(params, 1,simulation_val, custom_priors, mode_vals, radial_mode_vals,taper_min, taper_max, gridding)
        return tf_list