import numpy as np, os, shutil, csv
import subprocess, json, time
from pathlib import Path
from skopt import Optimizer, dump, load
from skopt.space import Real, Categorical
# from skopt.utils import dump
import multiprocessing as mp
import matplotlib.gridspec as gridspec
import seaborn as sns
import datetime
import glob
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
            row_num, excess = number_rows(core_num)
            hcoord, vcoord = old_generate_hex_grid(row_num, fixed_params["core_sep"], include_centre = Simulation_params["plot_centre_core"])
            
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
        
        if grid_type == "Triangle":
            hcoord, vcoord = generate_triangular_grid(fixed_params["core_sep"])
            self.core_positions = list(zip(hcoord, vcoord))
            self.cladd_positions = None

            with open("core_positions.json", "w") as g:
                json.dump(self.core_positions, g)
    

    def RunRSoftSim(self, name_tag, femsim_name_tag, fixed, vars, fixed_length, param_range, simulation_val, csv_path, json_config, prior_space_pid, wave,fem=False):
        filename = f"{name_tag}.ind"
        filename_FS = f"{femsim_name_tag}.ind"
        sim_tool = simulation_val.get("sim_tool", RSoft_params["sim_tool"])
        iter_number = simulation_val["iter_num"]
        # Run RSoft simulation
        if sim_tool == "ST_BEAMPROP":
            prefix_BP   = f"prefix={name_tag}"
            if fem:
                folder_BP = "FemSIM_DET"
            else:
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
        onedrive_neff_csv_path = Path(onedrive_results_folder) / f"{wave}_Guided Modes_{pid_csv}.csv"
        neff_csv_path = Path(results_folder) / f"{wave}_Guided Modes_{pid_csv}.csv"

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

        stored_data = pd.read_csv(r"C:\Users\RSoft Things\OneDrive - The University of Sydney (Students)\Apps\VSCode\Sellmeier_Considerations\Sellmeier_vals.csv")
        _, idx = find_nearest(stored_data["Wavelength (um)"].to_numpy(), wave)
        fixed_params["cladding_neff"] = stored_data["SiO2"].to_numpy()[idx]
        guided_mask = y_neff > fixed_params["cladding_neff"]
        guided_neff = np.array(y_neff[guided_mask])

        with open(neff_csv_path, mode="w", newline="") as f_neff:
            writer = csv.writer(f_neff)
            writer.writerow(["Mode_Index", "n_eff", "Wavelength (um)", "PID"])
            for idx, nval in enumerate(guided_neff, start=1):
                writer.writerow([idx, nval, wave, pid_csv])

        with open(onedrive_neff_csv_path, mode="w", newline="") as f_neff_od:
            writer = csv.writer(f_neff_od)
            writer.writerow(["Mode_Index", "n_eff", "Wavelength (um)", "PID"])
            for idx, nval in enumerate(guided_neff, start=1):
                writer.writerow([idx, nval, wave, pid_csv])

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
                return transfer_vector, -throughput, results_folder, pid_csv
            else:
                transfer_vector, throughput = transfer_matrix_component(csv_pathway, row)
                return transfer_vector, -throughput, results_folder, pid_csv

    def build_circuit(self, params, build_tf, json_config, csv_path, simulation_val, prior_space_pid, wave, delta_index_at_reference_wavelength, fem=False): # maybe put this into its own function. Make it universal.
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
        # capillary_beg_dims = ((cladd_diam / taper)*1.3, (cladd_diam / taper)*1.3) 
        # capillary_end_dims = (cladd_diam * 1.3, cladd_diam * 1.3)

        core_beg_dims_list = []
        core_end_dims_list = []

        if sim_param["mode_selective"] == 1:
            for j, core_key in enumerate(core_name, start=1):
                if j == Simulation_params["core_to_monitor"]:
                    # need to modify the core_neff according to the wavelength
                    # core to be optimized by skopt
                    core_diam = variable_params.get("core_diam")
                    core_neff = fixed_params["cladding_neff"] + delta_index_at_reference_wavelength#variable_params.get("core_neff", fixed_params.get("core_neff"))
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
                    # capillary_beg_dims, capillary_end_dims,
                    core_beg_dims_list, core_end_dims_list,
                    simulation_val,self.cladd_positions)
            
        if simulation_val["launch_type"] == LaunchType.SM:
            launch_mode = simulation_val["launch_mode"]
            launch_mode_radial = simulation_val["launch_mode_radial"]
            param_string = "_".join(f"{key}_{val:.6f}" for key, val in param_dict.items())
            name_tag = f"{wave}_LP{launch_mode}{launch_mode_radial}_{param_string}"
        else:
            grid = simulation_val["grid_size"] # use only when trying to find the optimal gridding to run BeamPROP in.
            name_tag = f"_Grid{grid}".join(f"{key}_{val:.6f}" for key, val in param_dict.items())
        
        if fem:
            self.sym["Name"] = name_tag
            self.circuit.write(f"{name_tag}.ind")
            femsim_name_tag = f"FemSim_File_DET_{name_tag}"
            self.circuit.write(f"{femsim_name_tag}.ind")
        else:
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
                path_num - 1, param_dict, simulation_val, wave,fem=fem)
        '''
        Manual setup to loop through a list of values. Runs the terminal line that will initiate RSoft and will calculate the 
        metric to test.
        All output files will appear in a subfolder on the Desktop (windows)
        '''

        if Simulation_params['metric'] != 'TF':
            average_throughput, res_folder, pid_csv = self.RunRSoftSim(name_tag, femsim_name_tag, fixed, 
                                                  vars, fixed_length, param_range, 
                                                  simulation_val, csv_path, json_config, 
                                                  prior_space_pid, wave)
            return average_throughput, res_folder, pid_csv
        else: 
            transfer_vector, average_throughput, res_folder, pid_csv = self.RunRSoftSim(name_tag, femsim_name_tag, fixed, 
                                                                   vars, fixed_length, param_range, 
                                                                   simulation_val, csv_path, json_config, 
                                                                   prior_space_pid, wave, fem = fem)
            return transfer_vector, average_throughput, res_folder, pid_csv

    # def MultProc(self, build_tf, json_config, csv_path, simulation_val, prior_space_pid): #csv_path
    #     images_dir = Path(os.path.expanduser("~/Desktop/Results/Images"))
    #     images_dir.mkdir(parents=True, exist_ok=True)

    #     # load prior space
    #     if build_tf:
    #         for attempt in range(10):
    #             try:
    #                 with open(prior_space_pid, "r") as read:
    #                     param_range = json.load(read)
    #                 break
    #             except json.decoder.JSONDecodeError:
    #                 time.sleep(0.2)
    #         else:
    #             raise RuntimeError(f"Failed to load {prior_space_pid} after retries.")
    #     else:
    #         for attempt in range(10):
    #             try:
    #                 with open("prior_space.json", "r") as read:
    #                     param_range = json.load(read)
    #                 break
    #             except json.decoder.JSONDecodeError:
    #                 time.sleep(0.2)
    #         else:
    #             raise RuntimeError("Failed to load prior_space.json after retries.")
            
    #     para_space = [Real(low, high, name=prior_name) for prior_name, (low, high) in param_range.items()]
        
    #     # backend of skopt.gp_minimize that can handle multiprocessing
    #     opt = Optimizer(
    #         dimensions = para_space,
    #         base_estimator = "GP",
    #         acq_func = "EI",
    #         random_state=42
    #     )

    #     sim_param = Simulation_params
    #     # how many values in each parameter space to run simulation with
    #     total_calls = sim_param["num_paras"]

    #     # this is the number of points to sample simultaneously. 
    #     # Increase to cycle through prior space quicker at the cost of CPU computation
    #     batch_size = sim_param["batch_num"]
    #     all_results = []
    #     # initialise the optimizer with template solutions
    #     param_names = [dim.name for dim in para_space]
    #     seed_params = [variable_params[k] for k in param_names]

    #     # self.sym["Name"] = "MCF_Test"
    #     if Simulation_params['metric'] != 'TF':
    #         throughput, seed_result, pid_csv = self.build_circuit(seed_params, build_tf, json_config, csv_path, simulation_val, prior_space_pid)
    #         opt.tell(seed_params, seed_result)
    #         tf_vector = None
    #     else:
    #         tf_vector, seed_result, pid_csv = self.build_circuit(seed_params, build_tf, json_config, csv_path, simulation_val, prior_space_pid)
    #         # seed_result = mode_selective_tf_matrix_metric(tf_vector, simulation_val["core_to_monitor"], simulation_val["modes_to_monitor"])
    #         opt.tell(seed_params, seed_result)

    #     all_results.append({
    #         "params": seed_params,
    #         "result": seed_result,
    #         "transfer_vector": tf_vector
    #     })

    #     log_optimizer_results(
    #         x_iters=[seed_params],
    #         y_vals=[-seed_result],  
    #         param_batch=[seed_params],
    #         result_batch=[seed_result],
    #         param_names=param_names,
    #         iteration_start=0,
    #         batch_size=1,
    #         penalty_batch=None,
    #         transfer_vector_batch=[tf_vector],
    #         csv_path = csv_path,
    #         name_tag = self.sym["Name"]
    #     )
    #     # run optimizer as normal
    #     for i in range(0, total_calls, batch_size):
            
    #         # Suggest next batch of points
    #         param_batch = opt.ask(batch_size)

    #         # Evaluate in parallel
    #         ctx = mp.get_context("spawn")
    #         args_list = [(params, build_tf, json_config, csv_path, simulation_val, prior_space_pid) for params in param_batch]
    #         with ctx.Pool(batch_size) as pool:
    #             result_batch = pool.map(run_rsoft_sim, args_list) 

    #         # Feed results back to optimizer
    #         # opt.tell(param_batch, result_batch)
    #         if Simulation_params['metric'] == 'TF':
    #             scores_only = [score for _, score in result_batch]
    #             opt.tell(param_batch, scores_only)
    #         else:
    #             opt.tell(param_batch, result_batch)
            
    #         for p, r in zip(param_batch, result_batch):
    #             if Simulation_params['metric'] == 'TF':
    #                 tf_vector, score = r
    #             else:
    #                 tf_vector = None
    #                 score = r
    #             all_results.append({
    #                 "params": p,
    #                 "result": score,
    #                 "transfer_vector": tf_vector
    #             })
                
    #         # all_results.extend(zip(param_batch, result_batch))

    #         '''
    #         save results for plotting/analysis
    #         '''
            
    #         # Unpack results
    #         x_iters = [r["params"] for r in all_results]  # parameter sets
    #         y_vals = [-r["result"] for r in all_results]  # throughput values
    #         tf_vector_val = [r["transfer_vector"] for r in all_results]

    #         param_names = [dim.name for dim in opt.space.dimensions]
    #         # log chosen values and penalties
    #         log_optimizer_results(x_iters, y_vals,
    #                               param_batch, 
    #                               [r["result"] for r in all_results[-batch_size:]],
    #                               param_names, iteration_start=i,
    #                               batch_size = batch_size,
    #                               penalty_batch= None,
    #                               transfer_vector_batch=tf_vector_val, 
    #                               csv_path = csv_path, 
    #                               name_tag = self.sym["Name"])

    def RunRSoft(self, simulation_val, prior_space_pid, wave, delta_index_at_reference_wavelength, csv_path, json_config, pid, fem=False, simulate=False, build_tf = True): #csv_path, 
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
            seed_result, res_folder, pid_csv = self.build_circuit(seed_params, build_tf, json_config, csv_path, simulation_val, prior_space_pid, wave, delta_index_at_reference_wavelength)
            tf_vector = None
        else:
            tf_vector, seed_result, res_folder, pid_csv = self.build_circuit(seed_params, build_tf, json_config, csv_path, simulation_val, prior_space_pid, wave, delta_index_at_reference_wavelength, fem = fem)

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
            results_folder = res_folder,
            csv_path = csv_path,
            name_tag = self.sym["Name"]
        )
        return results_folder, res_folder, pid_csv

def run_rsoft_sim(args):
    from RSoftSimulation import RSoftSim  
    from Functions import overwrite_template_val

    params, build_tf, json_config, csv_path, simulation_val, prior_space_pid = args
    # this needs to be defined here as well or 
    # else some paras won't be updated for some reason???
    overwrite_template_val(json_config)
    sim = RSoftSim()
    sim.generate_core_positions()
    return sim.build_circuit(params, build_tf, json_config, csv_path, simulation_val, prior_space_pid, wave, delta_index_at_reference_wavelength)

#############################################################################################################################################################################
"""
Below is the current workflow for building the transfer matrix for a PL. Above is blindly optimizing a single core in the PL using 
it's throughput as the metric.

CURRENT: simultaneously chooses pairs of modes to run RSoft with to develop the transfer matrix faster than running individually.
Multiprocessing to occur after the parameter is chosen that sequentially injects individual modes to build the transfer matrix. 
Multiprocessing that picks a new set of parameters, injects individual LP modes to build the transfer matrix, 
and then suggest new parameters to test.
"""
import copy

def multiple_mode_tf(arg_list):
    '''
    TO DO: fix up the gridding part of this code.
    '''
    if len(arg_list) == 13:
        sim_val, custom_priors, wave, m, rm, params, fem, taper_min, taper_max, gridding, delta_index_at_reference_wavelength, core_neff_idx, iteration_num = arg_list
    # elif len(arg_list) == 11:
    #     sim_val, custom_priors, wave, m, rm, params, fem, taper_min, taper_max, gridding, iteration_num = arg_list
    elif len(arg_list) == 8:
        sim_val, custom_priors, gr, params, taper_min, taper_max, gridding, iteration_num = arg_list
    else:
        raise RuntimeError(f"There are some values unaccounted for while building the multprocessor. The total number of arguments should be {len(arg_list)}")
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
        # sim_val, custom_priors, wave, m, rm, params, fem, taper_min, taper_max, gridding, iteration_num = arg_list
        sim_val = copy.deepcopy(sim_val)
        sim_val["launch_mode"] = m
        sim_val["launch_mode_radial"] = rm
        sim_val["iter_num"] = iteration_num
        sim_val["free_space_wavelength"] = wave

        # Open file containing the refractive indices determined from the Selmeier equation
        stored_data = pd.read_csv(r"C:\Users\RSoft Things\OneDrive - The University of Sydney (Students)\Apps\VSCode\Sellmeier_Considerations\Sellmeier_vals.csv")
        _, idx = find_nearest(stored_data["Wavelength (um)"].to_numpy(), wave)
        sim_val["core_neff"] = stored_data["GeO2_2_mol%"].to_numpy()[idx]
        fixed_params["cladding_neff"] = stored_data["SiO2"].to_numpy()[idx]
        Launch_params["cladding_neff"] = fixed_params["cladding_neff"]
        RSoft_params["background_index"] = stored_data["F_2_mol%"].to_numpy()[idx]

        # n_ms(lambda) = n_SiO2(lambda) + Delta n
        # params[core_neff_idx] = fixed_params["cladding_neff"] + delta_index_at_reference_wavelength
        # write core diameter properties to simulation_val and set special core properties to None for SKOPT to overwrite
        assign_core_properties(sim_val)
        core_to_monitor = sim_val["core_to_monitor"]

        # dynamically load parameters to vary
        param_names = list(variable_params.keys())

        # Assign the new params directly to the special core
        for pname, pval in zip(param_names, params):
            sim_val[f"core_{core_to_monitor}"][pname] = pval
            variable_params[pname] = pval
        
        if fem:
            variable_params["core_neff"] = sim_val["core_neff"]

        # dump configuration paras into json for use later
        # pid is needed to avoid cross-talking between files created/used in multiprocessing tasks
        pid = os.getpid()
        # prior space identifier
        prior_space_pid = f"{wave}_LP_{m}{rm}_prior_space_{pid}.json"
        # launch config identifier
        code_config = f"{wave}_LP_{m}{rm}_launch_config_{pid}.json"
        # csv to store results later
        optimizer_result = f"{wave}_LP_{m}{rm}_Optimizer_Result_{pid}.csv"
        param_num = f"{wave}_LP{m}{rm}"
    
    # dump launch config profile into simulation_val
    with open(code_config, "w") as launch_config:
        json.dump(sim_val, launch_config, indent = 2)

    build_tf = True

    sim = RSoftSim()
    sim.init_priors(prior_space_pid, build_tf, custom_priors)

    # run simulation
    results_folder, res_folder, pid_csv = sim.RunRSoft(sim_val, prior_space_pid, wave, delta_index_at_reference_wavelength,csv_path = optimizer_result, json_config = code_config, pid = pid, fem = fem, build_tf = build_tf)

    # Load results
    best_para_log = os.path.join(results_folder, f"best_params_log_{pid}.csv")    
    data = pd.read_csv(best_para_log)

    # Extract parameter names
    param_names = list(custom_priors.keys())

    # Identify and extract TF columns
    tf_columns = [col for col in data.columns if col.startswith("TF_")]
    if tf_columns:
        tf_vectors = data.loc[:, tf_columns].to_numpy(dtype=float)
        # tf_vectors = data[tf_columns].values.tolist()  
    else:
        tf_vectors = None

    # Call plotting function
    plotting_optimizer_results(data, param_names, tf=tf_vectors, plot= False)
    return (param_num, tf_vectors, wave, res_folder, pid_csv)

def run_tf_multproc(params, iteration_num, simulation_val, custom_priors, mode_vals, radial_mode_vals, taper_min, taper_max, fem = False, gridding = False): 

    tf_list = []

    if "core_neff" in variable_params:
        # get index of core_neff in variable_params
        for i, key in enumerate(variable_params):
            if key == "core_neff":
                core_neff_idx = i

        # define index contrast to scale each sampled refractive index according to some fixed index contrast calculated at a reference wavelength
        stored_data = pd.read_csv(r"C:\Users\RSoft Things\OneDrive - The University of Sydney (Students)\Apps\VSCode\Sellmeier_Considerations\Sellmeier_vals.csv")
        _, refractive_index_at_reference_wave = find_nearest(stored_data["Wavelength (um)"].to_numpy(), min(simulation_val["free_space_wavelength"])) # only calculate the contrast from the smallest wavelength simulated
        Silica_refractive_index_at_reference_wavelength = stored_data["SiO2"].to_numpy()[refractive_index_at_reference_wave]
        delta_index_at_reference_wavelength = params[core_neff_idx] - Silica_refractive_index_at_reference_wavelength
    
    if not gridding:
        # initialise global parent argument list. This creates multiple instances of args_list based on the length of
        # mode_vals or radial_mode_vals. i.e. 
        # [(simulation_val, 0, 1, params, False),(simulation_val, 1, 1, params, False),(simulation_val, -1, 1, params, False), .....]
        if "core_neff" in variable_params.keys():
            args_list = [(simulation_val, custom_priors,wavelengths,m, rm, params, fem, taper_min, taper_max, gridding, delta_index_at_reference_wavelength, core_neff_idx, iteration_num) for wavelengths in simulation_val["free_space_wavelength"] for m, rm in zip(mode_vals, radial_mode_vals)] 
        else:
            args_list = [(simulation_val, custom_priors,wavelengths,m, rm, params, fem, taper_min, taper_max, gridding, iteration_num) for wavelengths in simulation_val["free_space_wavelength"] for m, rm in zip(mode_vals, radial_mode_vals)] 
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
    with mp.get_context("spawn").Pool(processes=int(simulation_val["core_num"])*len(simulation_val["free_space_wavelength"]) if not gridding else 20) as pool:
        # supply multiple_mode_tf with the arguments required to run. 
        # Since the number of processes match the length of the mode_vals/radial_mode_vals
        # then each worker gets an instance of arg_list. i.e. arg_list[i]
        results = pool.map(multiple_mode_tf, args_list) # (param_num, tf_vectors, wave, res_folder, pid_csv)
    
    res_folder = results[-1][3]
    for param, result, wave, _, csv_pid in results:
        # print(f"Param: {param}, Result: {result}")
        # tf_list.append(results)
        tf_list.append((param, result, wave, csv_pid))
        # csv_pid_arr.append(csv_pid)
        
    if gridding:
        return grid_size_range, tf_list, res_folder
    else:
        return tf_list, res_folder

def run_all_modes_for_params(params, iteration_num, simulation_val, custom_priors, taper_min, taper_max, gridding=False):
    """
    For a single param vector, run all modes and aggregate result.
    params: list of optimized parameter values (from skopt)
    simulation_val: base simulation config
    mode_list: list of (m, rm) tuples
    gridding: bool, determines whether to run gridding determination or not
    """

    mode_vals = simulation_val["mode_vals"]
    radial_mode_vals = simulation_val["radial_mode_vals"]
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

    hyp_param_b = simulation_val.get("hyp_param_b", Simulation_params["hyp_param_b"])
    hyp_param_c = simulation_val.get("hyp_param_c", Simulation_params["hyp_param_c"])

    waves = np.asarray([w for (_, _, w, _) in tf_list], dtype=float)
    unique_waves = np.unique(waves)
    wave_rows = []
    row = []
    for w in unique_waves:
        tf_list_w = [(lab, arr, wave, pid) for (lab, arr, wave, pid) in tf_list if float(wave) == float(w)]
        tf_list_w_arr = [arr for (_, arr, _, _) in tf_list_w]
        mode_labels_raw = [lab for (lab, _, _, _) in tf_list_w]
        pid_w = [pid_raw for (_, _, _, pid_raw) in tf_list_w]
        # compute scalar loss for this wavelength
        loss, arr_results, _, len_modes_arr, loss_a_num_extra_modes = mode_selective_tf_matrix_metric(
            tf_list_w, res_folder, w, pid_w,
            hyp_param_b,
            hyp_param_c,
            core_to_monitor=core_to_monitor,
            modes_to_monitor=modes_to_monitor,
            simulation_val = simulation_val
        )

        og_amp, og_phase, ex_amp, ex_phase, _ = extract_portmon_amp_phase(tf_list_w_arr, core_num=simulation_val["core_num"])
        n_modes, n_cores = og_amp.shape
        stored_data = pd.read_csv(r"C:\Users\RSoft Things\OneDrive - The University of Sydney (Students)\Apps\VSCode\Sellmeier_Considerations\Sellmeier_vals.csv")
        
        # find the refractive index for each segment according to the wavelength simulated
        _, idx = find_nearest(stored_data["Wavelength (um)"].to_numpy(), w) # find nearest index to the simulated wavelength
        Other_core_ref_ind = stored_data["GeO2_2_mol%"].to_numpy()[idx] # find value for non-ms core refractive index
        Cladding_ref_ind = stored_data["SiO2"].to_numpy()[idx] # find value for cladding refractive index
        Capillary_ref_ind = stored_data["F_2_mol%"].to_numpy()[idx] # find value for capillary refractive index

        # extract variable parameter keys
        k_arr = []
        for k in variable_params.keys():
            k_arr.append(k)
        k_arr = np.array(k_arr)

        ## Code to search for a single .ind file and copy it to Wavelength_results
        results_path = Path(rf'C:\Users\RSoft Things\Desktop\Results\BP_SimulationNum_{iteration_num}')
        wavelength_results_folder = r'C:\Users\RSoft Things\Desktop\Results\Wavelength_results'
        
        ind_files = list(results_path.glob("1.9_LP01_*.ind")) # returns a list
        if not ind_files:
            raise FileNotFoundError(f"No .ind file found in: {results_path}")

        ind_file = ind_files[0]
        shutil.copy2(ind_file, wavelength_results_folder)

        for m in range(n_modes):
            mode_label = (
                mode_labels_raw[m] if (mode_labels_raw is not None and m < len(mode_labels_raw))
                else f"Mode{m+1}"
            )

            row={
                "Simulation Date": datetime.datetime.now(),
                "Iteration": int(iteration_num),
                "Wavelength": float(w),
                "Loss Value": float(loss),
                "Loss_a": arr_results[0],
                "Loss_b": arr_results[1],
                "Loss_c": arr_results[2],
                "Loss_d": arr_results[3],
                f"{k_arr[0]}": float(params[0]),
                f"{k_arr[1]}": float(params[1]),
                f"{k_arr[2]}": float(params[2]),
                f"Delta n({simulation_val['free_space_wavelength'][0]} um)": float(params[1] - Cladding_ref_ind),
                "Non-MS Core Refractive Index": Other_core_ref_ind,
                "Cladding Refractive Index": Cladding_ref_ind,
                "Capillary Refractive Index": Capillary_ref_ind,
                "Guided Modes":int(len_modes_arr[0]),
                "Extra Mode Intensity in Loss_a": int(len(loss_a_num_extra_modes[0])) if simulation_val["all_modes"] else "None",
                "Injected Mode": str(mode_label),
                "Mode Index": int(m),
                "PID": pid_w
            }

            for c in range(n_cores):
                row[f"Core_{c+1}_Amp"] = og_amp[m, c]
                row[f"Core_{c+1}_Phase"] = og_phase[m, c]
            
            _, n_ex = ex_amp.shape

            # if m==0 or mode_label.endswith("_LP01"):
            for c in range(n_ex):
                row[f"{LP_mode_dict_rot[c+1]}_Amp"] = float(ex_amp[m, c])
                row[f"{LP_mode_dict_rot[c+1]}_Phase"] = float(ex_phase[m, c])

            wave_rows.append(row)
    

    ## Globally fixed parameters
    core_pos = core_pos_geo(simulation_val)

    glob_fix_param = {
        "Time CSV Created": datetime.datetime.now(),
        "Non-MS Core Diameter ($\mu m$)": core_params[f"core_{(simulation_val['core_to_monitor'] + 1)%simulation_val['core_num']}"]["core_diam"],
        "Cladding Diameter ($\mu m$)": fixed_params["MCFCladd"],
        "Core Separation ($\mu m$)": fixed_params["core_sep"],
        "MS Core Position": core_pos[simulation_val['core_to_monitor']-1],
        "MS Mode": LP_mode_dict_rot[0], # Need to somehow make this dynamic, only selects LP01 atm
        "Core Configuration": simulation_val['grid_type'],
        "Number of Cores": simulation_val["core_num"],
        "Example .ind File Used": ind_file,
        "Loss_a config.": "LP01" if not simulation_val["all_modes"] else "LP01 + higher order modes"
    }

    if "taper" not in k_arr:
        glob_fix_param["Taper"] = fixed_params["taper"]

    # append_kv_rows(wave_rows, "GLOBAL", glob_fix_param, base_cols)

    ## Legend
    leg = {
        "Loss_a": ("Intensity of MS mode in the MS core with the sum of intensities of higher order modes in ms core" if simulation_val["all_modes"] else "Intensity of MS mode in the MS core"),
        "Loss_b": "Mean intensity of non MS modes in non MS cores",
        "Loss_c": "Mean intensity of non MS modes exciting LP01 in MS core",
        "Loss_d": "Mean intensity of MS mode in non MS cores",
        "Loss": "-Loss_a - Loss_b + (Loss_c + Loss_d) + 2",
        "Extra Mode Intensity in Loss_a": "Total number of amplitudes corresponding to higher order modes included in Loss_a",
        f"Delta n({simulation_val['free_space_wavelength'][0]} um)": "Refractive index scale factor relative to the index difference between the selected refractive index and the index of silica at a reference wavelength. This should give a slightly different value for different wavelengths.",
        "Guided Modes": "Total number of modes, including rotations AND polarisations, being guided in the fibre."
    }

    df_wave_log = pd.DataFrame(wave_rows)
    outdir = r"C:\Users\RSoft Things\Desktop\Results\Wavelength_results"
    os.makedirs(outdir, exist_ok=True)
    pid = os.getpid()
    out_csv = os.path.join(outdir, f"wavelength_loss_iter_{iteration_num}_{simulation_val['core_num']}{simulation_val['grid_type']}_{simulation_val['mon_type']}_NumModes_{len(simulation_val['mode_vals'])}_{pid}.csv")
    df_wave_log.to_csv(out_csv, index=False)

    # now append the global and legend
    with open(out_csv, "a", newline="") as f:
        f.write("\n")  # blank line

        f.write("Globally Fixed Parameters\n")
        for k, v in glob_fix_param.items():
            f.write(f"{k}: {v}\n")

        f.write("\nLegend\n")
        for k, v in leg.items():
            f.write(f"{k}: {v}\n")

    if len(unique_waves) == 1:
        final_loss = float(df_wave_log["Loss Value"].iloc[0])
    else:
        L = (
            df_wave_log
            .dropna(subset=["Loss Value"])
            .groupby("Wavelength")["Loss Value"]
            .first()
            .to_numpy(dtype=float)
        )
        final_loss = float(np.sqrt(np.mean((L - np.mean(L))**2))+np.mean(L))

    return final_loss, df_wave_log

def main_optimizer(prior_space_pid, simulation_val, custom_priors, mode_vals, radial_mode_vals, taper_min, taper_max, gridding, total_calls, simulate_tf_metric = True):

    # create image folder in results location
    results_dir = Path(os.path.expanduser("~/Desktop/Results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    images_dir = results_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    results_checkpoint_path = results_dir / "optimizer_results_checkpoint.npy"
    opt_checkpoint_path = results_dir / "optimizer_state_checkpoint.pkl"
    # images_dir = Path(os.path.expanduser("~/Desktop/Results/Images"))
    # images_dir.mkdir(parents=True, exist_ok=True)

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
        # para_space = []
        # sampling_regions = sample_range(param_range, bestvals, shrink=15.0)
        # for prior_name, (low, high) in sampling_regions.items():
        #     para_space.append(Real(low, high, name=prior_name))

        mean_vals = np.array(list(bestvals.values()), dtype=float)
        mean_limits = np.array(list(bestval_limits.values()), dtype=float)
        sigmas = np.array([2.0, 1.0, 3000.0])
        scales = np.array([1.0, 1.0, 10000.0])
        _, accepted = monte_carlo_rej(mean_vals, mean_limits, scales, sigmas, simulation_val["num_paras"])
        # para_space = coarse_sampler(bestvals)
        para_space = accepted.T # shape(len(simulation_val["num_paras"]), len(bestvals.keys()))
        print(f"Sampling of {para_space.shape[0]} parameters, begin.")
    else:
        para_space = []
        for prior_name, (low, high) in param_range.items():
            if prior_name == "core_neff":
                para_space.append(Real(low, high, name=prior_name))
            else:
                para_space.append(Real(low, high, name=prior_name))
        # para_space = [Real(low, high, name=prior_name) for prior_name, (low, high) in param_range.items()]
    
    if simulate_tf_metric and opt_checkpoint_path.exists() and not bestvals:
        opt = load(opt_checkpoint_path)
        print(f"Loaded optimiser checkpoint from {opt_checkpoint_path}")
    else:
        opt = Optimizer(
            dimensions=para_space, # Parameter search space (bounds + types)
            base_estimator="GP", # Surrogate model (Gaussian Process)
            acq_func="EI", # Acquisition function (chooses next point)
            acq_func_kwargs={"xi": 0.8}, # EI exploration strength (higher = more exploration)
            acq_optimizer="sampling", # How acquisition is maximised (random sampling)
            # acq_optimizer_kwargs = {"n_points": 10000}, # Number of samples used to find best next point
            random_state=None, # Random seed (None = non-reproducible)
            n_initial_points=50 # Number of random iterations before BO starts
        )

    # if true, run optimisation testing the loss metric
    if simulate_tf_metric:
        all_results = load_checkpoint_npy(results_checkpoint_path)
        wave_logs = []
        start_iter = len(all_results)

        if start_iter > 0:
            print(f"Resuming from iteration {start_iter + 1}")
        else:
            print("No existing results checkpoint found. Starting fresh.")

        if bestvals:
            # checking if femsim files exist. If they do, continue. If not, generate the,
            femSIM_file_example = "FemSim_File_DET_1.5_LP01_core_diam_6.500000_core_neff_1.447962_Taper_L_50000.000000_ex.m00"
            if not fem_fields_present(femSIM_file_example):
                print("No suitable FemSIM field profiles detected. Generating...")
                param_names = ["core_diam", "core_neff"]
                params = [variable_params[k] for k in param_names]
                run_tf_multproc(params, 1,simulation_val, custom_priors, mode_vals, 
                                radial_mode_vals,taper_min, taper_max, fem = True, gridding=gridding)
            
            for batch_idx, para_set in enumerate(para_space):
                param_batch = para_set
                print("Trying " + ", ".join(
                f"Core neff: {param_batch[l]:.3f}" if text == "core_neff"
                else f"{text}: {param_batch[l]:.3f}"
                for l, text in enumerate(variable_params.keys())
                ))         

                result_batch, df_wave_log = run_all_modes_for_params(param_batch, batch_idx + 1, 
                                                                        simulation_val, custom_priors, taper_min, 
                                                                        taper_max, gridding = gridding)
                
                print(f"Iteration {batch_idx + 1}: {result_batch:.6f}")
                for w, group in df_wave_log.groupby("Wavelength"):
                    row = group.iloc[0]  # safe: all rows for this wavelength share same loss terms

                    print(
                        f"  wavelength ={w:.3f} µm | "
                        f"(a, b, c, d)=("
                        f"{row['Loss_a']:.6g}, "
                        f"{row['Loss_b']:.6g}, "
                        f"{row['Loss_c']:.6g}, "
                        f"{row['Loss_d']:.6g})"
                    )

                # convert df_wave_log to numpy array
                loss_terms = df_wave_log[["Wavelength","Loss_a","Loss_b","Loss_c","Loss_d"]].to_numpy()
                
                core_amp_cols = sorted(
                    [c for c in df_wave_log.columns if c.startswith("Core_") and c.endswith("_Amp")],
                    key=lambda s: int(s.split("_")[1])
                )

                core_phase_cols = sorted(
                    [c for c in df_wave_log.columns if c.startswith("Core_") and c.endswith("_Phase")],
                    key=lambda s: int(s.split("_")[1])
                )
                amp = df_wave_log[core_amp_cols].to_numpy(dtype=float)
                phase = df_wave_log[core_phase_cols].to_numpy(dtype=float)

                extra_amp_cols = sorted(
                    [c for c in df_wave_log.columns if c.startswith("Extra_") and c.endswith("_Amp")],
                    key=lambda s: int(s.split("_")[1])
                )

                extra_phase_cols = sorted(
                    [c for c in df_wave_log.columns if c.startswith("Extra_") and c.endswith("_Phase")],
                    key=lambda s: int(s.split("_")[1])
                )

                ex_amp = df_wave_log[extra_amp_cols].to_numpy(dtype=float)
                ex_phase = df_wave_log[extra_phase_cols].to_numpy(dtype=float)

                number_of_guided_modes = df_wave_log["Guided Modes"].to_numpy()
                iter_result = {
                    'params': param_batch,
                    'result': result_batch,
                    'Iteration': batch_idx + 1,
                    "Loss Metric": loss_terms,
                    "Number of Guided Modes": number_of_guided_modes,
                    "Core Amplitudes": amp,
                    "Core Phases": phase,
                    "Extra Core Amplitudes": ex_amp,
                    "Extra Core Phases": ex_phase
                }

                all_results.append(iter_result)
                atomic_save_npy(all_results, results_checkpoint_path)
            return all_results
        
        # checking if femsim files exist. If they do, continue. If not, generate the,
        femSIM_file_example = "FemSim_File_DET_1.5_LP01_core_diam_6.500000_core_neff_1.447962_Taper_L_50000.000000_ex.m00"
        if not fem_fields_present(femSIM_file_example):
            print("No suitable FemSIM field profiles detected. Generating...")
            param_names = ["core_diam", "core_neff"]
            params = [variable_params[k] for k in param_names]
            run_tf_multproc(params, 1,simulation_val, custom_priors, mode_vals, 
                            radial_mode_vals,taper_min, taper_max, fem = True, gridding=gridding)

        for batch_idx in range(start_iter, total_calls):
            # ask for 1 set of parameter vectors only to prevent daemonic process having children 
            param_batch = opt.ask() 

            print("Trying " + ", ".join(
                f"Core neff: {param_batch[l]:.3f}" if text == "core_neff"
                else f"{text}: {param_batch[l]:.3f}"
                for l, text in enumerate(variable_params.keys())
            ))         
            
            result_batch, df_wave_log = run_all_modes_for_params(param_batch, batch_idx + 1, 
                                                                      simulation_val, custom_priors, taper_min, 
                                                                      taper_max, gridding = gridding)

            # tell optimiser the performance of the chosen parameters
            opt.tell(param_batch, result_batch)

            # log iteration of parameters, store for later use, rows are the wavelength used
            print(f"Iteration {batch_idx + 1}: {result_batch:.6f}")

            for w, group in df_wave_log.groupby("Wavelength"):
                row = group.iloc[0]  # safe: all rows for this wavelength share same loss terms

                print(
                    f"  wavelength ={w:.3f} µm | "
                    f"(a, b, c, d)=("
                    f"{row['Loss_a']:.6g}, "
                    f"{row['Loss_b']:.6g}, "
                    f"{row['Loss_c']:.6g}, "
                    f"{row['Loss_d']:.6g})"
                )

            # convert df_wave_log to numpy array
            loss_terms = df_wave_log[["Wavelength","Loss_a","Loss_b","Loss_c","Loss_d"]].to_numpy()
            
            core_amp_cols = sorted(
                [c for c in df_wave_log.columns if c.startswith("Core_") and c.endswith("_Amp")],
                key=lambda s: int(s.split("_")[1])
            )

            core_phase_cols = sorted(
                [c for c in df_wave_log.columns if c.startswith("Core_") and c.endswith("_Phase")],
                key=lambda s: int(s.split("_")[1])
            )

            amp = df_wave_log[core_amp_cols].to_numpy(dtype=float)
            phase = df_wave_log[core_phase_cols].to_numpy(dtype=float)

            extra_amp_cols = sorted(
                    [c for c in df_wave_log.columns if c.startswith("Extra_") and c.endswith("_Amp")],
                    key=lambda s: int(s.split("_")[1])
                )

            extra_phase_cols = sorted(
                    [c for c in df_wave_log.columns if c.startswith("Extra_") and c.endswith("_Phase")],
                    key=lambda s: int(s.split("_")[1])
                )

            ex_amp = df_wave_log[extra_amp_cols].to_numpy(dtype=float)
            ex_phase = df_wave_log[extra_phase_cols].to_numpy(dtype=float)

            number_of_guided_modes = df_wave_log["Guided Modes"].to_numpy()

            iter_result = {'params': param_batch, 
                           'result': result_batch, 
                            'Iteration': batch_idx + 1, 
                            "Loss Metric": loss_terms, 
                            "Number of Guided Modes": number_of_guided_modes,
                            "Core Amplitudes": amp, 
                            "Core Phases": phase,
                            "Extra Core Amplitudes": ex_amp, 
                            "Extra Core Phases": ex_phase
                            }
            all_results.append(iter_result)
            # save simulation and optimisation results instantaneously
            atomic_save_npy(all_results, results_checkpoint_path)
            dump(opt, opt_checkpoint_path, store_objective=False)
        return all_results
    
    # if false, run tf code for the template parameters
    else:
        param_names = ["core_diam", "core_neff"]
        params = [variable_params[k] for k in param_names]

        # checking if femsim files exist. If they do, continue. If not, generate the,
        femSIM_file_example = "FemSim_File_DET_*_ex.m00"
        if not fem_fields_present(femSIM_file_example):
            print("No suitable FemSIM field profiles detected. Generating...")
            param_names = ["core_diam", "core_neff"]
            params = [variable_params[k] for k in param_names]
            run_tf_multproc(params, 1,simulation_val, custom_priors, mode_vals, 
                            radial_mode_vals,taper_min, taper_max, fem = True, gridding=gridding)
            
        if gridding:
            grid_size_range, tf_list, _ = run_tf_multproc(params, 1, simulation_val, custom_priors, mode_vals, radial_mode_vals,taper_min, taper_max, gridding)
            return grid_size_range, tf_list, _
        else:
            tf_list = run_tf_multproc(params, 1,simulation_val, custom_priors, mode_vals, radial_mode_vals,taper_min, taper_max, gridding)
        return tf_list