import numpy as np, os, shutil, csv
import subprocess, json, time
from pathlib import Path
from skopt import Optimizer, dump, load
from skopt.space import Real, Categorical
import multiprocessing as mp
import datetime
from collections import defaultdict
from Circuit_Properties import *
from Functions import *
from HexProperties import *
from template import * 
from rstools import RSoftUserFunction, RSoftCircuit # type:ignore

class RSoftSim:
    def __init__(self):
        self.sym = {}
        self.last_sim_health = {"status": "OK", "message": "", "simulation": ""}

    def dummy_tf_result(self, simulation_val):
        if simulation_val.get("mon_type", Launch_params["mon_type"]) == "port_mon":
            skip_core = simulation_val.get("skip_core", Simulation_params["skip_core"])
            core_num = simulation_val.get("core_num", Simulation_params["core_num"])
            if skip_core is not None:
                core_monitors = core_num - len(skip_core)
            else:
                core_monitors = core_num
            extra_monitors = len(get_higher_order_modes(simulation_val))
            return np.zeros(1 + 2 * (core_monitors + extra_monitors), dtype=float)

        monitor_count = simulation_val.get("core_num", Simulation_params["core_num"]) + 1
        return np.zeros(monitor_count, dtype=float)

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
    

    def RunRSoftSim(self, name_tag, femsim_name_tag, fixed, 
                    vars, fixed_length, param_range, simulation_val, 
                    csv_path, json_config, prior_space_pid, wave, run_tag, fem=False):
        
        filename = f"{name_tag}.ind"
        filename_FS = f"{femsim_name_tag}.ind"
        sim_tool = simulation_val.get("sim_tool", RSoft_params["sim_tool"])
        iter_number = simulation_val["iter_num"]
        expected_monitor_files = []
        pid_csv = os.getpid()
        self.last_sim_health = {
            "status": "OK",
            "message": "",
            "simulation": f"{name_tag} | {femsim_name_tag}",
        }
        # Run RSoft simulation
        if sim_tool == "ST_BEAMPROP":
            prefix_BP   = f"prefix={name_tag}"
            if fem:
                folder_BP = "FemSIM_DET"
            else:
                folder_BP   = f"BP_SimulationNum_{iter_number}"
            prefix_FS = f"prefix={femsim_name_tag}"

            results_folder = create_folders(folder_BP, "Desktop")

            # # we want to copy the important files in Onedrive for backup purposes
            # onedrive_results_folder = create_folders(folder_BP, "Onedrive")

            # we want to copy the important files into a separate folder for backup purposes
            backup_results_folder = create_folders(folder_BP, "analysis_path")

            try:
                subprocess.run(
                    [r"C:\Keysight\PhotonicSolutions\2026\RSoft\bin\femsim.exe", "-hide", filename_FS, prefix_FS, "wait=0"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                # functions to ensure that the correct monitor files are being used, and that they exist prior to BPM running
                expected_monitor_files = extract_monitor_files_from_ind(filename)

                wait_for_files_stable(
                    expected_monitor_files,
                    timeout=300,
                    interval=1.0,
                    stable_checks=3,
                    min_size=390*1024, # waits for files > 390 KB to be made
                    require_rsoft_header=True
                )

                subprocess.run(
                    [r"C:\Keysight\PhotonicSolutions\2026\RSoft\bin\bsimw32.exe", "-hide", filename, prefix_BP, "wait=0"],
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
            except (TimeoutError, ValueError) as e:
                print(e)
                self.last_sim_health = {
                    "status": "TIMEOUT",
                    "message": str(e).replace("\r", " ").replace("\n", " | "),
                    "simulation": f"{name_tag} | {femsim_name_tag}",
                }
                if Simulation_params["metric"] == "TF":
                    return self.dummy_tf_result(simulation_val), 0.0, results_folder, pid_csv
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
        # Move all output files immediately after simulation
        
        # files to copy to backup location
        backup_filename = name_tag + ".ind"
        backup_filename_results = name_tag + "_mon.dat"
        backup_field_results = name_tag + ".fld"
        backup_femsim_filename_results = femsim_name_tag + ".ind"
        backup_neff_csv_path = Path(backup_results_folder) / f"{wave}_Guided Modes_{run_tag}.csv"
        neff_csv_path = Path(results_folder) / f"{wave}_Guided Modes_{run_tag}.csv"

        files_to_copy_to_backup = {
           backup_filename, backup_filename_results, backup_field_results,
           backup_femsim_filename_results
        }

        files_to_move = [
            filename,
            filename_FS,
            femsim_name_tag,
            *[f for f in expected_monitor_files if os.path.basename(f).startswith("REF_")],
            csv_path,
            json_config,
            prior_space_pid,
            neff_csv_path]
        
        # Normalize to basenames in case paths are used
        files_to_move = [os.path.basename(f) for f in files_to_move]
        files_to_copy_to_backup = {os.path.basename(k) for k in files_to_copy_to_backup}

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
            mon_path = Path(f"{name_tag}_mon.dat")
            nef_path = Path(f"{femsim_name_tag}.nef")

            try:
                wait_for_files_stable(
                    [mon_path, nef_path],
                    timeout=120,
                    interval=0.5,
                    stable_checks=2,
                    min_size=10
                )
            except (TimeoutError, ValueError) as e:
                print(e)
                self.last_sim_health = {
                    "status": "TIMEOUT",
                    "message": str(e).replace("\r", " ").replace("\n", " | "),
                    "simulation": f"{name_tag} | {femsim_name_tag}",
                }
                if Simulation_params["metric"] == "TF":
                    return self.dummy_tf_result(simulation_val), 0.0, results_folder, pid_csv
                return -1e6

        # Read .mon file from moved location
        uf = RSoftUserFunction()
        uf.read(str(mon_path))

        # make guided modes save to output csv
        uf_neff = RSoftUserFunction()
        uf_neff.read(str(nef_path))
        x_neff, y_neff = uf_neff.get_arrays()
        x_all, y_all, z_all = uf.get_arrays()
        del uf
        del uf_neff

        guided_mask = y_neff > fixed_params["cladding_neff"]
        guided_neff = np.array(y_neff[guided_mask])

        with open(neff_csv_path, mode="w", newline="") as f_neff:
            writer = csv.writer(f_neff)
            writer.writerow(["Mode_Index", "n_eff", "Wavelength (um)", "PID", "Run Tag"])
            for idx, nval in enumerate(guided_neff, start=1):
                writer.writerow([idx, nval, wave, pid_csv, run_tag])

        os.makedirs(backup_neff_csv_path.parent, exist_ok=True)
        with open(backup_neff_csv_path, mode="w", newline="") as f_neff_od:
            writer = csv.writer(f_neff_od)
            writer.writerow(["Mode_Index", "n_eff", "Wavelength (um)", "PID", "Run Tag"])
            for idx, nval in enumerate(guided_neff, start=1):
                writer.writerow([idx, nval, wave, pid_csv, run_tag])

        owned_files = []
        for file in os.listdir():
            if (
                file in files_to_move
                or file == f"{name_tag}.ind"
                or file == f"{femsim_name_tag}.ind"
                or file.startswith(f"{name_tag}_")
                or file.startswith(f"{name_tag}.")
                or file.startswith(f"{femsim_name_tag}_")
                or file.startswith(f"{femsim_name_tag}.")
            ):
                owned_files.append(file)

        # move files AFTER they have been read
        for file in owned_files:
            if file in files_to_copy_to_backup:
                os.makedirs(backup_results_folder, exist_ok=True)
                copy_when_available(file, os.path.join(backup_results_folder, file))

            # move everything to the desktop
            move_when_available(file, os.path.join(results_folder, file))

        if Launch_params["mon_type"] == "pathway_mon":
            num_monitors = z_all.shape[1]

            # Write throughput CSV to same folder
            csv_tag = f"Throughput_{name_tag}.csv"
            csv_pathway = Path(results_folder) / csv_tag
            csv_pathway_backup = Path(backup_results_folder) / csv_tag

            with open(csv_pathway, mode="w", newline="") as file:
                writer = csv.writer(file)
                header = ["x"] + [f"Monitor_{i}" for i in range(num_monitors)]
                writer.writerow(header)
                for i in range(z_all.shape[0]):
                    row = [x_all[i]] + [np.real(z_all[i, j]) for j in range(z_all.shape[1])]
                    writer.writerow(row)

            os.makedirs(csv_pathway_backup.parent, exist_ok=True)
            with open(csv_pathway_backup, mode="w", newline="") as file_backup:
                writer = csv.writer(file_backup)
                header = ["x"] + [f"Monitor_{i}" for i in range(num_monitors)]
                writer.writerow(header)
                for i in range(z_all.shape[0]):
                    row = [x_all[i]] + [np.real(z_all[i, j]) for j in range(z_all.shape[1])]
                    writer.writerow(row)
        
        elif Launch_params["mon_type"] == "port_mon":
            if Simulation_params["skip_core"] is not None:
                num_monitors = (z_all.shape[1] - (Simulation_params["core_num"] - len(Simulation_params["skip_core"])))
            else:
                num_monitors = (z_all.shape[1] - Simulation_params["core_num"])

            # Write throughput CSV to same folder
            csv_tag = f"Throughput_{name_tag}.csv"
            csv_pathway = Path(results_folder) / csv_tag
            csv_pathway_backup = Path(backup_results_folder) / csv_tag

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

            os.makedirs(csv_pathway_backup.parent, exist_ok=True)
            with open(csv_pathway_backup, mode="w", newline="") as file_backup:
                writer = csv.writer(file_backup)
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

        if Simulation_params["metric"] == "TF":
            if Launch_params["mon_type"] == "port_mon":
                transfer_vector, throughput = transfer_matrix_component(csv_pathway, row, port_mon = True)
                return transfer_vector, -throughput, results_folder, pid_csv
            else:
                transfer_vector, throughput = transfer_matrix_component(csv_pathway, row)
                return transfer_vector, -throughput, results_folder, pid_csv

    def build_circuit(self, params, build_tf, json_config, csv_path, simulation_val, 
                      prior_space_pid, wave, delta_index_at_reference_wavelength, run_tag, fem=False): 
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

        # need two circuits:
        #   1) BPM: to run the BPM simulations - special core location is set by siulation_val
        #   2) FemSIM: to calculate the FemSIM files needed to correctly run BPM - special core location is fixed to the centre
        bp_circuit = RSoftCircuit()
        fs_circuit = RSoftCircuit()
        self.circuit = bp_circuit

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
            bp_circuit.set_symbol(key, val)
            fs_circuit.set_symbol(key, val)

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
        core_num = sim_param["core_num"]
        core_name = [f"core_{n}" for n in range(1, core_num + 1)]
        structure = Simulation_params["Structure"]

        cladd_diam = fixed["MCFCladd"]
        cladding_beg_dims = (cladd_diam / taper, cladd_diam / taper) 
        cladding_end_dims = (cladd_diam , cladd_diam)

        bp_core_to_monitor = Simulation_params["core_to_monitor"]
        fs_core_positions = self.cladd_positions if simulation_val["skip_core"] is not None and self.cladd_positions is not None else self.core_positions
        fs_core_to_monitor = centre_core_index(fs_core_positions)
        bp_add_cladding_to_cores = Simulation_params["add_cladding_to_cores"]
        if bp_add_cladding_to_cores is None:
            fs_add_cladding_to_cores = None
        else:
            fs_add_cladding_to_cores = sorted(
                ({*bp_add_cladding_to_cores, bp_core_to_monitor - 1} - {fs_core_to_monitor - 1})
            )
        bp_core_beg_dims_list, bp_core_end_dims_list, bp_core_params = core_layout_for_special_core(bp_core_to_monitor, sim_param, simulation_val, core_name,param_dict,delta_index_at_reference_wavelength, taper)
        fs_core_beg_dims_list, fs_core_end_dims_list, fs_core_params = core_layout_for_special_core(fs_core_to_monitor, sim_param, simulation_val, core_name,param_dict,delta_index_at_reference_wavelength, taper)

        core_params.clear()
        core_params.update(bp_core_params)

        # functions to generate the core layout, either a standard fibre or a complicated photonic lantern setup (either in hex, pent or circular geometry)
        if structure == "Fibre":
            path_num, core_positions, core_final_dims_list = build_fibre(bp_circuit, 0, self.core_positions,
                        core_name, Taper_L, bp_core_beg_dims_list, bp_core_end_dims_list)
            fs_path_num, fs_core_positions, fs_core_final_dims_list = build_fibre(fs_circuit, 0, fs_core_positions,
                        core_name, Taper_L, fs_core_beg_dims_list, fs_core_end_dims_list)

        elif structure == "PL":
            path_num, core_positions, core_final_dims_list = build_PL(bp_circuit, 0, self.core_positions,
                    core_name, taper, Taper_L,
                    cladding_beg_dims, cladding_end_dims,
                    # capillary_beg_dims, capillary_end_dims,
                    bp_core_beg_dims_list, bp_core_end_dims_list,
                    simulation_val,self.cladd_positions,
                    add_cladding_to_cores=bp_add_cladding_to_cores,
                    core_to_monitor=bp_core_to_monitor)
            fs_path_num, fs_core_positions, fs_core_final_dims_list = build_PL(fs_circuit, 0, fs_core_positions,
                    core_name, taper, Taper_L,
                    cladding_beg_dims, cladding_end_dims,
                    fs_core_beg_dims_list, fs_core_end_dims_list,
                    simulation_val,fs_core_positions,
                    add_cladding_to_cores=fs_add_cladding_to_cores,
                    core_to_monitor=fs_core_to_monitor)
            
        if simulation_val["launch_type"] == LaunchType.SM:
            launch_mode = simulation_val["launch_mode"]
            launch_mode_radial = simulation_val["launch_mode_radial"]
            param_string = "_".join(f"{key}_{val:.6f}" for key, val in param_dict.items())
            name_tag = f"{wave}_LP{launch_mode}{launch_mode_radial}_{param_string}"
        else:
            grid = simulation_val["grid_size"] # use only when trying to find the optimal gridding to run BeamPROP in.
            launch_mode = simulation_val.get("launch_mode", "MM")
            launch_mode_radial = simulation_val.get("launch_mode_radial", "")
            param_string = "_".join(f"{key}_{val:.6f}" for key, val in param_dict.items())
            name_tag = f"{wave}_LP{launch_mode}{launch_mode_radial}_Grid{grid}_{param_string}"
        if run_tag:
            name_tag = f"{name_tag}_{run_tag}"
        
        if fem:
            self.sym["Name"] = name_tag
            bp_circuit.write(f"{name_tag}.ind")
            femsim_name_tag = f"FemSim_File_DET_{name_tag}"
            fs_circuit.write(f"{femsim_name_tag}.ind")
        else:
            self.sym["Name"] = name_tag
            bp_circuit.write(f"{name_tag}.ind")

            # create separate file for FemSIM field determination
            femsim_name_tag = f"FS_{name_tag}"
            fs_circuit.write(f"{femsim_name_tag}.ind")

        """
        Append all pathway, monitor, and launch field blocks based 
        on launch parameters. 
        """ 
        
        AddHack(name_tag, femsim_name_tag, launch, 
                path_num - 1, param_dict, simulation_val, wave,core_positions,fem=fem,
                core_params_bp=bp_core_params, core_params_fs=fs_core_params,
                fs_core_to_monitor=fs_core_to_monitor,
                bp_add_cladding_to_cores=bp_add_cladding_to_cores,
                fs_add_cladding_to_cores=fs_add_cladding_to_cores,
                fs_core_num=fs_path_num - 1,
                fs_core_positions=fs_core_positions)
        
        force_file_to_disk(f"{name_tag}.ind")
        force_file_to_disk(f"{femsim_name_tag}.ind")
        '''
        Manual setup to loop through a list of values. Runs the terminal line that will initiate RSoft and will calculate the 
        metric to test.
        All output files will appear in a subfolder on the Desktop (windows)
        '''

        if Simulation_params['metric'] != 'TF':
            average_throughput, res_folder, pid_csv = self.RunRSoftSim(name_tag, femsim_name_tag, fixed, 
                                                  vars, fixed_length, param_range, 
                                                  simulation_val, csv_path, json_config, 
                                                  prior_space_pid, wave, run_tag)
            return average_throughput, res_folder, pid_csv
        else: 
            transfer_vector, average_throughput, res_folder, pid_csv = self.RunRSoftSim(name_tag, femsim_name_tag, fixed, 
                                                                   vars, fixed_length, param_range, 
                                                                   simulation_val, csv_path, json_config, 
                                                                   prior_space_pid, wave, run_tag, fem = fem)
            return transfer_vector, average_throughput, res_folder, pid_csv

    def RunRSoft(self, simulation_val, prior_space_pid, wave, 
                 delta_index_at_reference_wavelength, csv_path, 
                 json_config, pid, fem=False, simulate=False, build_tf = True, run_tag=None): #csv_path, 
        '''
        Multiprocessing must to be run outside of a Jupyter cell or it will silently 
        fail/infinitely loop on the first batch
        '''

        # write in the values within simulation_val
        overwrite_template_val(json_config)
        
        # generate the positions of the cores. 
        self.generate_core_positions()

        # remove old results
        if os.path.exists(csv_path):
            os.remove(csv_path)

        if simulate:
            self.MultProc(build_tf, json_config, csv_path, simulation_val, prior_space_pid) 
            return  

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
            seed_result, res_folder, pid_csv = self.build_circuit(seed_params, build_tf, json_config, csv_path, 
                                                                  simulation_val, prior_space_pid, wave, 
                                                                  delta_index_at_reference_wavelength, run_tag)
            tf_vector = None
        else:
            tf_vector, seed_result, res_folder, pid_csv = self.build_circuit(seed_params, build_tf, json_config, csv_path, 
                                                                             simulation_val, prior_space_pid, wave, 
                                                                             delta_index_at_reference_wavelength, run_tag, fem = fem)

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
            name_tag = self.sym["Name"],
            run_tag = run_tag,
            health_batch=[self.last_sim_health]
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
    task_idx = None
    if len(arg_list) in (18, 17, 16, 14):
        task_idx, *arg_list = arg_list

    if len(arg_list) == 15:
        sim_val, custom_priors, wave, m, rm, cand_idx, param, fem, taper_min, taper_max, gridding, delta_index_at_reference_wavelength, core_neff_idx, wave_indices, iteration_num = arg_list
    elif len(arg_list) == 13:
        sim_val, custom_priors, wave, m, rm, cand_idx, param, fem, taper_min, taper_max, gridding, iteration_num = arg_list
        wave_indices = None
    elif len(arg_list) == 9:
        sim_val, custom_priors, gr, cand_idx, param, taper_min, taper_max, gridding, iteration_num = arg_list
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
        sim_val = copy.deepcopy(sim_val)
        sim_val["launch_mode"] = m
        sim_val["launch_mode_radial"] = rm
        sim_val["iter_num"] = iteration_num
        sim_val["cand_idx"] = cand_idx
        if fem:
            sim_val["Fem_present"] = False

        if wave_indices is None and sim_val["sellmeier"]:
            stored_data = pd.read_csv(r"C:\Users\RSoft Things\OneDrive - The University of Sydney (Students)\Apps\VSCode\Sellmeier_Considerations\Sellmeier_vals.csv")
            wave_indices = get_wavelength_dependent_indices(wave, sim_val, fixed_params, stored_data)

        if wave_indices is not None:
            sim_val["core_neff"] = wave_indices["non_ms_core_neff"]
            fixed_params["cladding_neff"] = wave_indices["cladding_neff"]
            Launch_params["cladding_neff"] = fixed_params["cladding_neff"]
            RSoft_params["background_index"] = wave_indices["capillary_neff"]

        # if sim_val["add_cladding_to_cores"] is not None:  
            # fixed_params["core_cladding_neff"] = core_cladding_refractive_index

        # write core diameter properties to simulation_val and set special core properties to None for SKOPT to overwrite, if applicable
        assign_core_properties(sim_val)
        core_to_monitor = sim_val["core_to_monitor"]

        # dynamically load parameters to vary
        param_names = list(variable_params.keys())

        # Assign the new params directly to the special core
        for pname, pval in zip(param_names, param):
            sim_val[f"core_{core_to_monitor}"][pname] = pval
            variable_params[pname] = pval
        
        if fem:
            variable_params["core_neff"] = sim_val["core_neff"]

        sim_val["free_space_wavelength"] = wave

        # dump configuration paras into json for use later
        # pid is needed to avoid cross-talking between files created/used in multiprocessing tasks
        pid = os.getpid()
        task_suffix = f"_t{task_idx}" if task_idx is not None else ""
        # run_tag format: i=iteration, c=candidate (for multiple parameter batches), p=pid, task_suffix = job number
        run_tag = f"i{iteration_num}_c{cand_idx}_p{pid}{task_suffix}"

        # prior space identifier
        prior_space_pid = f"{run_tag}_prior_space.json"
        # launch config identifier
        code_config = f"{run_tag}_launch_config.json"
        # csv to store results later
        optimizer_result = f"{run_tag}_Optimizer_Result.csv"
        param_num = f"cand_{cand_idx}_wave_{wave}_LP{m}{rm}"
    
    # dump launch config profile into simulation_val
    with open(code_config, "w") as launch_config:
        json.dump(sim_val, launch_config, indent = 2)

    build_tf = True

    sim = RSoftSim()
    sim.init_priors(prior_space_pid, build_tf, custom_priors)

    # run simulation
    results_folder, res_folder, pid_csv = sim.RunRSoft(sim_val, prior_space_pid, wave, delta_index_at_reference_wavelength[cand_idx],
                                                       csv_path = optimizer_result, json_config = code_config, pid = pid, fem = fem, 
                                                       build_tf = build_tf, run_tag=run_tag)

    # Load results
    best_para_log = os.path.join(results_folder, f"best_params_log_{run_tag}.csv")    
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

    health = {
        "status": "OK",
        "message": "",
        "simulation": "",
    }
    if "Simulation Health" in data.columns:
        health["status"] = str(data["Simulation Health"].iloc[0])
    if "Simulation Message" in data.columns:
        health["message"] = "" if pd.isna(data["Simulation Message"].iloc[0]) else str(data["Simulation Message"].iloc[0])
    if "Failed Simulation" in data.columns:
        health["simulation"] = "" if pd.isna(data["Failed Simulation"].iloc[0]) else str(data["Failed Simulation"].iloc[0])

    # Call plotting function
    plotting_optimizer_results(data, param_names, tf=tf_vectors, plot= False)
    return (cand_idx, param_num, tf_vectors, wave, res_folder, pid_csv, run_tag, health)

def run_tf_multproc(params, iteration_num, simulation_val, custom_priors,  taper_min, taper_max, fem = False, gridding = False):

    tf_list = []
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params[np.newaxis, :]

    stored_data = pd.read_csv(r"C:\Users\RSoft Things\OneDrive - The University of Sydney (Students)\Apps\VSCode\Sellmeier_Considerations\Sellmeier_vals.csv")
    _, refractive_index_at_reference_wave = find_nearest(stored_data["Wavelength (um)"].to_numpy(), 1.5)
    Silica_refractive_index_at_reference_wavelength = stored_data["SiO2"].to_numpy()[refractive_index_at_reference_wave]

    if "core_neff" in variable_params:
        # get index of core_neff in variable_params
        for i, key in enumerate(variable_params):
            if key == "core_neff":
                core_neff_idx = i

        # Keep the MS core offset relative to the configured cladding at the
        # Sellmeier reference wavelength, then add it to each wavelength's
        # cladding later when building the .ind file.
        cladding_ref_at_reference = get_wavelength_dependent_indices(
            1.5,
            simulation_val,
            fixed_params,
            stored_data
        )["cladding_neff"]
        delta_index_at_reference_wavelength = params[:, core_neff_idx] - cladding_ref_at_reference
    else:
        core_neff_idx = None
        cladding_ref_at_reference = get_wavelength_dependent_indices(
            1.5,
            simulation_val,
            fixed_params,
            stored_data
        )["cladding_neff"]
        delta_index_at_reference_wavelength = np.full(
            params.shape[0],
            simulation_val.get("core_neff", fixed_params.get("core_neff")) - cladding_ref_at_reference
        )

    # Include sine/cosine orientations for each supported LP family.
    mode_labels = np.array(list(LP_mode_rsoft_dict.keys()))
    mode_vals = np.array([LP_mode_rsoft_dict[m][0] for m in mode_labels])
    radial_mode_vals = np.array([LP_mode_rsoft_dict[m][1] for m in mode_labels])
    res_dict = []

    # dynamically generate an array of arrays listing the supported number of modes per free space wavelength
    for w in simulation_val["free_space_wavelength"]:
        b_arr = []
        wave_indices = get_wavelength_dependent_indices(w, simulation_val, fixed_params, stored_data)
        cladding_refractive_index_at_wavelength = wave_indices["cladding_neff"]
        capillary_refractive_index_at_wavelength = wave_indices["capillary_neff"]

        numerical_apeture = ofiber.numerical_aperture(cladding_refractive_index_at_wavelength, capillary_refractive_index_at_wavelength)
        v_number = ofiber.V_parameter((fixed_params["MCFCladd"]/fixed_params["taper"])/2, numerical_apeture, w)

        # calculate and print out the propagation constants
        for ell in range(Simulation_params["max_ell"]+1):
            all_b = ofiber.LP_mode_values(v_number, ell)
            for i, b in enumerate(all_b):
                b_arr.append(b)

        res = {
            "Wavelength": w,
            "Cladding Index": cladding_refractive_index_at_wavelength,
            "Capillary Index": capillary_refractive_index_at_wavelength,
            "Index Values": wave_indices,
            "Propagation Constants":np.array(b_arr) # organised as LP01, LP02, LP11, LP21, LP31,..., increasing l in LPln
        }
        res_dict.append(res)
        results_df = pd.DataFrame(res_dict)

    total = []
    for k in range(results_df.shape[0]):
        total_spatial_modes_supported = results_df["Propagation Constants"].iloc[k].shape[0]
        count = 0
        for mode_num, rot_val in enumerate(LP_mode_dict.values(), start=1):
            count += rot_val
            if mode_num == total_spatial_modes_supported:
                total_mode = {
                    "Total supported scalar modes": min(count, len(mode_labels))
                }
                total.append(total_mode)
    total_df = pd.DataFrame(total)
    # combine results
    combined_df = results_df.join(total_df)
    # now determine mode_vals and radial_mode_vals for each supported number of modes
    rsoft_modes = []
    for n in range(combined_df.shape[0]):
        total_modes_to_inject = combined_df["Total supported scalar modes"].iloc[n]
        injected_mode_vals = mode_vals[:total_modes_to_inject]
        injected_radial_mode_vals = radial_mode_vals[:total_modes_to_inject]

        injected_res = {
            "mode_vals": injected_mode_vals,
            "radial_mode_vals": injected_radial_mode_vals
        }
        rsoft_modes.append(injected_res)

    rsoft_modes_df = pd.DataFrame(rsoft_modes)

    final_df = combined_df.join(rsoft_modes_df)

    if not gridding:
        args_list = [
            (
                simulation_val,
                custom_priors,
                row["Wavelength"],
                int(m),
                int(rm),
                cand_idx,
                param,
                fem,
                taper_min,
                taper_max,
                gridding,
                delta_index_at_reference_wavelength,
                core_neff_idx,
                row["Index Values"],
                iteration_num,
            )
            for cand_idx, param in enumerate(params)
            for _, row in final_df.iterrows()
            for m, rm in zip(row["mode_vals"], row["radial_mode_vals"])
        ]
    else:
        # run gridding determination
        grid_size_list = np.arange(0.1, 2.1, 0.1)
        grid_size_range = np.linspace(0.1, 2.0, len(grid_size_list))

        if simulation_val["launch_type"] != "LAUNCH_MULTIMODE" and Launch_params["launch_random_set"] != 0:
            raise Exception("Launch type must be multimode with a fixed random set when determining optimal grid sizes!")
        
        # initialise global parent argument list
        args_list = [(simulation_val, custom_priors, gr, params, taper_min, taper_max, gridding, iteration_num) for gr in grid_size_range] 

        
    if not gridding:
        args_list = [(task_idx, *args) for task_idx, args in enumerate(args_list)]

    # 'spawn' starts separate processes with separate memory; each job must receive all state it needs.
    with mp.get_context("spawn").Pool(processes=int(30-Simulation_params["number_of_rsoft_instances"])) as pool:
        results = pool.map(multiple_mode_tf, args_list) # (param_num, tf_vectors, wave, res_folder, pid_csv)
    
    res_folder = results[-1][4]
    for result_item in results:
        if len(result_item) >= 8:
            cand_idx, param, result, wave, _, csv_pid, run_tag = result_item[:7]
            health = result_item[7]
        elif len(result_item) == 7:
            cand_idx, param, result, wave, _, csv_pid, run_tag = result_item
            health = {"status": "OK", "message": "", "simulation": ""}
        else:
            raise ValueError(f"Unexpected worker result shape: expected 7 or 8+ values, got {len(result_item)}")
        tf_list.append((cand_idx, param, result, wave, csv_pid, run_tag, health))
        
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

    if gridding:
        # run gridding determination
        grid_size_range, tf_list, res_folder = run_tf_multproc(params, iteration_num, simulation_val, 
                                                               custom_priors, 
                                                               taper_min, taper_max, gridding=gridding)
        return grid_size_range, tf_list
    
    # multiprocessing in here, we only want multiprocessing in this function and not in main_optimizer!!!!
    tf_list, res_folder = run_tf_multproc(params, iteration_num, simulation_val, 
                                            custom_priors, 
                                            taper_min, taper_max, gridding)
    
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params[np.newaxis, :]

    by_candidate = defaultdict(list)
    for item in tf_list:
        if len(item) >= 7:
            cand_idx, param_num, tf_vec, wave, csv_pid, run_tag = item[:6]
            health = item[6]
        elif len(item) == 6:
            cand_idx, param_num, tf_vec, wave, csv_pid, run_tag = item
            health = {"status": "OK", "message": "", "simulation": ""}
        else:
            raise ValueError(f"Unexpected tf_list item shape: expected 6 or 7+ values, got {len(item)}")
        by_candidate[cand_idx].append((param_num, tf_vec, wave, csv_pid, run_tag, health))
    
    outdir = r"C:\Users\RSoft Things\Desktop\Results\Wavelength_results"
    os.makedirs(outdir, exist_ok=True)

    final_losses = []
    df_wave_logs = []

    for cand_idx in sorted(by_candidate.keys()):
        candidate_tf_list = by_candidate[cand_idx]
        candidate_params = params[cand_idx]
        run_tag = candidate_tf_list[0][4]

        results_path = Path(res_folder)
        wavelength_results_folder = Path(outdir)
        
        param_names = list(variable_params.keys())
        chosen_wavelength = simulation_val["free_space_wavelength"][0]
        param_tag = "_".join(
            f"{pname}_{float(candidate_params[i]):.6f}"
            for i, pname in enumerate(param_names)
        )

        ind_pattern = f"{chosen_wavelength}_LP01_{param_tag}*.ind"
        ind_files = list(results_path.glob(ind_pattern))

        if not ind_files:
            print(f"[WARN] No .ind file found for candidate {cand_idx} with pattern: {ind_pattern}")
            ind_file = None
        else:
            ind_file = ind_files[0]
            shutil.copy2(ind_file, wavelength_results_folder)

        final_loss, df_wave_log = build_df_wave_log_for_candidate(
            candidate_tf_list=candidate_tf_list,
            candidate_params=candidate_params,
            candidate_idx=cand_idx,
            iteration_num=iteration_num,
            simulation_val=simulation_val,
            res_folder=res_folder
        )

        out_csv = os.path.join(
            outdir,
            f"wavelength_loss_iter_{iteration_num}_cand_{cand_idx}_{simulation_val['core_num']}{simulation_val['grid_type']}_{simulation_val['mon_type']}.csv" #_NumModes_{len(simulation_val['mode_vals'])}
        )
        df_wave_log.to_csv(out_csv, index=False)

        # Globally fixed parameters
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
            "Example .ind File Used": str(ind_file),
            "Loss_a config.": "LP01" if not simulation_val["all_modes"] else "LP01 + higher order modes",
            "Pre-tapered": simulation_val["pre_taper"],
            "Pre-taper factor": simulation_val["pre_taper_val"] if simulation_val["pre_taper"] else None,
            "Parameter vectors": simulation_val["n_points"],
            "Acquisition type": Simulation_params["acq_type"],
            "Acquisition hyperparameter": Simulation_params["acq_hyperparam"],
            "Acquistion optimiser": Simulation_params["acq_opt"],
            "Initial points": Simulation_params["n_init_points"] 
        }

        if "taper" not in param_names:
            glob_fix_param["Taper"] = fixed_params["taper"]

        ## Legend
        leg = {
            "Loss_a": ("Intensity of MS mode in the MS core with the sum of intensities of higher order modes in ms core" if simulation_val["all_modes"] else "Intensity of MS mode in the MS core"),
            "Loss_b": "Mean intensity of non MS modes in non MS cores",
            "Loss_c": "Mean intensity of non MS modes exciting LP01 in MS core",
            "Loss_d": "Mean intensity of MS mode in non MS cores",
            "Loss": "-Loss_a - Loss_b + (Loss_c + Loss_d) + 2",
            "Extra Mode Intensity in Loss_a": "Total number of amplitudes corresponding to higher order modes included in Loss_a",
            f"Delta n({simulation_val['free_space_wavelength'][0]} um)": "Refractive index scale factor relative to the index difference between the selected refractive index and the index of silica at a reference wavelength. This should give a slightly different value for different wavelengths.",
            "Guided Modes": "Total number of modes, including rotations AND polarisations, being guided in the fibre.",
            "Simulation Health": "OK if the RSoft simulation completed; TIMEOUT if a guarded wait returned a dummy zero transfer vector.",
            "Simulation Message": "Timeout details, including missing, small, or bad-header files when applicable.",
            "Failed Simulation": "Name tags for the BPM and FemSIM files associated with the failed simulation.",
            "Pre-tapered": "Inidication if the MS core is pre-tapered to a different spec before being inserted and tapered with the rest of the cores.",
            "Pre-taper factor": "Factor by which the MS core is pre-tapered by.",
            "Parameter vectors": "Number of simultaneous parameter vectors sampled per iteration",
            "Acquisition type": "Describe the acquisition function used to select new values to sample. EI = Expected Improvement",
            "Acquisition hyperparameter": "Set's the hyperparameter for the acquisition function",
            "Acquisition optimiser": "Algorithm used to optimise the selection of parameters to sample",
            "Initial points": "Number of randomly drawn points before bayesian optimisation kicks in"
        }

        # now append the global and legend
        with open(out_csv, "a", newline="") as f:
            f.write("\n")  # blank line

            f.write("Globally Fixed Parameters\n")
            for k, v in glob_fix_param.items():
                f.write(f"{k}: {v}\n")

            f.write("\nLegend\n")
            for k, v in leg.items():
                f.write(f"{k}: {v}\n")
    # uncomment these for multi-point sampling
    #     final_losses.append(final_loss)
    #     df_wave_logs.append(df_wave_log)


    # # preserve old behavior for single-point ask()
    # if len(final_losses) == 1:
    #     return final_losses[0], df_wave_logs[0]
    
    return final_loss, df_wave_log

def main_optimizer(prior_space_pid, simulation_val, custom_priors,  taper_min, taper_max):
    simulate_tf_metric = simulation_val["simulate_tf_metric"]
    gridding = simulation_val["gridding"]
    total_calls = simulation_val["num_paras"]

    # create image folder in results location
    results_dir = Path(os.path.expanduser("~/Desktop/Results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    images_dir = results_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    results_checkpoint_path = results_dir / "optimizer_results_checkpoint.npy"
    opt_checkpoint_path = results_dir / "optimizer_state_checkpoint.pkl"

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
        mean_vals = np.array(list(bestvals.values()), dtype=float)
        mean_limits = np.array(list(bestval_limits.values()), dtype=float)
        sigmas = np.array([2.0, 1.0, 3000.0])
        scales = np.array([1.0, 1.0, 10000.0])
        _, accepted = monte_carlo_rej(mean_vals, mean_limits, scales, sigmas, simulation_val["num_paras"])
        para_space = accepted.T # shape(len(simulation_val["num_paras"]), len(bestvals.keys()))
        print(f"Sampling of {para_space.shape[0]} parameters, begin.")
    else:
        para_space = []
        for prior_name, (low, high) in param_range.items():
            if prior_name == "core_neff":
                para_space.append(Real(low, high, name=prior_name))
            else:
                para_space.append(Real(low, high, name=prior_name))

    if simulate_tf_metric and opt_checkpoint_path.exists() and not simulation_val["use_previous_results"]: #and not bestvals
        opt = load(opt_checkpoint_path)
        print(f"Loaded optimiser checkpoint from {opt_checkpoint_path}")
    elif simulate_tf_metric and results_checkpoint_path.exists() and simulation_val["use_previous_results"]:
        print(f"Using previous optimisation results from optimiser checkpoint: {results_checkpoint_path}")
        opt = Optimizer(
            dimensions=para_space, # Parameter search space (bounds + types)
            base_estimator="GP", # Surrogate model (Gaussian Process)
            acq_func=Simulation_params["acq_type"], # Acquisition function (EI or LCB, chooses next point)
            acq_func_kwargs={"xi": Simulation_params["acq_hyperparam"]}, # EI exploration strength (higher = more exploration, default=0.01)
            acq_optimizer=Simulation_params["acq_opt"], # How the acquisition function is optimised (random sampling, lbfgs)
            random_state=None, # Random seed (None = non-reproducible)
            n_initial_points=int(Simulation_params["n_init_points"]) # Number of random iterations before BO starts. Note this is NOT the number of parameters chosen before BO starts - it is the number of REPORTS via tell().
        )

        previous_optimisation_results = np.load(results_checkpoint_path)
        tried_params = np.array([r["params"] for r in previous_optimisation_results])
        tried_results = np.array([r["results"] for r in previous_optimisation_results])

        for tp, tr in zip(tried_params, tried_results):
            opt.tell(tp,tr)
    else:
        opt = Optimizer(
            dimensions=para_space, # Parameter search space (bounds + types)
            base_estimator="GP", # Surrogate model (Gaussian Process)
            acq_func=Simulation_params["acq_type"], # Acquisition function (EI or LCB, chooses next point)
            acq_func_kwargs={"xi": Simulation_params["acq_hyperparam"]}, # EI exploration strength (higher = more exploration, default=0.01)
            acq_optimizer=Simulation_params["acq_opt"], # How the acquisition function is optimised (random sampling, lbfgs)
            random_state=None, # Random seed (None = non-reproducible)
            n_initial_points=int(Simulation_params["n_init_points"]) # Number of random iterations before BO starts. Note this is NOT the number of parameters chosen before BO starts - it is the number of REPORTS via tell().
        )

    # if true, run optimisation testing the loss metric
    if simulate_tf_metric:
        wave_logs = []
        if simulation_val["use_previous_results"]:
            all_results = []
        else:
            all_results = load_checkpoint_npy(results_checkpoint_path)
        
        completed_candidates = len(all_results)
        completed_batches = completed_candidates // simulation_val["n_points"]

        if completed_candidates > 0:
            print(f"Resuming from iteration {completed_batches + 1}")
        else:
            print("No existing results checkpoint found. Starting fresh.")

        if bestvals:
            # checking if femsim files exist. If they do, continue. If not, generate them
            femSIM_file_example = "FemSim_File_DET_1.5_LP01_core_diam_8.300000_core_neff_1.449200_Taper_L_50000.000000_i1_c0_p31892_t0_ex.m00"
            if not fem_fields_present(femSIM_file_example):
                simulation_val["Fem_present"] = False
                print("No suitable FemSIM field profiles detected. Generating...")
                param_names = ["core_diam", "core_neff"]
                params = [variable_params[k] for k in param_names]
                run_tf_multproc(
                    params, 1, simulation_val, custom_priors, 
                    taper_min, taper_max, fem=True, gridding=gridding
                )

            para_space = np.asarray(para_space, dtype=float)
            if para_space.ndim == 1:
                para_space = para_space[np.newaxis, :]

            batch_size = simulation_val["n_points"]

            for batch_idx, start_idx in enumerate(range(completed_candidates, len(para_space), batch_size)):
                param_batch = para_space[start_idx:start_idx + batch_size]

                print(f"Iteration {batch_idx + 1}:")
                for cand_idx, para in enumerate(param_batch):
                    print(
                        f"  Candidate {cand_idx}: "
                        + ", ".join(
                            f"Core neff: {para[l]:.3f}" if text == "core_neff"
                            else f"{text}: {para[l]:.3f}"
                            for l, text in enumerate(variable_params.keys())
                        )
                    )

                result_batch, df_wave_logs = run_all_modes_for_params(
                    param_batch, batch_idx + 1,
                    simulation_val, custom_priors, taper_min,
                    taper_max, gridding=gridding
                )

                # if only one candidate came back, normalise to list form
                if not isinstance(result_batch, list):
                    result_batch = [result_batch]
                    df_wave_logs = [df_wave_logs]

                for cand_idx, (param_vec, df_wave_log, loss_val) in enumerate(
                    zip(param_batch, df_wave_logs, result_batch)
                ):
                    print(f"\nCandidate {cand_idx}: final_loss = {loss_val:.6f}")

                    for w, group in df_wave_log.groupby("Wavelength"):
                        row = group.iloc[0]

                        print(
                            f"  wavelength = {w:.3f} µm | "
                            f"(a, b, c, d)=("
                            f"{row['Loss_a']:.6g}, "
                            f"{row['Loss_b']:.6g}, "
                            f"{row['Loss_c']:.6g}, "
                            f"{row['Loss_d']:.6g})"
                        )

                    loss_terms = df_wave_log[["Wavelength", "Loss_a", "Loss_b", "Loss_c", "Loss_d"]].to_numpy()

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
                        "params": param_vec,
                        "result": loss_val,
                        "Iteration": batch_idx + 1,
                        "Candidate Index": cand_idx,
                        "Global Candidate Index": start_idx + cand_idx,
                        "Loss Metric": loss_terms,
                        "Number of Guided Modes": number_of_guided_modes,
                        "Core Amplitudes": amp,
                        "Core Phases": phase,
                        "Extra Core Amplitudes": ex_amp,
                        "Extra Core Phases": ex_phase
                    }

                    all_results.append(iter_result)

                atomic_save_npy(all_results, results_checkpoint_path)
                # dump(opt, opt_checkpoint_path, store_objective=False)
            return all_results
        
        # checking if femsim files exist. If they do, continue. If not, generate the,
        femSIM_file_example = "FemSim_File_DET_1.5_LP01_core_diam_8.300000_core_neff_1.449200_Taper_L_50000.000000_i1_c0_p31892_t0_ex.m00"
        if not fem_fields_present(femSIM_file_example):
            print("No suitable FemSIM field profiles detected. Generating...")
            param_names = ["core_diam", "core_neff"]
            params = [variable_params[k] for k in param_names]
            run_tf_multproc(params, 1,simulation_val, custom_priors, 
                            taper_min, taper_max, fem = True, gridding=gridding)

        # start at whatever the last iteration was (or 0), but scale the total number of iterations based
        # on the number of selected parameter vectors.
        for batch_idx in range(completed_batches, total_calls//simulation_val["n_points"]):
            # adaptable gridding to hasten simulations slightly after Bayesian optimisation kicks in
            if batch_idx < 2*int(Simulation_params["n_init_points"]):
                simulation_val["grid_size"] = 0.74
                simulation_val["grid_size_y"] = 0.74
            else:
                simulation_val["grid_size"] = 0.37
                simulation_val["grid_size_y"] = 0.37
            # ask for 1 set of parameter vectors only to prevent daemonic process having children 
            param_batch = opt.ask(n_points=simulation_val["n_points"]) 

            print(f"Iteration {batch_idx + 1}:")
            for para in param_batch:
                print("Trying " + ", ".join(
                    f"Core Refractive Index: {para[l]:.3f}" if text == "core_neff"
                    else f"{text}: {para[l]:.3f}"
                    for l, text in enumerate(variable_params.keys())
                ))         
            
            result_batch, df_wave_logs = run_all_modes_for_params(param_batch, batch_idx + 1, 
                                                                simulation_val, custom_priors, taper_min, 
                                                                taper_max, gridding = gridding)

            # tell optimiser the performance of the chosen parameters
            if simulation_val["n_points"] == 1:
                opt.tell(param_batch[0], result_batch)
            else:
                opt.tell(param_batch, result_batch)

            if simulation_val["n_points"] != 1:
                for cand_idx, (param_vec, df_wave_log, loss_val) in enumerate(zip(param_batch, df_wave_logs, result_batch)):
                    print(f"\nCandidate {cand_idx}: final_loss = {loss_val:.6f}")
                    for w, group in df_wave_log.groupby("Wavelength"):
                        row = group.iloc[0]

                        print(
                            f"  wavelength = {w:.3f} µm | "
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
                        'params': param_vec,
                        'result': loss_val,
                        'Iteration': batch_idx + 1,
                        "Candidate Index": cand_idx,
                        "Loss Metric": loss_terms,
                        "Number of Guided Modes": number_of_guided_modes,
                        "Core Amplitudes": amp,
                        "Core Phases": phase,
                        "Extra Core Amplitudes": ex_amp,
                        "Extra Core Phases": ex_phase
                    }

                    all_results.append(iter_result)
                atomic_save_npy(all_results, results_checkpoint_path)
                dump(opt, opt_checkpoint_path, store_objective=False)
            else:
                print(f"Iteration {batch_idx+1}: {result_batch:.6f}")
                for w, group in df_wave_logs.groupby("Wavelength"):
                    row = group.iloc[0]

                    print(
                        f"  wavelength = {w:.3f} µm | "
                        f"(a, b, c, d)=("
                        f"{row['Loss_a']:.6g}, "
                        f"{row['Loss_b']:.6g}, "
                        f"{row['Loss_c']:.6g}, "
                        f"{row['Loss_d']:.6g})"
                    )

                # convert df_wave_log to numpy array
                loss_terms = df_wave_logs[["Wavelength","Loss_a","Loss_b","Loss_c","Loss_d"]].to_numpy()
                
                core_amp_cols = sorted(
                    [c for c in df_wave_logs.columns if c.startswith("Core_") and c.endswith("_Amp")],
                    key=lambda s: int(s.split("_")[1])
                )

                core_phase_cols = sorted(
                    [c for c in df_wave_logs.columns if c.startswith("Core_") and c.endswith("_Phase")],
                    key=lambda s: int(s.split("_")[1])
                )
                amp = df_wave_logs[core_amp_cols].to_numpy(dtype=float)
                phase = df_wave_logs[core_phase_cols].to_numpy(dtype=float)

                extra_amp_cols = sorted(
                    [c for c in df_wave_logs.columns if c.startswith("Extra_") and c.endswith("_Amp")],
                    key=lambda s: int(s.split("_")[1])
                )

                extra_phase_cols = sorted(
                    [c for c in df_wave_logs.columns if c.startswith("Extra_") and c.endswith("_Phase")],
                    key=lambda s: int(s.split("_")[1])
                )

                ex_amp = df_wave_logs[extra_amp_cols].to_numpy(dtype=float)
                ex_phase = df_wave_logs[extra_phase_cols].to_numpy(dtype=float)

                number_of_guided_modes = df_wave_logs["Guided Modes"].to_numpy()
                iter_result = {
                    'params': param_batch[0],
                    'result': result_batch,
                    'Iteration': batch_idx + 1,
                    "Candidate Index": 0,
                    "Loss Metric": loss_terms,
                    "Number of Guided Modes": number_of_guided_modes,
                    "Core Amplitudes": amp,
                    "Core Phases": phase,
                    "Extra Core Amplitudes": ex_amp,
                    "Extra Core Phases": ex_phase
                }

                all_results.append(iter_result)
                atomic_save_npy(all_results, results_checkpoint_path)
                dump(opt, opt_checkpoint_path, store_objective=False)
        return all_results
    
    # if false, run tf code for the template parameters
    else:
        param_names = variable_params.keys() #["core_diam", "core_neff"]
        params = [variable_params[k] for k in param_names]

        # checking if femsim files exist. If they do, continue. If not, generate the,
        femSIM_file_example = "FemSim_File_DET_1.5_LP01_core_diam_8.300000_core_neff_1.449200_Taper_L_50000.000000_i1_c0_p31892_t0_ex.m00"
        # femSIM_file_example = "1.55_GIF_outer_fibre_ex.m00"
        if not fem_fields_present(femSIM_file_example):
            print("No suitable FemSIM field profiles detected. Generating...")
            param_names = ["core_diam", "core_neff"]
            params = [variable_params[k] for k in param_names]
            run_tf_multproc(params, 1,simulation_val, custom_priors, 
                            taper_min, taper_max, fem = True, gridding=gridding)
            
        if gridding:
            grid_size_range, tf_list, _ = run_tf_multproc(params, 1, simulation_val, custom_priors,
                                                          taper_min, taper_max, gridding)
            return grid_size_range, tf_list, _
        else:
            tf_list, res_folder = run_tf_multproc(params, 1,simulation_val, custom_priors, 
                                      taper_min, taper_max, gridding)
            
            # code to save results to csv
            params = np.asarray(params, dtype=float)
            if params.ndim == 1:
                params = params[np.newaxis, :]

            by_candidate = defaultdict(list)
            for item in tf_list:
                if len(item) >= 7:
                    cand_idx, param_num, tf_vec, wave, csv_pid, run_tag = item[:6]
                    health = item[6]
                elif len(item) == 6:
                    cand_idx, param_num, tf_vec, wave, csv_pid, run_tag = item
                    health = {"status": "OK", "message": "", "simulation": ""}
                else:
                    raise ValueError(f"Unexpected tf_list item shape: expected 6 or 7+ values, got {len(item)}")
                by_candidate[cand_idx].append((param_num, tf_vec, wave, csv_pid, run_tag, health))
            
            outdir = r"C:\Users\RSoft Things\Desktop\Results\Wavelength_results"
            os.makedirs(outdir, exist_ok=True)

            final_losses = []
            df_wave_logs = []

            for cand_idx in sorted(by_candidate.keys()):
                candidate_tf_list = by_candidate[cand_idx]
                candidate_params = params[cand_idx]
                run_tag = candidate_tf_list[0][4]

                results_path = Path(res_folder)
                wavelength_results_folder = Path(outdir)
                
                param_names = list(variable_params.keys())
                chosen_wavelength = simulation_val["free_space_wavelength"][0]
                param_tag = "_".join(
                    f"{pname}_{float(candidate_params[i]):.6f}"
                    for i, pname in enumerate(param_names)
                )

                ind_pattern = f"{chosen_wavelength}_LP01_{param_tag}*.ind"
                ind_files = list(results_path.glob(ind_pattern))

                if not ind_files:
                    print(f"[WARN] No .ind file found for candidate {cand_idx} with pattern: {ind_pattern}")
                    ind_file = None
                else:
                    ind_file = ind_files[0]
                    shutil.copy2(ind_file, wavelength_results_folder)

                final_loss, df_wave_log = build_df_wave_log_for_candidate(
                    candidate_tf_list=candidate_tf_list,
                    candidate_params=candidate_params,
                    candidate_idx=cand_idx,
                    iteration_num=0,
                    simulation_val=simulation_val,
                    res_folder=res_folder
                )

                out_csv = os.path.join(
                    outdir,
                    f"wavelength_loss_iter_{0}_cand_{cand_idx}_{simulation_val['core_num']}{simulation_val['grid_type']}_{simulation_val['mon_type']}.csv" #_NumModes_{len(simulation_val['mode_vals'])}
                )
                df_wave_log.to_csv(out_csv, index=False)

                # Globally fixed parameters
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
                    "Example .ind File Used": str(ind_file),
                    "Loss_a config.": "LP01" if not simulation_val["all_modes"] else "LP01 + higher order modes",
                    "Parameter vectors": simulation_val["n_points"]
                }

                if "taper" not in param_names:
                    glob_fix_param["Taper"] = fixed_params["taper"]

                ## Legend
                leg = {
                    "Loss_a": ("Intensity of MS mode in the MS core with the sum of intensities of higher order modes in ms core" if simulation_val["all_modes"] else "Intensity of MS mode in the MS core"),
                    "Loss_b": "Mean intensity of non MS modes in non MS cores",
                    "Loss_c": "Mean intensity of non MS modes exciting LP01 in MS core",
                    "Loss_d": "Mean intensity of MS mode in non MS cores",
                    "Loss": "-Loss_a - Loss_b + (Loss_c + Loss_d) + 2",
                    "Extra Mode Intensity in Loss_a": "Total number of amplitudes corresponding to higher order modes included in Loss_a",
                    f"Delta n({simulation_val['free_space_wavelength'][0]} um)": "Refractive index scale factor relative to the index difference between the selected refractive index and the index of silica at a reference wavelength. This should give a slightly different value for different wavelengths.",
                    "Guided Modes": "Total number of modes, including rotations AND polarisations, being guided in the fibre.",
                    "Simulation Health": "OK if the RSoft simulation completed; TIMEOUT if a guarded wait returned a dummy zero transfer vector.",
                    "Simulation Message": "Timeout details, including missing, small, or bad-header files when applicable.",
                    "Failed Simulation": "Name tags for the BPM and FemSIM files associated with the failed simulation.",
                    "Parameter vectors": "Number of simultaneous parameter vectors sampled per iteration"
                }

                # now append the global and legend
                with open(out_csv, "a", newline="") as f:
                    f.write("\n")  # blank line

                    f.write("Globally Fixed Parameters\n")
                    for k, v in glob_fix_param.items():
                        f.write(f"{k}: {v}\n")

                    f.write("\nLegend\n")
                    for k, v in leg.items():
                        f.write(f"{k}: {v}\n")
        return tf_list
