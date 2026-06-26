import numpy as np, pandas as pd, math
import json, os, csv, ofiber, random, time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from template import *
from collections import defaultdict
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib import colors
from matplotlib.colors import Normalize
import datetime
import glob
import ehtplot.color
import cmocean
import itertools
from HexProperties import *
import shutil, tempfile
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
            comp_name = line_to_check.split("=", 1)[1].strip()
            if segment_filter is None:
                matches_segment = True
            elif comp_name == segment_filter:
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
def create_folders(folder_name, pos):
    '''
    Creates folder to be placed within the 'Results' folder on the desktop.

    Arguments:
        - folder_name: string entry that will become the name of the folder
        - pos: string to determine where the folder is placed
    
    Returns:
        - pathway to results folder
    '''

    user_home = os.path.expanduser("~")

    if pos == "Desktop":
        desktop_path = os.path.join(user_home, "Desktop")
        results_root = os.path.join(desktop_path, "Results")
        results_folder = os.path.join(results_root, folder_name)
        os.makedirs(results_folder, exist_ok=True)
        return results_folder
    elif pos == "Onedrive":
        onedrive_path = os.path.join(user_home, r"C:\Users\RSoft Things\OneDrive - The University of Sydney (Students)\RSoft Automaton Results")
        results_root_onedrive = os.path.join(onedrive_path, "Results")
        results_folder_onedrive = os.path.join(results_root_onedrive, folder_name)
        os.makedirs(results_folder_onedrive, exist_ok=True)
    elif pos == "analysis_path":
        analysis_path = os.path.join(user_home, r"C:\Users\RSoft Things\Desktop\Results\To compress")
        results_folder_analysis_path = os.path.join(analysis_path, folder_name)
        os.makedirs(results_folder_analysis_path, exist_ok=True)
        return results_folder_analysis_path


def wait_for_files_stable(
    files,
    timeout=300,
    interval=1.0,
    stable_checks=3,
    min_size=100,
    require_rsoft_header=False
    ):
    """
    Wait until all files exist, are larger than min_size, and have stable sizes.

    files: list of file paths
    timeout: maximum wait time in seconds
    interval: time between checks
    stable_checks: number of consecutive unchanged-size checks required
    min_size: reject suspiciously tiny/empty files
    require_rsoft_header: require the first non-space byte to look like an RSoft
        user data header. This catches empty/truncated field files before BeamPROP
        opens them and raises a modal error.
    """
    files = [Path(f) for f in files]
    if not files:
        raise ValueError("No files were supplied to wait_for_files_stable.")

    start = time.time()

    last_sizes = {f: None for f in files}
    stable_counts = {f: 0 for f in files}

    while True:
        if time.time() - start > timeout:
            missing = [str(f) for f in files if not f.exists()]
            small = [
                f"{f} ({f.stat().st_size} bytes)"
                for f in files
                if f.exists() and f.stat().st_size < min_size
            ]
            bad_header = [
                str(f)
                for f in files
                if (
                    require_rsoft_header
                    and f.exists()
                    and f.stat().st_size >= min_size
                    and not rsoft_user_data_header_present(f)
                )
            ]

            raise TimeoutError(
                "Timed out waiting for FemSIM files to become stable.\n"
                f"Missing files: {missing}\n"
                f"Small files: {small}\n"
                f"Bad RSoft headers: {bad_header}"
            )

        all_stable = True

        for f in files:
            if not f.exists():
                stable_counts[f] = 0
                all_stable = False
                continue

            size = f.stat().st_size

            if size < min_size:
                stable_counts[f] = 0
                all_stable = False
                continue

            if require_rsoft_header and not rsoft_user_data_header_present(f):
                stable_counts[f] = 0
                all_stable = False
                continue

            if size == last_sizes[f]:
                stable_counts[f] += 1
            else:
                stable_counts[f] = 0

            last_sizes[f] = size

            if stable_counts[f] < stable_checks:
                all_stable = False

        if all_stable:
            return True

        time.sleep(interval)

def rsoft_user_data_header_present(path):
    try:
        with open(path, "rb") as f:
            prefix = f.read(256)
    except OSError:
        return False

    prefix = prefix.lstrip()
    return bool(prefix) and prefix.startswith(b"/")

def extract_monitor_files_from_ind(ind_path, components=("ex", "ey", "hx", "hy")):
    ind_path = Path(ind_path)
    monitor_files = []

    with open(ind_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            if line.startswith("monitor_file"):
                _, rhs = line.split("=", 1)
                fname = rhs.strip()

                p = Path(fname)
                for component in components:
                    # RSoft monitor_file entries point at the base mode name
                    # (file.m00); BeamPROP then reads file_ex.m00, file_ey.m00,
                    # etc. for the vector field data.
                    monitor_files.append(p.with_name(f"{p.stem}_{component}{p.suffix}"))

    # remove duplicates while preserving order
    return list(dict.fromkeys(monitor_files))

# def expected_femsim_mode_files(prefix_FS, mode_indices):
#     return [Path(f"{prefix_FS}.m{i:02d}") for i in mode_indices]
#######################################################################################################################################################
def get_higher_order_modes(config=None):
    config = config or Simulation_params
    if "higher_order_modes" in config:
        return list(config["higher_order_modes"])
    return list(config.get("higher_mode_indices", Simulation_params.get("higher_mode_indices", [])))

def copy_when_available(src, dst, timeout=30):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    t_start = time.time()
    while True:
        try:
            shutil.copy(src, dst)
            return
        except PermissionError:
            if time.time() - t_start > timeout:
                raise
            time.sleep(0.2)

def move_when_available(src, dst, timeout=30):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    t_start = time.time()
    while True:
        try:
            shutil.move(src, dst)
            return
        except PermissionError:
            if time.time() - t_start > timeout:
                raise
            time.sleep(0.2)
#######################################################################################################################################################
def AddHack(file_name, FS_file_name, json_file, core_num, param_dict, simulation_val, wave, core_positions, fem=False,
            core_params_bp=None, core_params_fs=None, fs_core_to_monitor=None,
            bp_add_cladding_to_cores=None, fs_add_cladding_to_cores=None,
            fs_core_num=None, fs_core_positions=None):
    '''
    Hacking function to add text that will import segments to RSoft that the Python API does not currently handle.

    file_name: name tag for the ind file to be hacked
    FS_file_name: femsim name tag
    json_file: contains a dictionary of parameters to be used by the launch field and pathway monitors
    core_num: number of cores in ind file
    param_dict: json file contain the names and values of all the parameters to be modified during simulations
    mon_type: specifies the type of monitor to use. 
        "pathway_mon" records the throughput at each iteration making the simulation time scale with Z and grid spacing,
        "port_mon" (default) records only the throughput at the end of the fibre, or the position at which the monitor is placed.
    '''
    core_to_monitor = simulation_val["core_to_monitor"]
    fs_core_to_monitor = fs_core_to_monitor or core_to_monitor
    fs_core_num = fs_core_num or core_num
    fs_core_positions = fs_core_positions or core_positions
    bp_add_cladding_to_cores = Simulation_params["add_cladding_to_cores"] if bp_add_cladding_to_cores is None else bp_add_cladding_to_cores
    fs_add_cladding_to_cores = bp_add_cladding_to_cores if fs_add_cladding_to_cores is None else fs_add_cladding_to_cores
    core_params_bp = core_params if core_params_bp is None else core_params_bp
    core_params_fs = core_params if core_params_fs is None else core_params_fs
    mon_type = simulation_val.get("mon_type", Launch_params["mon_type"])
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
        if Simulation_params["use_profile"]:
            profile_text={
            "user_profile":'''
user_profile {u}
    type = UF_DATAFILE
    filename = {file} 
end user_profile
'''
            }
            # Open FS file in append mode
            with open(f"{FS_file_name}.ind", "a") as fs_prof:
                for i in range(Simulation_params["num_profile"]):
                    prof_text = profile_text["user_profile"].format(u=i+1, file=Simulation_params["profile"][i])
                    fs_prof.write(prof_text)
            # Open BPM file in append mode
            with open(f"{file_name}.ind", "a") as bpm_prof:
                for i in range(Simulation_params["num_profile"]):
                    prof_text = profile_text["user_profile"].format(u=i+1, file=Simulation_params["profile"][i])
                    bpm_prof.write(prof_text)
            
        block_text = { 
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
        # Open FS file in append mode
        with open(f"{FS_file_name}.ind", "a") as fs:
            # Write all pathways
            for i in range(1, fs_core_num + 2):  # +1 for cladding
                text = block_text["pathway"].format(n=i)
                fs.write(text)

            # Write only one launch field (for the cladding (MMF case)/core (SMF case))
            text = block_text["launch_field"].format(
                n=1,
                cladding_idx = 1,
                launch_type=launch_array["launch_type"],
                launch_tilt=launch_array["launch_tilt"],
                launch_normalization=launch_array["launch_normalization"],
                launch_align_file = launch_array["launch_align_file"],
                launch_mode=launch_array["launch_mode"],
                launch_mode_radial=launch_array["launch_mode_radial"],
                launch_random_set=launch_array["launch_random_set"],
                launch_phase = launch_array["launch_phase"],
            )
            fs.write(text)
        # Open BP file in append mode
        with open(f"{file_name}.ind", "a") as f:

            # Write all pathways
            for i in range(1, core_num + 2):  # +1 for cladding
                text = block_text["pathway"].format(n=i)
                f.write(text)

                # Write only one launch field (for the cladding (MMF case)/core (SMF case))
            text = block_text["launch_field"].format(
                n=1,
                # cladding_idx = 1,
                launch_type=launch_array["launch_type"],
                launch_tilt=launch_array["launch_tilt"],
                launch_normalization=launch_array["launch_normalization"],
                launch_align_file = launch_array["launch_align_file"],
                launch_mode=launch_array["launch_mode"],
                launch_mode_radial=launch_array["launch_mode_radial"],
                launch_random_set=launch_array["launch_random_set"],
                launch_phase = launch_array["launch_phase"],
            )
            f.write(text)

    # Open FS file in read mode
    with open(f"{FS_file_name}.ind", "r") as fs:
        lines_fs = fs.readlines()
    # Open BP file in read mode
    with open(f"{file_name}.ind", "r") as f:
        lines = f.readlines()

    # Insert delta after core and cladding segment start
    core_name = np.array([f"core_{n}" for n in range(1, core_num+1)])
    fs_core_name = np.array([f"core_{n}" for n in range(1, fs_core_num+1)])

    for core_key in core_name:
        # add core properties to each segment
        lines = insert_after_match(lines, "begin.width =", [
            f"\tbegin.delta = {core_params_bp[core_key]['neff'] - RSoft_params['background_index']}\n",
            f"\tend.delta = {core_params_bp[core_key]['neff'] - RSoft_params['background_index']}\n"
        ], segment_filter=f"{core_key}")

        # code to add the user profile only to the special core
        if Simulation_params["use_profile"]:
            if core_key == f"core_{core_to_monitor}":
                lines = insert_after_match(lines, f"comp_name = core_{core_to_monitor}", [
                    f"\tprofile_type = PROF_USER_1\n",
                ], segment_filter=f"{core_key}")

    for core_key in fs_core_name:
        lines_fs = insert_after_match(lines_fs, "begin.width =", [
            f"\tbegin.delta = {core_params_fs[core_key]['neff'] - RSoft_params['background_index']}\n",
            f"\tend.delta = {core_params_fs[core_key]['neff'] - RSoft_params['background_index']}\n"
        ], segment_filter=f"{core_key}")

    if Simulation_params["use_profile"]:
        lines_fs = insert_after_match(lines_fs,  f"comp_name = core_{fs_core_to_monitor}", [
            f"\tprofile_type = PROF_USER_1\n",
        ], segment_filter=f"core_{fs_core_to_monitor}")

        # if core_key != f"core_{simulation_val['core_to_monitor']}":
        #     # assign material to the cores
        #     lines = insert_after_match(lines, "end.delta =", [
        #         f"\tmat_name = GeO2_2_mol%\n"
        #     ], segment_filter=f"{core_key}") 
        #     lines_fs = insert_after_match(lines_fs, "end.delta =", [
        #         f"\tmat_name = GeO2_2_mol%\n"
        #     ], segment_filter=f"{core_key}") 

    # append core refractive index delta into each ind file
    lines = insert_after_match(lines, "begin.width =", [
        f"\tbegin.delta = {launch_array['cladding_neff']- RSoft_params['background_index']}\n",
        f"\tend.delta = {launch_array['cladding_neff']- RSoft_params['background_index']}\n"
    ], segment_filter="Super Cladding") 
    lines_fs = insert_after_match(lines_fs, "begin.width =", [
        f"\tbegin.delta = {launch_array['cladding_neff']- RSoft_params['background_index']}\n",
        f"\tend.delta = {launch_array['cladding_neff']- RSoft_params['background_index']}\n"
    ], segment_filter="Super Cladding") 

    # # assign material to the cladding
    # lines = insert_after_match(lines, "end.delta =", [
    #     f"\tmat_name = SiO2\n",
    # ], segment_filter="Super Cladding") 
    # lines_fs = insert_after_match(lines_fs, "end.delta =", [
    #     f"\tmat_name = SiO2\n"
    # ], segment_filter="Super Cladding") 

    if bp_add_cladding_to_cores is not None:
        # lines = insert_after_match(lines, "begin.width =", ["profile_type = PROF_INACTIVE"
        # ], segment_filter="Super Cladding") 
        for cladd_num in bp_add_cladding_to_cores:
            lines = insert_after_match(lines, "begin.width =", [
            f"\tbegin.delta = {launch_array['cladding_neff']- RSoft_params['background_index']}\n",
            f"\tend.delta = {launch_array['cladding_neff']- RSoft_params['background_index']}\n"
            ], segment_filter=f"Core {cladd_num + 1} Cladding") 

            # assign material to thecore  claddings
            lines = insert_after_match(lines, "end.delta =", [
                f"\tmat_name = SiO2\n",
            ], segment_filter=f"Core {cladd_num + 1} Cladding")

    if fs_add_cladding_to_cores is not None:
        for cladd_num in fs_add_cladding_to_cores:
            lines_fs = insert_after_match(lines_fs, "begin.width =", [
                f"\tbegin.delta = {launch_array['cladding_neff']- RSoft_params['background_index']}\n",
                f"\tend.delta = {launch_array['cladding_neff']- RSoft_params['background_index']}\n"
            ], segment_filter=f"Core {cladd_num + 1} Cladding") 

            lines_fs = insert_after_match(lines_fs, "end.delta =", [
                f"\tmat_name = SiO2\n"
            ], segment_filter=f"Core {cladd_num + 1} Cladding") 

    # for i in range(1, core_num + 1):
    #     if i == core_to_monitor:
    #         lines = insert_after_match(lines, "begin.width =", [
    #             f"\tbegin.delta = {launch_array['cen_core_cladding_neff']}\n",
    #             f"\tend.delta = {launch_array['cen_core_cladding_neff']}\n"
    #         ], segment_filter=f"Cladding: {i}")
    #     else:
    #         lines = insert_after_match(lines, "begin.width =", [
    #             f"\tbegin.delta = {launch_array['core_cladding_neff']}\n",
    #             f"\tend.delta = {launch_array['core_cladding_neff']}\n"
    #         ], segment_filter=f"Cladding: {i}")

    # Build the updated lines
    modified_lines = []
    modified_lines_fs = []

    for curr_lines, curr_file_name, arr_name, curr_core_to_monitor, curr_core_num in zip(
        [lines, lines_fs],
        [file_name, FS_file_name],
        [modified_lines, modified_lines_fs],
        [core_to_monitor, fs_core_to_monitor],
        [core_num, fs_core_num]
    ):
        in_segment_header = False
        current_segment_is_super_cladding = False

        for line in curr_lines:
            line_strip = line.strip()
            replaced = False

            if line_strip.startswith("comp_name = Super Cladding"):
                current_segment_is_super_cladding = True
                in_segment_header = True
                arr_name.append(line)
                continue
            elif line_strip.startswith("comp_name = core_") or line_strip.startswith("comp_name = Cladding"):
                current_segment_is_super_cladding = False
                in_segment_header = True
                arr_name.append(line)
                continue

            if line_strip.startswith("extended ="):
                if not current_segment_is_super_cladding:
                    continue

            if in_segment_header and not line_strip.startswith("begin."):
                arr_name.append("\twidth_taper = TAPER_LINEAR\n")
                arr_name.append("\theight_taper = TAPER_LINEAR\n")
                arr_name.append("\tposition_taper = TAPER_LINEAR\n")
                arr_name.append("\tposition_y_taper = TAPER_LINEAR\n")
                in_segment_header = False

            for param, val in param_dict.items():
                if param == "core_diam":
                    core_diam_replaced = val
                elif param == "Taper_L":
                    taper_length_replaced = val
                if line_strip.startswith(f"{param} ="):
                    arr_name.append(f"{param} = {val:.6f}\n")
                    replaced = True
                    break

            if not replaced:
                arr_name.append(line)

        with open(f"{curr_file_name}.ind", "w") as out:
            out.writelines(arr_name)

        # hack in the port monitor stuff to monitor femSIM files rather than the launch field
        if mon_type == "port_mon":
            final_lines = []
            in_time_monitor = False
            inserted = False
            mon_number = 0
            extra_monitors = 0

            for line in arr_name:
                
                line_strip = line.strip()
                # by default add_portmonitor sets the monitor to overlap, which needs FemSIM files. Should get similar results
                # if using default field
                # if "type = TIMEMON_EXTENDED" in line:
                #     line = line.replace("type = TIMEMON_EXTENDED", "type = TIMEMON_FIELD")

                # forecfully fix certain port monitor parameters that appear as default otherwise  for port monitors
                # default text
                port_mon_text_arr = ["phi = default", 
                                    "begin.width = default", 
                                    "begin.height = default"]
                # replacement text
                port_mon_text_replace = ["phi = 0", 
                                        "begin.width = default", 
                                        "begin.height = default"]
                port_mon_text_replace_special = ["phi = 0", 
                                                f"begin.width = {core_diam_replaced}", 
                                                f"begin.height = {core_diam_replaced}"]
                            
                # for p, r, s in zip(port_mon_text_arr, port_mon_text_replace, port_mon_text_replace_special):
                #     if p in line:
                #         line = line.replace(p, r)
                if line_strip.startswith("time_monitor"):
                    in_time_monitor = True
                    inserted = False  # reset insertion flag for each time_monitor

                if line_strip.startswith(f"time_monitor {curr_core_num * 2 + 1 + extra_monitors}"):
                    in_extra_time_monitor = True
                    inserted_extra = False  # reset insertion flag for each time_monitor
                else:
                    in_extra_time_monitor = False

                if in_time_monitor:
                    # replace the special core's port monitor properties
                    if mon_number == curr_core_to_monitor:
                        for p, r in zip(port_mon_text_arr, port_mon_text_replace_special):
                            if p in line:
                                line = line.replace(p, r)
                    elif mon_number == curr_core_num * 2 + 1 + extra_monitors:
                        for p, r in zip(port_mon_text_arr, port_mon_text_replace_special):
                            if p in line:
                                line = line.replace(p, r)
                    else:
                        for p, r in zip(port_mon_text_arr, port_mon_text_replace):
                            if p in line:
                                line = line.replace(p, r)
                # elif in_extra_time_monitor:
                #     for p, r in zip(port_mon_text_arr_special, port_mon_text_replace_special):
                #         if p in line:
                #             line = line.replace(p, r)
                    # else:
                    #     for p, r in zip(port_mon_text_arr, port_mon_text_replace):
                    #         if p in line:
                    #             line = line.replace(p, r)

                # Remove 'comp_name' and 'portnum' lines
                if line_strip.startswith("portnum"): 
                    continue 

                final_lines.append(line)

                if in_time_monitor and line_strip.startswith("monitoroutputmask") and not inserted:
                    final_lines.append("\tmonitoroutputformat = OUTPUT_AMP_PHASE\n")
                    final_lines.append("\toverlap_type = 1\n")

                    # Extra modes we want for the *central* core:
                    # LP11a, LP11b, LP21a, LP21b, LP02
                    higher_order_modes = get_higher_order_modes(simulation_val)

                    if Simulation_params["fixed_fem_file"]:
                        # All monitors use the same supplied port_mon_file
                        final_lines.append(f"\tmonitor_file = {Simulation_params['port_mon_file']}\n")

                    elif Simulation_params["skip_core"] is not None:
                        final_lines.append(f"\tmonitor_file = {FS_file_name}.m00\n")
                    
                    elif fem: # for femsim file determination
                        final_lines.append(f"\tmonitor_file = {FS_file_name}.m00\n")
                    else:
                        if mon_number < curr_core_num:
                            if mon_number == curr_core_to_monitor-1:
                                final_lines.append(f"\tmonitor_file = {FS_file_name}.m00\n")
                            else:
                                # enforce other cores to have a field profile as a function of wavelength
                                monfiledir = r"C:\Users\RSoft Things\Desktop\Results\FemSIM_DET"
                                ref_prefix = f"REF_{os.getpid()}_{abs(hash(file_name)) % 1000000}"
                                base_files = find_field_base_filenames(monfiledir, wave, dest_prefix=ref_prefix) # <- this should copy the ALL .m00 femsim files with prefix ex, ey, hx, and hy to the working directory, and list the files copied.
                                # check if no files were copied
                                if len(base_files) == 0:
                                    raise FileNotFoundError(
                                        f"No FEM field files found for wave={wave} in {monfiledir}"
                                    )
                                #  note that multiple files have the same field profile, only need one.
                                monitor_file = base_files[0]
                                final_lines.append(f"\tmonitor_file = {monitor_file}\n")

                        # Extra monitors after the first core_num ports:
                        else:
                            # index of this extra monitor among the higher modes
                            idx_extra = mon_number - curr_core_num  # 0,1,2,3,4,...

                            if 0 <= idx_extra < len(higher_order_modes) and mon_number >= (curr_core_to_monitor - 1):
                                mode_idx = higher_order_modes[idx_extra]
                                if mode_idx >= 10:
                                    final_lines.append(f"\tmonitor_file = {FS_file_name}.m{mode_idx}\n")
                                    for p, r in zip(port_mon_text_arr, port_mon_text_replace_special):
                                        if p in line:
                                            line = line.replace(p, r)
                                else:
                                    final_lines.append(f"\tmonitor_file = {FS_file_name}.m0{mode_idx}\n")
                                    for p, r in zip(port_mon_text_arr, port_mon_text_replace_special):
                                        if p in line:
                                            line = line.replace(p, r)
                            else:
                                final_lines.append(f"\tmonitor_file = {FS_file_name}.m00\n")

                    final_lines.append("\tpolarizer = 2\n")
                    inserted = True
                    mon_number += 1
                # if in_time_monitor and line_strip.startswith("monitoroutputmask") and not inserted:
                #     final_lines.append("\tmonitoroutputformat = OUTPUT_AMP_PHASE\n")
                #     final_lines.append("\toverlap_type = 1\n")

                #     if Simulation_params["skip_core"] is not None and Simulation_params["fixed_fem_file"] is False:
                #         first_mode_first_pol_file = f"monitor_file = {FS_file_name}.m00" #LP01
                #         final_lines.append(f"\t{first_mode_first_pol_file}\n")

                #     elif Simulation_params["skip_core"] is not None and Simulation_params["fixed_fem_file"] is True:
                #         first_mode_first_pol_file = f"monitor_file = {Simulation_params['port_mon_file']}"
                #         final_lines.append(f"\t{first_mode_first_pol_file}\n")
                #     elif Simulation_params["fixed_fem_file"] is True:
                #         first_mode_first_pol_file = f"monitor_file = {Simulation_params['port_mon_file']}"
                #         final_lines.append(f"\t{first_mode_first_pol_file}\n")
                #     else:
                #         if mon_number == (core_to_monitor - 1):
                #             first_mode_first_pol_file = f"monitor_file = {FS_file_name}.m00" #LP01
                            
                #             final_lines.append(f"\t{first_mode_first_pol_file}\n")
                #         else:
                #             first_mode_first_pol_file = f"monitor_file = {Simulation_params['port_mon_file']}"
                #             final_lines.append(f"\t{first_mode_first_pol_file}\n")     

                #     if in_extra_time_monitor and not inserted_extra:
                #         # each value corresponds to an additional port monitor
                #         higher_mode_indices = [2, 4, 6, 8, 10]  # LP11a, LP11b, LP21a, LP21b, LP02
                #         # find where these "extra" monitors start
                #         start_index = (simulation_val["core_num"] * 2 + 1)
                #         for i, mode_idx in enumerate(higher_mode_indices):
                #             time_mon_num = start_index + i  # e.g., 16, 17, 18, 19, 20
                #             # Write corresponding monitor block only if present downstream
                #             # if line_strip.startswith(f"time_monitor {time_mon_num}"):
                #             final_lines.append(f"\tmonitor_file = {FS_file_name}.m0{mode_idx}\n")
                #             extra_monitors += 1
                #     inserted_extra = True

                        
                            
                            # for p, r in zip(port_mon_text_arr, port_mon_text_replace):
                            #     if line_strip.startswith(p):
                            #         line_strip = line_strip.replace(p,r)   
      
                    # final_lines.append(f"\tpolarizer = 2\n")
                    # inserted = True
                    # mon_number += 1
                
                if in_time_monitor and line_strip.startswith("end monitor"):
                    in_time_monitor = False
                if in_extra_time_monitor and line_strip.startswith("end monitor"):
                    in_extra_time_monitor = False
                    
            with open(f"{curr_file_name}.ind", "w") as out:
                out.writelines(final_lines)
            
            # Append/replace certain names in the femsim file
            with open(f"{FS_file_name}.ind", "r") as fin:
                lines = fin.readlines()

            output_lines = []
            for idx, line in enumerate(lines):
                stripped = line.strip()

                output_lines.append(line)
                # Insert boundary_* after boundary_gap_z = 0
                if stripped == "boundary_gap_z = 0":
                    if RSoft_params['femsim_boundary_gap_x'] >= fixed_params["core_sep"]//2:
                        raise RuntimeError(rf"Femsim boundary of {RSoft_params['femsim_boundary_gap_x']} $\mu m$ is greater than or equal to half the inner core separation of {fixed_params['core_sep']//2} $\mu m$. Mode overlap is possible.")
                    output_lines.extend([
                        f"boundary_max = {RSoft_params['femsim_boundary_gap_x']}\n",
                        f"boundary_max_y = {RSoft_params['femsim_boundary_gap_y']}\n",
                        f"boundary_min = -{RSoft_params['femsim_boundary_gap_x']}\n",
                        f"boundary_min_y = -{RSoft_params['femsim_boundary_gap_y']}\n"
                    ])
                # Insert domain_min after dimension = 3
                if stripped == "dimension = 3":
                    if "Taper_L" in variable_params:
                        output_lines.append(f"domain_min = {taper_length_replaced}\n")
                    else:
                        output_lines.append(f"domain_min = {fixed_params['Taper_L']}\n")

            # Process other replacements in a second pass
            final_lines = []
            for line in output_lines:
                stripped = line.strip()
                if stripped == "sim_tool = ST_BEAMPROP":
                    final_lines.append("sim_tool = ST_FEMSIM\n")
                elif stripped == f"grid_size = {simulation_val['grid_size']}":
                    final_lines.append("grid_size = 0.5\n")
                elif stripped == f"grid_size_y = {simulation_val['grid_size_y']}":
                    final_lines.append("grid_size_y = 0.5\n")
                elif stripped == "metric = TF":
                    final_lines.append("mode_output_format = OUTPUT_REAL_IMAG\n")
                else:
                    final_lines.append(line)

            # Write the output file
            with open(f"{FS_file_name}.ind", "w") as fout:
                fout.writelines(final_lines)

def centre_core_index(core_positions):
    if not core_positions:
        raise ValueError("Core positions must be generated before building the circuit.")
    return min(
        range(1, len(core_positions) + 1),
        key=lambda idx: core_positions[idx - 1][0] ** 2 + core_positions[idx - 1][1] ** 2
    )

def core_layout_for_special_core(special_core_idx, sim_param, simulation_val, core_name,param_dict,delta_index_at_reference_wavelength, taper):
    core_beg_dims = []
    core_end_dims = []
    circuit_core_params = {}

    if sim_param["mode_selective"] == 1:
        for j, core_key in enumerate(core_name, start=1):
            if j == special_core_idx:
                # need to modify the core_neff according to the wavelength
                # core to be optimized by skopt
                core_diam = variable_params.get("core_diam")
                core_taper = param_dict.get("taper", fixed_params.get("taper"))
                # if simulation_val["Fem_present"]:
                #     # Wavelength-dependent index:
                #     # special-core index is defined relative to the current silica index.
                #     # This applies both during optimisation and when using template/fixed parameters.
                #     if simulation_val["sellmeier"]:
                #         """THIS NEEDS TO BE WAVELENGTH DEPENDENT!!!!!!!"""
                #         core_neff = fixed_params["silica_at_any_wavelength"] + delta_index_at_reference_wavelength

                #     # Fixed-index mode:
                #     # only used when sellmeier is explicitly disabled.
                #     else:
                #         core_neff = param_dict.get("core_neff", fixed_params.get("core_neff"))

                if simulation_val["Fem_present"] and (simulation_val["simulate_tf_metric"] or simulation_val["sellmeier"]):
                    # core_neff = simulation_val["reference_silica_index"] + delta_index_at_reference_wavelength
                    core_neff = fixed_params["silica_index"] + delta_index_at_reference_wavelength
                # elif simulation_val["Fem_present"] and not simulation_val["simulate_tf_metric"]:
                #     core_neff = fixed_params["silica_index"] + delta_index_at_reference_wavelength
                elif simulation_val["Fem_present"] and (not simulation_val["simulate_tf_metric"] or not simulation_val["sellmeier"]):
                    core_neff = param_dict.get("core_neff", fixed_params.get("core_neff"))
                else:
                    # fix the geometry and index to that of other cores to determine the FemSIM files.
                    core_diam = fixed_params["other_core_diam"]
                    core_neff = simulation_val.get("core_neff", fixed_params.get("core_neff"))
                
                # code to cover the pre-tapering of the special core
                if simulation_val["Fem_present"] and simulation_val["pre_taper"]:
                    if not isinstance(simulation_val["pre_taper_val"], float):
                        raise RuntimeError(f"The value for pre_taper_val must be a float! Current value is {simulation_val['pre_taper_val']}.")
                    pre_taper_diam = core_diam / simulation_val["pre_taper_val"]
                    core_beg_dims.append((pre_taper_diam / taper, pre_taper_diam / taper))
                    core_end_dims.append((core_diam, core_diam))
                else:
                    core_beg_dims.append((core_diam / taper, core_diam / taper))
                    core_end_dims.append((core_diam, core_diam))
            else:
                # use preconfigured values to specify core parameters
                core_diam = fixed_params["other_core_diam"]
                core_neff = simulation_val.get("core_neff", fixed_params.get("core_neff"))
                core_taper = param_dict.get("taper", fixed_params.get("taper"))
            
                core_beg_dims.append((core_diam / taper, core_diam / taper))
                core_end_dims.append((core_diam, core_diam))
            
            circuit_core_params[core_key] = {
                "core_diam": core_diam,
                "neff": core_neff,
                "taper": core_taper
            }
    else:
        for core_key in core_name:
            core_diam = core_params[core_key]["core_diam"]
            core_neff = core_params[core_key]["neff"]

            core_beg_dims.append((core_diam / taper, core_diam / taper))
            core_end_dims.append((core_diam, core_diam))
            circuit_core_params[core_key] = {
                **core_params[core_key],
                "core_diam": core_diam,
                "neff": core_neff
            }

    return core_beg_dims, core_end_dims, circuit_core_params
        
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
                          penalty_batch=None, transfer_vector_batch=None, results_folder="",
                          csv_path="", name_tag=None, run_tag=None, health_batch=None):
    """
    Save a batch of scikit-optimize parameter evaluations to CSV, and plot the results.
    Moves both csv_path and best_params_log_{pid}.csv to the folder named by name_tag if provided.
    """

    include_penalty = penalty_batch is not None
    include_tf = transfer_vector_batch is not None and transfer_vector_batch[0] is not None
    include_health = health_batch is not None

    # Setup dynamic header
    header = ["Iteration"] + param_names + ["Throughput"]
    if include_penalty:
        header.append("penalty")
    if include_tf:
        tf_len = len(transfer_vector_batch[0])
        header += [f"TF_{k+1}" for k in range(tf_len)]
    if include_health:
        header += ["Simulation Health", "Simulation Message", "Failed Simulation"]

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
            if include_health:
                health = health_batch[j] or {}
                row.extend([
                    health.get("status", "OK"),
                    health.get("message", ""),
                    health.get("simulation", ""),
                ])
            writer.writerow(row)

    # Log best point
    best_idx = np.argmax(y_vals)
    best_params = x_iters[best_idx]
    best_throughput = y_vals[best_idx]
    best_tf = transfer_vector_batch[best_idx] if include_tf else []
    pid = os.getpid()
    if run_tag is None:
        para_tag = f"best_params_log_{pid}.csv"
    else:
        para_tag = f"best_params_log_{run_tag}.csv"
    best_health = health_batch[best_idx] if include_health else {}
    with open(para_tag, "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["Iteration"] + param_names + ["Throughput"] +
                        ([f"TF_{i+1}" for i in range(len(best_tf))] if include_tf else []) +
                        (["Simulation Health", "Simulation Message", "Failed Simulation"] if include_health else []))
        writer.writerow([iteration_start // batch_size + 1] + list(best_params) + [best_throughput] +
                        (list(best_tf) if include_tf else []) +
                        ([
                            best_health.get("status", "OK"),
                            best_health.get("message", ""),
                            best_health.get("simulation", ""),
                        ] if include_health else []))

    # Move both CSV files to the results folder, if name_tag is specified
    if name_tag is not None:
        # Build the results folder path (adapt to your exact convention)
        # results_folder = os.path.join(os.path.expanduser("~/Desktop/Results"), f"BP_{name_tag}")
        # os.makedirs(results_folder, exist_ok=True)
        # Move the best_params_log_{pid}.csv file
        try:
            shutil.move(para_tag, os.path.join(results_folder, para_tag))
        except Exception as e:
            print(f"Warning: Could not move {para_tag}: {e}")
        # Move the main batch CSV file
        try:
            shutil.move(csv_path, os.path.join(results_folder, os.path.basename(csv_path)))
        except Exception as e:
            print(f"Warning: Could not move {csv_path}: {e}")
    return results_folder

def build_df_wave_log_for_candidate(
    candidate_tf_list,
    candidate_params,
    candidate_idx,
    iteration_num,
    simulation_val,
    res_folder
):
    core_to_monitor = simulation_val["core_to_monitor"] - 1
    modes_to_monitor = ["LP01"]

    hyp_param_b = simulation_val.get("hyp_param_b", Simulation_params["hyp_param_b"])
    hyp_param_c = simulation_val.get("hyp_param_c", Simulation_params["hyp_param_c"])

    waves = np.asarray([item[2] for item in candidate_tf_list], dtype=float)
    unique_waves = np.unique(waves)

    wave_rows = []

    stored_data = pd.read_csv(
        r"C:\Users\RSoft Things\OneDrive - The University of Sydney (Students)\Apps\VSCode\Sellmeier_Considerations\Sellmeier_vals.csv"
    )

    k_arr = np.array(list(variable_params.keys()))
    candidate_params = np.asarray(candidate_params, dtype=float)

    def material_indices_for_wave(w):
        if simulation_val["sellmeier"]:
            indices = get_wavelength_dependent_indices(w, simulation_val, fixed_params, stored_data)

            special_core_ref_ind = None
            if "core_neff" in k_arr:
                core_neff_idx = np.where(k_arr == "core_neff")[0][0]
                ref_indices = get_wavelength_dependent_indices(1.5, simulation_val, fixed_params, stored_data)
                special_core_offset = candidate_params[core_neff_idx] - ref_indices["silica_index"]
                special_core_ref_ind = indices["silica_index"] + special_core_offset

            return {
                "special_core": special_core_ref_ind,
                "other_core": indices["non_ms_core_neff"],
                "cladding": indices["cladding_neff"],
                "capillary": indices["capillary_neff"],
                "silica": indices["silica_index"]
            }

        return {
            "special_core": None,
            "other_core": simulation_val.get("core_neff", fixed_params.get("core_neff")),
            "cladding": fixed_params["cladding_neff"],
            "capillary": RSoft_params["background_index"],
        }

    for w in unique_waves:
        tf_list_w = [
            item
            for item in candidate_tf_list
            if float(item[2]) == float(w)
        ]
        tf_list_w_metric = [item[:5] for item in tf_list_w]
        health_entries = [
            item[5]
            for item in tf_list_w
            if len(item) > 5 and item[5] and item[5].get("status", "OK") != "OK"
        ]
        health_status = "TIMEOUT" if health_entries else "OK"
        health_message = " || ".join(h.get("message", "") for h in health_entries)
        failed_simulation = " || ".join(h.get("simulation", "") for h in health_entries)
        # run_tag_w = tf_list_w[0][4] # all rows in this wavelength group should belong to the same candidate and share the same run tag
        run_tags_w = [rtag for (_, _, _, _, rtag) in tf_list_w_metric]
        run_tag_w = run_tags_w[0]
        for candidate_run_tag in run_tags_w:
            guided_mode_pattern = os.path.join(res_folder, f"{w}_Guided Modes_{candidate_run_tag}.csv")
            if glob.glob(guided_mode_pattern):
                run_tag_w = candidate_run_tag
                break
        tf_list_w_arr = [arr for (_, arr, _, _,_) in tf_list_w_metric]
        mode_labels_raw = [lab for (lab, _, _, _,_) in tf_list_w_metric]
        pid_w = [pid_raw for (_, _, _, pid_raw,_) in tf_list_w_metric]
        loss, arr_results, _, len_modes_arr, loss_a_num_extra_modes = mode_selective_tf_matrix_metric(
            tf_list_w_metric,
            res_folder,
            w,
            pid_w,
            hyp_param_b,
            hyp_param_c,
            core_to_monitor=core_to_monitor,
            modes_to_monitor=modes_to_monitor,
            simulation_val=simulation_val, 
            run_tag=run_tag_w
        )

        og_amp, og_phase, ex_amp, ex_phase, _ = extract_portmon_amp_phase(
            tf_list_w_arr,
            core_num=simulation_val["core_num"]
        )

        n_modes, n_cores = og_amp.shape
        material_indices = material_indices_for_wave(w)
        Special_core_ref_ind = material_indices["special_core"]
        Other_core_ref_ind = material_indices["other_core"]
        Cladding_ref_ind = material_indices["cladding"]
        Capillary_ref_ind = material_indices["capillary"]
        silica_index = material_indices["silica"]
            
        for m in range(n_modes):
            mode_label = (
                mode_labels_raw[m]
                if (mode_labels_raw is not None and m < len(mode_labels_raw))
                else f"Mode{m+1}"
            )

            row = {
                "Simulation Date": datetime.datetime.now(),
                "Iteration": int(iteration_num),
                "Candidate": int(candidate_idx),
                "Wavelength": float(w),
                "Loss Value": float(loss),
                "Loss_a": arr_results[0],
                "Loss_b": arr_results[1],
                "Loss_c": arr_results[2],
                "Loss_d": arr_results[3],
                "Injected Mode": str(mode_label),
                "Mode Index": int(m),
                "PID": pid_w,
                "Simulation Health": health_status,
                "Simulation Message": health_message,
                "Failed Simulation": failed_simulation,
                "Guided Modes": int(len_modes_arr[0]),
                "Extra Mode Intensity in Loss_a": (
                    int(len(loss_a_num_extra_modes[0]))
                    if simulation_val["all_modes"] else "None"
                ),
                "Non-MS Core Refractive Index": Other_core_ref_ind,
                "Cladding Refractive Index": Cladding_ref_ind,
                "Capillary Refractive Index": Capillary_ref_ind,
            }
            if Special_core_ref_ind is not None:
                row["MS Core Refractive Index"] = Special_core_ref_ind

            # write varied parameters once per row
            for p_idx, pname in enumerate(k_arr):
                row[pname] = float(candidate_params[p_idx])

            if "core_neff" in k_arr:
                core_neff_idx = np.where(k_arr == "core_neff")[0][0]
                row[f"Silica index"] = float(
                    silica_index
                )
                row[f"Delta n({simulation_val['free_space_wavelength'][0]} um)"] = float(
                    candidate_params[core_neff_idx] - silica_index
                )

            for c in range(n_cores):
                row[f"Core_{c+1}_Amp"] = og_amp[m, c]
                row[f"Core_{c+1}_Phase"] = og_phase[m, c]

            _, n_ex = ex_amp.shape
            for c in range(n_ex):
                row[f"{LP_mode_dict_rot[c+1]}_Amp"] = float(ex_amp[m, c])
                row[f"{LP_mode_dict_rot[c+1]}_Phase"] = float(ex_phase[m, c])

            wave_rows.append(row)

    df_wave_log = pd.DataFrame(wave_rows)
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
            elif param_name == "core_neff":
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
    core_segment_propterties = [path_num, core_positions, final_dims_list]
    return core_segment_propterties

def build_PL(circuit, path_num, core_positions, core_names, taper, Taper_length,
             cladd_beginning_diam, cladd_final_diam,
            #  capillary_beg_dims, capillary_end_dims,
             core_beginning_dims_list, core_final_dims_list, 
             simulation_val,cladding_positions = None, add_cladding_to_cores = None,
             core_to_monitor = None):
    if simulation_val["skip_core"] == None:
        cladding_positions = core_positions
    if add_cladding_to_cores is None:
        add_cladding_to_cores = Simulation_params["add_cladding_to_cores"]
    if core_to_monitor is None:
        core_to_monitor = Simulation_params["core_to_monitor"]
    cladding = circuit.add_segment(
        position=(0, 0, 0),
        offset=(0, 0, Taper_length),
        dimensions=cladd_beginning_diam,
        dimensions_end=cladd_final_diam
    )
    cladding.set_name("Super Cladding")
    path_num += 1 # account for the cladding
    # capillary = circuit.add_segment(position=(0, 0, 0),
    #     offset=(0, 0, Taper_length),
    #     dimensions=capillary_beg_dims,
    #     dimensions_end=capillary_end_dims
    # )
    # capillary.set_name("Capillary")

    # Store segments and monitors for attachment
    core_segments = []
    port_monitors = []

    for j, (x, y) in enumerate(core_positions):
        path_num += 1
        core = circuit.add_segment(
            position=(x / taper, y / taper, 0),
            # offset=(x, y, Taper_length),
            offset=(x - (x / taper), y - (y / taper), Taper_length),
            dimensions=core_beginning_dims_list[j],
            dimensions_end=core_final_dims_list[j]
        )
        core.set_name(core_names[j])
        # circuit.attach(core, cladding, 0, 0, 0)
        core_segments.append(core)
    if add_cladding_to_cores is not None:
        for j, (x, y) in enumerate(cladding_positions):
            for i in add_cladding_to_cores:
                if i == j:
                    if fixed_params["core_cladding_diam"] is not None:
                        core_cladding_beg_dims = (fixed_params["core_cladding_diam"] / taper,fixed_params["core_cladding_diam"] / taper)
                        core_cladding_end_dims = (fixed_params["core_cladding_diam"], fixed_params["core_cladding_diam"])
                        core_cladding = circuit.add_segment(
                            position=(x / taper, y / taper, 0),
                            offset=(x, y, Taper_length),
                            dimensions=core_cladding_beg_dims,
                            dimensions_end=core_cladding_end_dims
                        )
                        core_cladding.color('0')
                        core_cladding.set_name(f"Core {i+1} Cladding")

                    else:
                        raise Exception("core_cladding_diam cannot be None!!!!")
    
    if Launch_params["mon_type"] == "port_mon":
        for j, (x, y) in enumerate(core_positions):
            # Place monitor at the end of the segment, note that these are not offset from anything and so need the extra distance to line up with the segments
            port = circuit.add_portmonitor(dimensions = core_final_dims_list[j])
            port_monitors.append(port)

        # Attach port monitors to core segments
        for core_seg, port_mon in zip(core_segments, port_monitors):
            # Attach monitor to the *output end* of the segment
            circuit.attach(port_mon, core_seg, 1, 0, attach_angles = 0, attach_dimensions = 1) 

        # Add extra port monitors to monitor higher LP modes
        for _ in range(len(get_higher_order_modes())):
            port = circuit.add_portmonitor(dimensions = core_final_dims_list[core_to_monitor-1])
            circuit.attach(port, core_segments[core_to_monitor-1], 1, 0, attach_angles = 0, attach_dimensions = 1)
            port_monitors.append(port)
    # if Launch_params["mon_type"] == "port_mon":
    core_segment_propterties = [path_num, core_positions, core_final_dims_list]
    return core_segment_propterties

# def build_pigtail(circuit, path_num, core_positions, core_names, taper, Taper_length,
#              cladd_beginning_diam, cladd_final_diam,
#              core_beginning_dims_list, core_final_dims_list, 
#              simulation_val, core_cladding_beg_dims, core_cladding_end_dims,
#              cen_core_cladding_beg_dims, cen_core_cladding_end_dims):
    
#     ms_core = simulation_val["core_to_monitor"]
#     """
#     WIP: small MM end before the pigtail 
#     """
#     cladding_beg = circuit.add_segment(
#         position=(0, 0, 0),
#         offset=(0, 0, Taper_length),
#         dimensions=cladd_beginning_diam,
#         dimensions_end=cladd_final_diam
#     )
#     cladding_beg.set_name("Super Cladding")
#     cladding_beg.color('0')
#     path_num += 1

#     # Store segments and monitors for attachment
#     core_segments = []
#     port_monitors = []

#     for j, (x, y) in enumerate(core_positions):
#         path_num += 1

#         if j == (ms_core - 1):
#             cladding = circuit.add_segment(
#                 position=(x / taper, y / taper, 0),
#                 offset=(x, y, Taper_length),
#                 dimensions=cen_core_cladding_beg_dims,
#                 dimensions_end=cen_core_cladding_end_dims
#             )
#         else:
#             cladding = circuit.add_segment(
#                 position=(x / taper, y / taper, 0),
#                 offset=(x, y, Taper_length),
#                 dimensions=core_cladding_beg_dims,
#                 dimensions_end=core_cladding_end_dims
#             )

#         core = circuit.add_segment(
#             position=(x / taper, y / taper, 0),
#             offset=(x, y, Taper_length),
#             # offset=(x - (x / taper), y - (y / taper), Taper_length),
#             dimensions=core_beginning_dims_list[j],
#             dimensions_end=core_final_dims_list[j]
#         )
#         core.color('4')

#         core.set_name(core_names[j])
#         cladding.set_name(f"Cladding: {j + 1}")
#         # circuit.attach(core, cladding, 0, 0, 0)
#         core_segments.append(core)

#     if Launch_params["mon_type"] == "port_mon":
#         for j, (x, y) in enumerate(core_positions):
#             # Place monitor at the end of the segment, note that these are not offset from anything and so need the extra distance to line up with the segments
#             if j == (ms_core - 1):
#                 port = circuit.add_portmonitor(dimensions = cen_core_cladding_end_dims)
#             else:
#                 port = circuit.add_portmonitor(dimensions = core_final_dims_list[j])

#             port_monitors.append(port)

#     # Attach port monitors to core segments
#     if Launch_params["mon_type"] == "port_mon":
#         for core_seg, port_mon in zip(core_segments, port_monitors):
#             # Attach monitor to the *output end* of the segment
#             circuit.attach(port_mon, core_seg, 1, 0, 1) 
#     return path_num
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
    
def transfer_matrix_component(csv_path, row, port_mon = False):
    df = pd.read_csv(csv_path)
    transfer_vector = []

    monitor_columns = [col for col in df.columns if col.startswith("Monitor_")]        
    if port_mon:
        if Simulation_params["skip_core"] is not None:
            ms_col = f"Monitor_{Simulation_params['core_to_monitor']-len(Simulation_params['skip_core'])}_Amplitude"
        else:
            ms_col = f"Monitor_{Simulation_params['core_to_monitor']}_Amplitude"
    else:
        ms_col = f"Monitor_{Simulation_params['core_to_monitor'] - 1}"
    
    if Launch_params["mon_type"] == "pathway_mon":
        for cl in monitor_columns:
            transfer_vector.append(df[cl].tail(10).mean())
        throughput = df[ms_col].tail(10).mean()
        return np.array(transfer_vector), throughput
    
    elif port_mon:
        # extract individual mode selective throughput, return all port
        # monitor outputs along with the specific mode selective throughput
        throughput = df[ms_col]
        return row, throughput

            
def mode_selective_tf_matrix_metric(tf_list, folder, wave, csv_pid, hyp_param_b, hyp_param_c, 
                                    core_to_monitor, modes_to_monitor, simulation_val, run_tag):
    """
    Function that will sort through tf_list, extract the mode selective core values in ms/non-ms modes and return the loss function needed by scikit
    Arguments:
        - tf_list: transfer matrix resulting from RSoft multiprocessing
        - wave: wavelength being simulated/accessed by the worker
        - core_to_monitor: special core to have ms capabilities. Must be a single integer
        - modes_to_monitor: modes to couple into the ms core. Must be an array of values (e.g. ["LP01", "LP11a",...])
    Returns:
        - loss function that will maximise the ms core in the ms mode(s), overall power in non-ms cores in non-ms modes, 
        while minimising ms core in non-ms modes and non-ms cores in ms-mode(s)
    """
    params, results, waves, _, _ = zip(*tf_list)
    tf_list_params_results = list(zip(params, results))
    core_num = simulation_val["core_num"]
    # search for each Guided Mode csv file created and pick only the most recent one to read, since they are all the same.
    guided_mode_pattern = os.path.join(folder, f"{wave}_Guided Modes_{run_tag}.csv")
    matches = glob.glob(guided_mode_pattern)
    if not matches:
        _, results, _, _, _ = zip(*tf_list)
        # failsafe in the event that the simulation failed. Adopt terrible result to keep optimisation running
        if all(np.allclose(np.asarray(arr, dtype=float), 0.0) for arr in results):
            return (
                2.0,
                np.zeros(4, dtype=float),
                wave,
                np.array([0]),
                np.array([[]], dtype=float),
            )
        raise FileNotFoundError(f"No {wave}_Guided Modes_{run_tag}.csv files found in {folder}")
    guided_path = max(matches, key=os.path.getmtime)
    df = pd.read_csv(guided_path)
    num_modes = len(df["Mode_Index"].values) # this takes all polarisations of each mode (including LP01), for plotting only
    num_mode_arr = df["Mode_Index"].iloc[2::2].values # this takes the first polarisation of each higher order mode (excluding LP01) 
                                                        # into account as that is what the port monitors are setup to measure
    label_replacements = {
        'LP01': 'LP01',
        'LP11': 'LP11a',
        'LP-11': 'LP11b',
        'LP21': 'LP21a',
        'LP-21': 'LP21b',
        'LP02': 'LP02',
        'LP31': 'LP31a',
        'LP-31': 'LP31b',
        'LP12': 'LP12a',
        'LP-12': 'LP12b',
        'LP41': 'LP41a',
        'LP-41': 'LP41b',
        'LP22': 'LP22a',
        'LP-22': 'LP22b',
        'LP03': 'LP03',
        'LP51': 'LP51a',
        'LP-51': 'LP51b'
    }
    # tf_list = tf_list[0]
    # tf_list = np.array(tf_list) #.reshape(simulation_val["free_space_wavelength"].size,simulation_val["mode_vals"].size)

    loss_arr = []
    collected_arr = []
    len_modes_arr = []
    loss_a_num_extra_modes = []
# for tf in tf_list:
    # relabel
    new_tf_list = [
        (label_replacements.get(label, label), arr)
        for label, arr in tf_list_params_results
    ]

    mode_list = [label for label, _ in new_tf_list]
    # # code for extra modes in central core
    # extra_mode_order = ["LP11a", "LP11b", "LP21a", "LP21b", "LP02"]
    # guided_modes_set = set(mode_list) 
    # guided_mask = np.array([m in guided_modes_set for m in extra_mode_order], dtype=bool)
    # extract the amplitudes only and leave the phase information
    mode_result = {}
    extra_result = {}

    # mode_result = {
    #     f"{label}_result": new_tf_list[idx][1][0][1::2]
    #     for idx, label in enumerate(mode_list)
    # }
    for label, arr in new_tf_list:
        arr = np.array(arr).flatten()

        amps = arr[1::2]
        # split amplitude values into main LP01 values (for each core) and extra values (higher order modes on special core)
        main_amps = amps[:core_num]
        extra_amps = amps[core_num:]

        # store into dictionaries
        mode_result[f"{label}_result"] = main_amps
        extra_result[f"{label}_extra"] = extra_amps

    # Select which core and mode are mode-selective
    ms_core = core_to_monitor            # Index (0-based) for the mode-selective core
    ms_mode_index = []

    for h in modes_to_monitor:
        ms_mode_index.append(list(label_replacements.values()).index(h))
    ms_modes = ms_mode_index         # Index for the mode-selective mode(s) 

    for mode_idx in ms_modes:

        mode_label = mode_list[mode_idx]  # 'LP01', specifies the label for the MS mode 
        ms_mode_vals = mode_result[f"{mode_label}_result"]   # extracts the core amplitudes of the 7 cores for the MS mode 
        ex_ms_mode_vals = extra_result[f"{mode_label}_extra"] # extracts the higher order mode amplitudes for the MS core
        # Prepare other modes
        other_mode_labels = [lab for idx, lab in enumerate(mode_list) if idx != mode_idx] 
        other_mode_vals = [mode_result[f"{lab}_result"] for lab in other_mode_labels] 
        ex_nonms_mode_vals = [extra_result[f"{lab}_extra"] for lab in other_mode_labels]

        # 1. MS core in MS mode:
        if simulation_val["all_modes"]:
            ms_core_mode = np.abs(ms_mode_vals[ms_core])**2 + np.sum(np.abs(ex_ms_mode_vals[:len(num_mode_arr)])**2)
        else:
            ms_core_mode = np.abs(ms_mode_vals[ms_core])**2 #- np.sum(np.abs(ex_ms_mode_vals[:num_modes])**2)

        # 2. All non-MS cores in MS mode:
        nonms_core_ms_mode = [np.abs(val)**2 for idx, val in enumerate(ms_mode_vals) if idx != ms_core] 
        nonms_core_ms_mode = np.mean(nonms_core_ms_mode)

        
        # 3. Mean of non-MS modes exciting LP01 and higher order modes in MS core
        ms_core_other_mode_vals = ([np.abs(vals[ms_core])**2 for vals in other_mode_vals] + [np.abs(val)**2 for vals in ex_nonms_mode_vals[:len(num_mode_arr)] for val in vals])

        ms_core_other_mode = np.mean(ms_core_other_mode_vals) 

        # 4. Mean of non-MS cores in non-MS modes: 
        nonms_core_other_mode_vals = [
            np.abs(val)**2
            for vals in other_mode_vals
            for idx, val in enumerate(vals) if idx != ms_core
        ] # extracts every non-ms core intensity and stores it in the array called nonms_core_other_mode_vals
        nonms_core_other_mode = np.mean(nonms_core_other_mode_vals) # averages the intensity of non-ms cores in non-ms modes. 
                                                                    # This is what should be maximised and is equivelant to taking 
                                                                    # the average of each non-ms core in each individual non-ms mode

        loss_func = -ms_core_mode -hyp_param_b*nonms_core_other_mode + hyp_param_c*(nonms_core_ms_mode + ms_core_other_mode) + 2
        array_of_results = np.array([ms_core_mode, #a
                            nonms_core_other_mode, #b
                            ms_core_other_mode, #c
                            nonms_core_ms_mode #d
                            ])
        loss_arr.append(loss_func)
        # collected_arr.append(array_of_results)
        len_modes_arr.append(num_modes)
        loss_a_num_extra_modes.append(num_mode_arr)
        return loss_func, array_of_results, waves, np.array(len_modes_arr), np.array(loss_a_num_extra_modes)
        # return np.asarray(loss_func, dtype=float), np.asarray(collected_arr, dtype=float)

def read_port_mon_file(filepath = ""):
    dat = pd.read_csv(filepath, skiprows = 3, sep=r'\s+', header = None)
    all_vals = dat.values.flatten()
    filtered = all_vals[ all_vals < 1]
    return filtered
#######################################################################################################################################################
def overwrite_template_val(json_file):
    with open(json_file, "r") as launch_config:
        simulation_val = json.load(launch_config)
    core_to_monitor = simulation_val["core_to_monitor"]
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
        elif "core_diam" in fixed_params and "core_neff" in fixed_params:
            core_params[core_key] = {
                "core_diam": fixed_params["core_diam"],
                "neff": fixed_params["core_neff"]
            }
        elif "core_diam" in variable_params and "core_neff" in variable_params:
            if i != core_to_monitor:
                core_params[core_key] = {
                    "core_diam": variable_params["core_diam"],
                    "neff": variable_params["core_neff"]
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

def lp_exists(V, ell, m):
    b = ofiber.LP_mode_value(V, ell, m)
    return (b is not None) and (not np.isnan(b))

def print_paras(radii, wavelengths, mode_count, NA, upper_bound, lower_bound):
    rows = []

    wanted  = [(0, 1), (0, 2), (1, 1), (2, 1)]      # LP01, LP02, LP11, LP21
    blocked = [(1, 2), (3, 1), (0, 3), (2, 2)]      # LP12, LP31, LP03, LP22

    for r in radii:
        for wl in wavelengths:
            if not (lower_bound <= wl <= upper_bound):
                continue

            V = ofiber.V_parameter(r, NA, wl)

            ok_wanted = all(lp_exists(V, ell, m) for (ell, m) in wanted)
            ok_block  = all(not lp_exists(V, ell, m) for (ell, m) in blocked)

            if ok_wanted and ok_block:
                rows.append((r, wl, V))

    df_filtered = pd.DataFrame(rows, columns=["Core Radius (µm)", "Wavelength (µm)", "V"])

    if df_filtered.empty:
        print("No (r, λ) found that supports exactly LP01, LP02, LP11, LP21 with current NA/range.")
        return df_filtered, None, None

    min_diam = 2 * df_filtered["Core Radius (µm)"].min()
    max_diam = 2 * df_filtered["Core Radius (µm)"].max()

    taper_max = fixed_params["MCFCladd"] / min_diam
    taper_min = fixed_params["MCFCladd"] / max_diam

    # pick the smallest-radius solution to sanity-check
    idx = df_filtered["Core Radius (µm)"].idxmin()
    wave = float(df_filtered.loc[idx, "Wavelength (µm)"])
    diam = [min_diam, max_diam]

    plot_available_modes(diam, wave, ell_num=4, NA=NA)  # ell_num just controls plotting panels

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

LP_mode_rsoft_dict = {
    "LP01": (0,1),
    "LP11a": (1, 1),
    "LP11b": (-1, 1),
    "LP21a": (2, 1),
    "LP21b": (-2, 1),
    "LP02": (0,2),
    "LP31a": (3,1),
    "LP31b": (-3,1),
    "LP12a": (1, 2),
    "LP12b": (-1, 2),
    "LP41a": (4, 1),
    "LP41b": (-4, 1),
    "LP22a": (2, 2),
    "LP22b": (-2, 2),
    "LP03": (0,3),
    "LP51a": (5, 1),
    "LP51b": (-5, 1),
    "LP32a": (3, 2),
    "LP32b": (-3, 2),
    "LP13a": (1, 3),
    "LP13b": (-1, 3),
    "LP61a": (6, 1),
    "LP61b": (-6, 1),
    "LP42a": (4, 2),
    "LP42b": (-4, 2),
    "LP23a": (2, 3),
    "LP23b": (-2, 3),
    "LP04": (0, 4),
    "LP71a": (7, 1),
    "LP71b": (-7, 1)
}
##############################################################################################################################################################################################################################################################################
## Stuff used for saving results
LP_mode_dict_rot = np.array([
    "LP01",
    "LP11a",
    "LP11b",
    "LP21a",
    "LP21b",
    "LP02",
    "LP31a",
    "LP31b",
    "LP12a",
    "LP12b",
    "LP41a",
    "LP41b",
    "LP22a",
    "LP22b",
    "LP03",
    "LP51a",
    "LP51b",
    "LP32a",
    "LP32b",
    "LP13a",
    "LP13b",
    "LP61a",
    "LP61b",
    "LP42a",
    "LP42b",
    "LP23a",
    "LP23b",
    "LP04",
    "LP71a",
    "LP71b"
])

def append_kv_rows(wave_rows, kind: str, mapping: dict, base_cols: set):
    """
    Append key/value metadata rows to wave_rows, using a consistent schema.
    base_cols = set of columns used by your DATA rows (so every row is a dict with same keys).
    """
    for k, v in mapping.items():
        row = {col: np.nan for col in base_cols}
        row["RowType"] = kind
        row["MetaKey"] = str(k)
        row["MetaValue"] = str(v)
        wave_rows.append(row)

def core_pos_geo(simulation_val):
    if simulation_val["grid_type"] == "Hex":
        if simulation_val["core_num"] == 7:
            core_pos = {
                0: "Centre",
                1: "Right",
                2: "Upper Right",
                3: "Upper Left",
                4: "Left",
                5: "Lower Left",
                6: "Lower Right"
            }
        elif simulation_val["core_num"] == 19:
            core_pos = {
                0: "Centre",
                1: "Inner Ring Right",
                2: "Inner Ring Upper Right",
                3: "Inner Ring Upper Left",
                4: "Inner Ring Left",
                5: "Inner Ring Lower Left",
                6: "Inner Ring Lower Right",
                7: "Outer Ring Right",
                8: "Outer Ring Middle Upper Right",
                9: "Outer Ring Upper Right",
                10: "Outer Ring Middle Top",
                11: "Outer Ring Upper Left",
                12: "Outer Ring Middle Upper Left",
                13: "Outer Ring Left",
                14: "Outer Ring Middle Lower Left",
                15: "Outer Ring Lower Left",
                16: "Outer Ring Middle Bottom",
                17: "Outer Ring Lower Right",
                18: "Outer Ring Middle Lower Right"
            }
        else:
            raise RuntimeError(f"No core position built-in yet for {simulation_val['core_num']} cores in Hex config.")
    elif simulation_val["grid_type"] == "Pent":
        if simulation_val["core_num"] == 6:
            core_pos = {
                0: "Centre",
                1: "Upper Right",
                2: "Top",
                3: "Upper Left",
                4: "Lower Left",
                5: "Lower Right"
            }
        elif simulation_val["core_num"] > 6:
            raise RuntimeError(f"No core position built-in yet for {simulation_val['core_num']} cores in Pent config.")
        
    else:
        raise RuntimeError(f"No core position applicable for the current grid type: {simulation_val['grid_type']}.")
    
    return core_pos
##############################################################################################################################################################################################################################################################################

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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import re

LABEL_MAP = {
    "LP01": "LP01",
    "LP11": "LP11a",
    "LP-11": "LP11b",
    "LP21": "LP21a",
    "LP-21": "LP21b",
    "LP02": "LP02",
}

def canonical_label(lab: str) -> str:
    # "1.5_LP-11" -> "LP-11"
    lab = re.sub(r"^[0-9]*\.?[0-9]+_", "", str(lab))
    return LABEL_MAP.get(lab, lab)

def plot_pl_results(
    *,
    simulation_val: dict,
    iteration_num: int,
    params,
    gridding: bool,
    # gridding inputs
    tf_list=None,
    grid_size_range=None,
    # polychromatic inputs
    df_wave_log: pd.DataFrame | None = None,
    modes_to_monitor=("LP01"),
    mode_reorder=[0, 5, 1, 2, 3, 4],
    reorder=True,
    image_dir=r"C:\Users\RSoft Things\Desktop\Results\Images",
    combined_dir=r"C:\Users\RSoft Things\Desktop\RSoft-Automaton",
    # plotting helpers you already have
    extract_portmon_amp_phase=None,
    plot_tf_matrix=None,
    plot_combined_tf_matrix=None,
    # print_max_amp_or_phase_value=None,
    use_seaborn_grid_plot=False,
    sns=None,
):
    """
    Plot either:
      - grid survey results (gridding=True), OR
      - wavelength-resolved results from df_wave_log (gridding=False)
    """

    core_number = simulation_val["core_num"]
    modes_to_monitor = list(modes_to_monitor)

    os.makedirs(image_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1) GRID SURVEY
    # ------------------------------------------------------------
    if gridding:
        if tf_list is None or grid_size_range is None:
            raise ValueError("For gridding=True you must provide tf_list and grid_size_range.")
        if extract_portmon_amp_phase is None:
            raise ValueError("extract_portmon_amp_phase must be passed in.")

        tf_vals = []
        for i in range(len(grid_size_range)):
            tf_vals.append(tf_list[0][i][1])  # your existing convention
        tf_vals = np.array(tf_vals)

        amp, phase, grid_size, _ = extract_portmon_amp_phase(tf_vals, core_number, grid_size_range)
        phase = np.unwrap(phase)

        amp_data, phase_data = [], []
        for gsize, a_vals, p_vals in zip(grid_size, amp, phase):
            for core_idx, val in enumerate(a_vals):
                amp_data.append({"Grid": gsize, "Core": core_idx + 1, "Amplitude": val})
            for core_idx, val in enumerate(p_vals):
                phase_data.append({"Grid": gsize, "Core": core_idx + 1, "Phase": val})

        amp_df = pd.DataFrame(amp_data)
        phase_df = pd.DataFrame(phase_data)
        phase_df["Phase_norm"] = phase_df.groupby("Core")["Phase"].transform(lambda x: x - x.iloc[0])

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        if use_seaborn_grid_plot:
            if sns is None:
                raise ValueError("use_seaborn_grid_plot=True but sns is None.")
            sns.lineplot(data=amp_df, x="Grid", y="Amplitude", hue="Core", marker="o", ax=axes[0])
            sns.lineplot(data=phase_df, x="Grid", y="Phase_norm", hue="Core", marker="o", ax=axes[1])
        else:
            # fallback: matplotlib only
            for core_id in sorted(amp_df["Core"].unique()):
                sub = amp_df[amp_df["Core"] == core_id]
                axes[0].plot(sub["Grid"], sub["Amplitude"], marker="o", label=f"Core {core_id}")
            for core_id in sorted(phase_df["Core"].unique()):
                sub = phase_df[phase_df["Core"] == core_id]
                axes[1].plot(sub["Grid"], sub["Phase_norm"], marker="o", label=f"Core {core_id}")

        axes[0].set_title("Effect of grid size on BeamPROP Results")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(loc="upper right", title="Core")

        axes[1].set_ylabel("Normalised Phase")
        axes[1].set_xlabel(r"Grid Size ($\mu m$)")
        axes[1].legend(loc="lower right", title="Core")

        plt.tight_layout()
        out = f"{core_number}c{simulation_val['grid_type']}PL_Grid_Survey.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        return {"grid_plot": out}

    # ------------------------------------------------------------
    # 2) WAVELENGTH-RESOLVED (mono or poly)
    # ------------------------------------------------------------
    if df_wave_log is None or len(df_wave_log) == 0:
        raise ValueError("For gridding=False you must provide a non-empty df_wave_log.")
    if plot_tf_matrix is None or plot_combined_tf_matrix is None:
        raise ValueError("plot_tf_matrix, plot_combined_tf_matrix must be passed in.")

    written = []
    tf_labels = ["Amplitude", "Phase"]

    core_amp_cols = sorted(
        [c for c in df_wave_log.columns if c.startswith("Core_") and c.endswith("_Amp")],
        key=lambda s: int(s.split("_")[1])
    )
    core_phase_cols = sorted(
        [c for c in df_wave_log.columns if c.startswith("Core_") and c.endswith("_Phase")],
        key=lambda s: int(s.split("_")[1])
    )

    extra_amp_cols = sorted(
        [c for c in df_wave_log.columns if c.startswith("Extra_") and c.endswith("_Amp")],
        key=lambda s: int(s.split("_")[1])
    )
    extra_phase_cols = sorted(
        [c for c in df_wave_log.columns if c.startswith("Extra_") and c.endswith("_Phase")],
        key=lambda s: int(s.split("_")[1])
    )

    # Group rows by wavelength
    for w, group in df_wave_log.sort_values("Wavelength").groupby("Wavelength"):

        # reconstruct matrices
        amp = group[core_amp_cols].to_numpy(dtype=float)
        phase = group[core_phase_cols].to_numpy(dtype=float)

        if extra_amp_cols:
            ex_amp = group[extra_amp_cols].to_numpy(dtype=float)
            ex_phase = group[extra_phase_cols].to_numpy(dtype=float)
        else:
            ex_amp = np.empty((amp.shape[0], 0))
            ex_phase = np.empty((amp.shape[0], 0))

        # injected modes in this wavelength
        labels = group["Injected Mode"].tolist()

        # reorder rows to match desired canonical order
        mode_order = ["_LP01", "_LP11a", "_LP11b", "_LP21a", "_LP21b", "_LP02"]

        idx = []
        for m in mode_order:
            for i, l in enumerate(labels):
                if l.endswith(m):
                    idx.append(i)

        amp = amp[idx, :]
        phase = phase[idx, :]
        if ex_amp.size:
            ex_amp = ex_amp[idx, :]
        if ex_phase.size:
            ex_phase = ex_phase[idx, :]

        # max_value = print_max_amp_or_phase_value(amp)

        tf_figure = plt.figure(figsize=(20, 12))
        param_str = ", ".join(f"{p:.3f}" for p in params)
        tf_figure.suptitle(
            f"Iteration {iteration_num} @ λ={w:.4g} μm\nParameters: [{param_str}]",
            y=0.7, x=0.24
        )

        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        ax0 = tf_figure.add_subplot(gs[0])
        ax1 = tf_figure.add_subplot(gs[1])
        ax2 = tf_figure.add_subplot(gs[2], sharey=ax1)

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
        axes = [ax1, ax2]
        tf_to_plot = [amp, phase]
        tf_to_plot_ex = [ex_amp, ex_phase]

        for i, (lab, tf_type, ex_type) in enumerate(zip(tf_labels, tf_to_plot, tf_to_plot_ex)):
            plot_phase = (lab == "Phase")

            tf_vectors = list(zip(group["Injected Mode"], tf_type))
            extra_tf_vectors = list(zip(group["Injected Mode"], tf_to_plot_ex))
            plot_tf_matrix(tf_vectors, simulation_val, extra_tf_vectors, matrix_type=f"{lab}", ax=axes[i],
                cbar=True, reorder=reorder, phase=plot_phase)
            # plot_tf_matrix(
            #     tf_type, simulation_val, ex_type,
            #     matrix_type=f"{lab}", ax=axes[i],
            #     cbar=True, reorder=reorder, phase=plot_phase
            # )
            ax2.set_ylabel(None)

        # combined matrix plot
        combined_name = (
            f"TF_Combined_{simulation_val['core_num']}c{simulation_val['grid_type']}PL_"
            f"iter{iteration_num:04d}_lam{w:.4g}.png"
        )

        plot_combined_tf_matrix(
            simulation_val,
            amp, phase,
            ex_amp, ex_phase,
            core_number,
            labels=group["Injected Mode"].tolist(),
            dir=combined_dir,
            name=combined_name
        )

        monitored_mode = modes_to_monitor[0]
        filename = f"transfer_matrix_{monitored_mode}_iter_{iteration_num:04d}_lam_{w:.4g}.png"
        save_path = os.path.join(image_dir, filename)
        tf_figure.savefig(save_path, dpi=300)
        plt.close(tf_figure)

        written.append(save_path)

    return {"wavelength_plots": written}

# def extract_portmon_amp_phase(tf_list, core_num, grid_size_range=None):

#     og_amp_list = []
#     og_phase_list = []
#     ex_amp_list = []
#     ex_phase_list = []
#     tf_result = []

#     for tf in tf_list:
#         arrs = np.array(tf).flatten()
#         tf_result.append(arrs)

#         amps = arrs[1::2]
#         phases = np.deg2rad(arrs[2::2])

#         # Correct separation:
#         og_amp = amps[:core_num]
#         og_phase = phases[:core_num]
#         og_amp_list.append(og_amp)
#         og_phase_list.append(og_phase)

#         if grid_size_range is None:
#             ex_amp = amps[core_num:]
#             ex_phase = phases[core_num:]
#             ex_amp_list.append(ex_amp)
#             ex_phase_list.append(ex_phase)

#     # Convert to 2D arrays (modes × cores)
#     og_amp_arr = np.array(og_amp_list)
#     og_phase_arr = np.array(og_phase_list)
#     if grid_size_range is None:
#         ex_amp_arr = np.array(ex_amp_list)
#         ex_phase_arr = np.array(ex_phase_list)

#     # Optional grid-size return
#     if grid_size_range is not None:
#         return og_amp_arr, og_phase_arr, list(grid_size_range), tf_result

#     return og_amp_arr, og_phase_arr, ex_amp_arr, ex_phase_arr, tf_result

def extract_portmon_amp_phase(tf_list, core_num, grid_size_range = None):
    tf_result = [np.asarray(tf, dtype=float).ravel() for tf in tf_list]

    tf_mat = np.vstack(tf_result)

    amps = tf_mat[:, 1::2]
    phases = np.deg2rad(tf_mat[:, 2::2])

    og_amp_arr = amps[:, :core_num]
    og_phase_arr = phases[:, :core_num]

    if grid_size_range is None:
        ex_amp_arr = amps[:, core_num:]
        ex_phase_arr = phases[:, core_num:]
        return og_amp_arr, og_phase_arr, ex_amp_arr, ex_phase_arr, tf_mat
    
    return og_amp_arr, og_phase_arr, np.asarray(list(grid_size_range)), tf_mat

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
    core_num = simulation_val["core_num"]
    geo = simulation_val["grid_type"]
    plot_centre_core = simulation_val.get("plot_centre_core", Simulation_params["plot_centre_core"])

    for label, vec in tf_vector:
        labels.append(label)
        # if geo == "Hex":
        #     if core_num == 19:
        #         reorder_indices = [9, 10, 14, 13, 8, 4, 5, 11, 15, 18, 17, 16, 12, 7, 3, 0, 1, 2, 6]
        #     elif core_num == 7:
        #         if simulation_val["skip_core"] is not None:
        #             if plot_centre_core:
        #                 reorder_indices = [5, 0, 1, 2, 3, 4]
        #         else:
        #             if plot_centre_core:
        #                 reorder_indices = [6, 0, 1, 2, 3, 4, 5]
        #             else:
        #                 reorder_indices = [0, 1, 2, 3, 4, 5]
        # elif geo == "Pent":
        #     if core_num == 6:
        #         # not much of a change since these positions are 
        #         # calculated in an anti-clockwise fashion to begin with
        #         reorder_indices = [5, 1, 2, 3, 4, 0] 
        # print(reorder_indices)
        vec = np.array(vec)#[reorder_indices]

        tf_matrix.append(vec)
    return labels, tf_matrix


def phase_to_pixel(phase_val, phase_min, phase_max, resolution):
    # Maps phase_val to a pixel index for a colorbar image
    return int(round((phase_val - phase_min) / (phase_max - phase_min) * (resolution - 1)))

def assign_17modes_to_tflist(tf_list, simulation_val):

    core_num = simulation_val.get("core_num", Simulation_params["core_num"])
    geo = simulation_val.get("grid_type", Simulation_params["grid_type"])
    if core_num == 19:
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
        ("LP51b", tf_list[16]),
        ("LP32a", tf_list[17]),
        ("LP32b", tf_list[18])
        ]
    elif core_num == 7 or geo == "Pent":
        tf_vectors_phase = [
        ("LP01", tf_list[0]),
        ("LP02", tf_list[5]),
        ("LP11a", tf_list[1]),
        ("LP11b", tf_list[2]),
        ("LP21a", tf_list[3]),
        ("LP21b", tf_list[4])
        ]
    return tf_vectors_phase

def plot_tf_matrix(tf_vectors, simulation_val, extra_modes,
                   matrix_type="", ax=None, cbar=True,
                   reorder=False, phase=False):

    core_num = simulation_val["core_num"]
    core_to_monitor = simulation_val["core_to_monitor"]

    # ------------------------------
    # Unpack main TF
    # ------------------------------
    labels, vectors = zip(*tf_vectors)
    tf_matrix = np.array(vectors, dtype=float).T  # (cores × modes)

    # ------------------------------
    # Unpack extra TF
    # ------------------------------
    if extra_modes:
        _, extra_vectors = zip(*extra_modes)
        extra_matrix = np.array(extra_vectors, dtype=float).T
    else:
        extra_matrix = np.empty((0, tf_matrix.shape[1]))

    # ------------------------------
    # Safe vmin/vmax
    # ------------------------------
    vmin = tf_matrix.min()
    vmax = tf_matrix.max()

    if extra_matrix.size:
        vmin = min(vmin, extra_matrix.min())
        vmax = max(vmax, extra_matrix.max())

    cmap = "twilight_shifted" if phase else "viridis"

    # ------------------------------
    # Ax handling
    # ------------------------------
    if isinstance(ax, plt.Axes):
        fig = ax.get_figure()
        ax.set_visible(False)
        gs = ax.get_subplotspec().subgridspec(
            2, 1, height_ratios=[1, 0.5], hspace=0.05
        )
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    else:
        ax1, ax2 = ax
        fig = ax1.get_figure()

    # ------------------------------
    # Main matrix
    # ------------------------------
    im_main = ax1.imshow(tf_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_xticks([])
    ax1.set_yticks(range(core_num))
    ax1.set_yticklabels([f"{i+1}" for i in range(core_num)], fontsize=10)
    ax1.set_ylabel("Output Core", fontsize=12)

    # ------------------------------
    # Extra matrix
    # ------------------------------
    if extra_matrix.size:
        im_extra = ax2.imshow(extra_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels([l.split("_")[-1] for l in labels], rotation=90, fontsize=10)

        lp_names = ["LP11a", "LP11b", "LP21a", "LP21b", "LP02"]
        ax2.set_yticks(range(extra_matrix.shape[0]))
        ax2.set_yticklabels(
            [fr"${core_to_monitor}_{{{m}}}$" for m in lp_names[:extra_matrix.shape[0]]],
            fontsize=10
        )
    else:
        im_extra = None
        ax2.text(0.5, 0.5, "No extra modes", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])

    ax2.set_xlabel("Input Mode", fontsize=12)

    if cbar:
        fig.colorbar(im_main, ax=[ax1, ax2]).set_label(matrix_type, fontsize=14)

    return im_main, im_extra

def plot_combined_tf_matrix(
    simulation_val,
    amp,
    phase,
    ex_amp,
    ex_phase,
    core_num,
    labels,
    phase_max=2*np.pi,
    amp_max=1.0,
    dir="",
    name=""
):
    """
    Combine amplitude and phase into complex transfer matrix image.
    Assumes amp, phase already ordered correctly by mode.
    """

    resolution = 500
    phase_min = 0
    amp_min = 0

    # amp and phase are already (n_modes, n_cores)
    comp_matrix = amp * np.exp(1j * phase)
    ex_comp_matrix = ex_amp * np.exp(1j * ex_phase) if ex_amp.size else np.empty((0, amp.shape[1]))

    # convert to image format (core × mode × RGB)
    amp_phase_img = np.transpose(
        apply_complex_map(comp_matrix, cmocean.cm.phase),
        (1, 0, 2)
    )

    if ex_comp_matrix.size:
        ex_amp_phase_img = np.transpose(
            apply_complex_map(ex_comp_matrix, cmocean.cm.phase),
            (1, 0, 2)
        )
    else:
        ex_amp_phase_img = np.zeros((1, amp_phase_img.shape[1], 3))

    amp_phase_colorbar = generate_complex_colorbar(resolution=resolution)

    fig, ax = plt.subplots(2, 1, figsize=(14, 8))
    ax1, ax2 = ax

    # ----------------------------
    # Main transfer matrix
    # ----------------------------
    ax1.imshow(amp_phase_img, aspect='auto', origin='lower')
    ax1.set_xticks([])
    ax1.set_yticks(range(core_num))
    ax1.set_yticklabels([f"{i+1}" for i in range(core_num)], fontsize=10)
    ax1.set_ylabel("Output Core", fontsize=12)
    ax1.set_title("Complex Transfer Matrix (Amplitude+Phase)", fontsize=18)

    # ----------------------------
    # Extra matrix
    # ----------------------------
    ax2.imshow(ex_amp_phase_img, aspect='auto', origin='lower')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels([l.split("_")[-1] for l in labels], rotation=90, fontsize=10)

    ax2.set_yticks(range(ex_amp_phase_img.shape[0]))
    ax2.set_ylabel("Extra Modes")
    ax2.set_xlabel("Input Mode", fontsize=12)

    # ----------------------------
    # Custom complex colourbar
    # ----------------------------
    cb_ax = fig.add_axes([1, 0.15, 0.03, 0.8])
    cb_ax.imshow(amp_phase_colorbar, aspect='auto', origin='lower')

    phase_tick_vals = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    phase_tick_labels = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]

    amp_tick_vals = [0, amp_max]
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

    save_path = os.path.join(dir, name)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

# def print_max_amp_or_phase_value(array):
#         """
#         Function that loops through each element in the array and prints out the maximum value

#         Arguments:
#             - array: 1D array of values
#         Returns:
#             - maximum value in the array
#         """
#         max_val = []
#         for i in array:
#             max_val.append(max(i))
#         max_value = max(max_val)
#         return max_value
#######################################################################################################################################################
def assign_core_properties(simulation_val):
    '''
    Function that takes in the dictionary simulation_val and assigns template parameters such as 
    core diameter and core neff prior to dumping json. If mode selective (=1) then the monitored core
    will be set to None so that skopt may optimise its parameters alone.
    '''
    if "core_num" in simulation_val: #and "core_neff" in simulation_val:
        ms_diam = [variable_params["core_diam"]] * simulation_val["core_num"]
        if "core_neff" in variable_params:
            ms_delta = [variable_params["core_neff"]] * simulation_val["core_num"]
        else:
            ms_delta = [fixed_params["core_neff"]] * simulation_val["core_num"]

        if len(ms_diam) != simulation_val["core_num"] or len(ms_delta) != simulation_val["core_num"]:
            raise Exception("Number of specified core properties does not match the number of modelled cores")

        for i, (diam, neff) in enumerate(zip(ms_diam, ms_delta), start=1):
            if simulation_val["mode_selective"] == 1 and i == simulation_val["core_to_monitor"]:
                simulation_val[f"core_{i}"] = {
                    "core_diam": None,
                    "core_neff": None
                }
            else:
                simulation_val[f"core_{i}"] = {
                    "core_diam": diam,
                    "core_neff": neff
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

    Nx, Ny = Z.shape#dat.shape
    # Ny = Nx
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
    writer = animation.FFMpegWriter(fps = 30, bitrate=8000, metadata=dict(artist = "Justin Vella"))
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
########################################################################################################################################################################################################################################################################################

"""
depreciated
"""
def coarse_sampler(best_vals, density = 4):
    half_widths = []
    best = np.array([best_vals[x] for x in best_vals])

    for y in best_vals:
        if y == "core_diam":
            half_widths.append(best_vals[y] * 0.1)
        if y == "core_neff":
            half_widths.append(best_vals[y] * 0.001)
        if y == "Taper_L":
            half_widths.append(best_vals[y] * 0.01)

    # search window
    half_widths = np.array(half_widths)

    ns = np.array([density] * len(best)) # resolution; number of samples per parameter

    grids = [np.linspace(best[i]-half_widths[i], best[i]+half_widths[i], ns[i]) for i in range(len(best))]

    # All parameter combinations
    param_grid = np.array(list(itertools.product(*grids)))  # shape (N, len(best)), e.g. N = n1*n2*n3
    return param_grid

def monte_carlo_rej(mean_vals, mean_limits, scales, sigmas, n_accept):
    """
    Function that samples values in gaussian distributions centred on some mean value to simulate.  
    """
    generated = []
    accepted = []

    if scales is None:
        scales = np.ones_like(mean_vals, dtype=float)
    else:
        scales = scales
        if scales.shape != mean_vals.shape: # ensure that the number of scales matches the number of variables
            raise ValueError(f"scales must have shape {mean_vals.shape}, got {scales.shape}")
        if np.any(scales <= 0): # ensure that the scales are positive definite
            raise ValueError("all scales must be > 0")

    rng = np.random.default_rng()
    for m, mu in enumerate(mean_vals):
        counter = 0
        
        s = scales[m]
        mu_s = mu / s
        sigma_s = sigmas[m] / s

        while counter < n_accept:
            x_s = rng.normal(mu_s, sigma_s)
            x = x_s * s
            generated.append(x)

            x_low, x_upp = mean_limits[m]
            if (x_low <= x <= x_upp):
                accepted.append(x)
                counter += 1
            else:
                continue

    return np.array(generated), np.array(accepted).reshape(len(mean_vals),n_accept)

def atomic_save_npy(obj, filepath):
    """
    Core to prevent loss of data in the event of computer crashes.
    Write the new checkpoint to a temporary file first. 
    Only once that write succeeds, swap it into place as the real checkpoint. 
    Then clean up any leftover temp file.

    Arguments: 
        - obj: Python object to save
        - filepath: location of the final .npy file
    """
    # convert directory into Path object
    filepath = Path(filepath)
    # ensure the parent directory exists. If not, create it.
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # create temporary file, fd == low-level file descriptor, tmp_name == string containing the full path to the temporary file
    fd, tmp_name = tempfile.mkstemp(dir=filepath.parent, suffix=".npy")
    # close fd since we aren't writing to it
    os.close(fd)
    tmp_path = Path(tmp_name)

    # try to save and replace the file
    try:
        # at this point the temporary file contains the new checkpoint data and hasn't affected the final file.
        np.save(tmp_path, np.array(obj, dtype=object), allow_pickle=True)
        # replace the final file with the temp. file
        os.replace(tmp_path, filepath)

    # regardless if the try block suceeds or fails, delete temp file 
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

def load_checkpoint_npy(filepath):
    """
    Load optimiser checkpoint if it exists

    Arguments:
        - filepath: location of the final .npy file
    Returns:
        - empty array or list of previous optimiser results
    """

    filepath = Path(filepath)
    if not filepath.exists():
        return []
    return np.load(filepath, allow_pickle=True).tolist()

def save_optimizer_progress_plot(results_log, images_dir):
    losses = []
    for record in results_log:
        if not isinstance(record, dict) or "result" not in record:
            continue

        try:
            loss_value = np.asarray(record["result"], dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue

        if loss_value.size != 1 or not np.isfinite(loss_value[0]):
            continue

        losses.append(float(loss_value[0]))

    if not losses:
        return

    losses = np.asarray(losses, dtype=float)
    x_vals = np.arange(1, len(losses) + 1)
    n_initial_points = int(Simulation_params.get("n_init_points", 0))

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, np.minimum.accumulate(losses))
    if n_initial_points > 0:
        plt.axvline(
            n_initial_points,
            ls="--",
            color="r",
            label=f"First {n_initial_points} evaluations"
        )
        plt.legend(loc="best")
    plt.ylabel("Loss Function Value")
    plt.xlabel("Parameter Iteration")
    plt.title("Variation of best result")
    plt.tight_layout()
    plt.savefig(images_dir / "optimiser_progress.png", dpi=300)
    plt.close()

def read_neff_values(filepath):
    with open(filepath, "r") as f:
        # convert to floats, remove empty lines
        return [float(line.strip()) for line in f if line.strip()]

def sample_range(custom_priors, bestvals, shrink=15.0):
    """
    Builds a 1D dictionary for new prior ranges, centred on the best values provided in bestvals.

    Arguments:
        custom_priors: the original dictionary of prior ranges with which skopt chooses from originally
        bestvals: dictionary containing the best values from KDE estimations. Must contain the same string entries as custom_priors
        shrink: affects how wide the new sampling region is. Defaults to 15.
    Returns:
        sample_spaces: dictionary containing focused prior spaces of each parameter in the same configuration as custom_priors.
    """
    sample_spaces = {}
    for name, (pmin, pmax) in custom_priors.items():
        if name not in bestvals:
            raise KeyError(f"Best value for '{name}' is not present in bestvals.")
        prior_range = np.array([pmin, pmax])
        prior_std = np.std(prior_range)

        centre_val = bestvals[name]
        half_width = prior_std / shrink
        low = max(pmin, centre_val - half_width)
        high = min(pmax, centre_val + half_width)

        sample_spaces[name] = (low, high)
    return sample_spaces

def plot_param_progression(tf_list, simulation_val, param_label_vector):
    '''
    save results for plotting/analysis
    '''
    # Unpack results
    param_label_1, param_label_2, param_label_3 = param_label_vector 
    x_iters = [r["params"] for r in tf_list]  # parameter sets
    y_vals = [r["result"] for r in tf_list]  # throughput values
    metric = [r["Loss Metric"] for r in tf_list] # array containing the loss metrics

    simulated_wavelength = np.array([arr[:,0] for arr in metric])
    unique_waves = np.unique(simulated_wavelength)
    a = np.array([arr[:,1] for arr in metric])
    b = np.array([arr[:,2] for arr in metric])
    c = np.array([arr[:,3] for arr in metric])
    d = np.array([arr[:,4] for arr in metric])

    batch_numbers = [r['Iteration'] for r in tf_list]

    # core_neff = [params[0] for params in x_iters]
    core_diam = [params[0] for params in x_iters]
    core_neff = [params[1] for params in x_iters]
    taper_length = [params[2] for params in x_iters]
    throughput = [y for y in y_vals]  
    hyp_param_b = simulation_val.get("hyp_param_b", Simulation_params["hyp_param_b"])
    hyp_param_c = simulation_val.get("hyp_param_c", Simulation_params["hyp_param_c"])

    fig_metric, axes_metric = plt.subplots(2,2, figsize = (8,8), sharex = True)
    for j, w in enumerate(unique_waves):
        ax = axes_metric[0,0] 
        ax.scatter(batch_numbers, a[:,j], label=f"{w} µm")
        ax.set_ylabel("a")

        ax = axes_metric[0,1]
        ax.scatter(batch_numbers, b[:,j], label=f"{w} µm")
        ax.set_ylabel("b")

        ax = axes_metric[1,0]
        ax.scatter(batch_numbers, c[:,j],label=f"{w} µm")
        ax.set_ylabel("c")
        ax.set_xlabel("Iteration number")

        ax = axes_metric[1,1]
        ax.scatter(batch_numbers, d[:,j],label=f"{w} µm")
        ax.set_ylabel("d")
        ax.set_xlabel("Iteration number")
    plt.suptitle(f"Metric results for {simulation_val['num_paras']} parameters \n and hyper parameters of {hyp_param_b} and {hyp_param_c}")
    fig_metric.legend()
    fig_metric.tight_layout()
    fig_metric.show()
    fig_metric.savefig(f"Bayesian Opt Results\{simulation_val['num_paras']}_params_gridsize_{simulation_val['grid_size']}bayesianmetric_hyperparam{hyp_param_b}_{hyp_param_c}.png")

    # make 3d scatter plot and individual parameter plots
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2])  # 3 rows, 2 columns
    plt.suptitle(f"Bayesian Optimiser Results for {simulation_val['num_paras']} parameter sets \n with hyperparams {hyp_param_b} and {hyp_param_c}.png")
    # Core neff vs Core diameter
    ax1 = fig.add_subplot(gs[0, 0])
    comp1 = ax1.scatter(core_diam, core_neff, c=throughput, cmap='viridis_r', s=80, edgecolor='k')
    ax1.set_xlabel(param_label_1)
    ax1.set_ylabel(param_label_2)

    # Taper ratio vs Core diameter
    ax2 = fig.add_subplot(gs[1, 0])
    comp2 = ax2.scatter(core_diam, taper_length, c=throughput, cmap='viridis_r', s=80, edgecolor='k')
    ax2.set_xlabel(param_label_1)
    ax2.set_ylabel(param_label_3)

    # Taper ratio vs Core neff
    ax3 = fig.add_subplot(gs[2, 0])
    comp3 = ax3.scatter(core_neff, taper_length, c=throughput, cmap='viridis_r', s=80, edgecolor='k')
    ax3.set_xlabel(param_label_2)
    ax3.set_ylabel(param_label_3)

    ax4 = fig.add_subplot(gs[:, 1], projection='3d')
    comp3d = ax4.scatter(core_diam, core_neff, taper_length, c=throughput, cmap='viridis_r', s=50, edgecolor='k')
    ax4.set_xlabel(param_label_1)
    ax4.set_ylabel(param_label_2)
    ax4.set_zlabel(param_label_3)
    cbar4 = plt.colorbar(comp3d, ax=ax4, shrink=0.6)
    cbar4.set_label("Loss Function")

    plt.tight_layout()
    plt.show()
    fig.savefig(f"Bayesian Opt Results\{simulation_val['num_paras']}_params_gridsize_{simulation_val['grid_size']}bayesianresult_hyperparam{hyp_param_b}_{hyp_param_c}.png")

    np.save(f"Bayesian Opt Results\{simulation_val['num_paras']}_params_gridsize_{simulation_val['grid_size']}list_of_results_hyperparam{hyp_param_b}_{hyp_param_c}.npy", tf_list, allow_pickle = True)

###################################################################################################################################################################################################################################################
def find_nearest(arr, val):
    """
    Function to find the index of the nearest value nearest of some number in an array.

    Inputs:
        arr: array to search
        val: float to find the nearest number of
    Returns:
        array[idx]: array value that is closest to the required number
        idx: index of the required number
    """
    array = np.asarray(arr)
    idx = (np.abs(array - val)).argmin()
    return array[idx], idx

def get_wavelength_dependent_indices(wave, simulation_val, fixed_params, sellmeier_df):
    """
    Return absolute refractive indices for one wavelength using fixed offsets
    calculated at the 1.5 um Sellmeier reference wavelength.
    """
    wavelengths = sellmeier_df["Wavelength (um)"].to_numpy()
    _, wave_idx = find_nearest(wavelengths, wave)
    _, ref_idx = find_nearest(wavelengths, 1.5)

    requested_non_ms_core = simulation_val.get("core_neff", fixed_params.get("core_neff"))
    if requested_non_ms_core is None:
        raise KeyError("core_neff missing from simulation_val and fixed_params.")

    requested_cladding = fixed_params["cladding_neff"]
    requested_capillary = fixed_params.get(
        "Capillary Refractive Index",
        fixed_params.get(
            "capillary_neff",
            fixed_params.get("capillary_refractive_index", RSoft_params["background_index"])
        )
    )

    # offsets are relative to silica at the reference wavelength
    non_ms_core_offset = requested_non_ms_core - sellmeier_df["SiO2"].to_numpy()[ref_idx]
    cladding_offset = requested_cladding - sellmeier_df["SiO2"].to_numpy()[ref_idx]
    capillary_offset = requested_capillary - sellmeier_df["SiO2"].to_numpy()[ref_idx]
    
    silica = sellmeier_df["SiO2"].to_numpy()[wave_idx]
    non_ms_core_neff = silica + non_ms_core_offset
    cladding_neff = silica + cladding_offset
    capillary_neff = silica + capillary_offset

    return {
        "non_ms_core_neff": non_ms_core_neff,
        "cladding_neff": cladding_neff,
        "capillary_neff": capillary_neff,
        "background_index": capillary_neff,
        "reference_wavelength": 1.5,
        "non_ms_core_offset": non_ms_core_offset,
        "cladding_offset": cladding_offset,
        "capillary_offset": capillary_offset,
        "silica_index": silica
    }

###################################################################################################################################################################################################################################################

def find_field_base_filenames(folder, wave, field=("ex", "ey", "hx", "hy"), dest_prefix=None):
    """
    Copies LP01 FEM field files (*_ex/_ey/_hx/_hy.m00) for a given wavelength
    into the working directory. When dest_prefix is supplied, the copied files
    are given run-specific names so concurrent BeamPROP runs do not read files
    while another worker is overwriting them.

    Returns:
        list of unique base filenames (ending in .m00)
    """
    folder = Path(folder)
    wave = str(wave)
    cwd = Path.cwd()

    suffixes = tuple(f"_{f}.m00" for f in field)
    base_files = set()

    for f in folder.iterdir():
        if not f.is_file():
            continue

        name = f.name.lower()

        if f"_{wave}_lp01_" not in name:
            continue

        if not name.endswith(suffixes):
            continue

        # identify which field suffix
        for fi in field:
            suf = f"_{fi}.m00"
            if name.endswith(suf):
                source_base = f.name[:-len(suf)]
                if dest_prefix:
                    dest_name = f"{dest_prefix}_{source_base}_{fi}.m00"
                    base_name = f"{dest_prefix}_{source_base}.m00"
                else:
                    dest_name = f.name
                    base_name = source_base + ".m00"

                shutil.copy2(f, cwd / dest_name)
                base_files.add(base_name)
                break

    return sorted(base_files)


def fem_fields_present(base_name, fields=("ex", "ey", "hx", "hy")):
    """
    Check whether FEM field files for a given base name
    already exist in the current working directory.

    base_name: e.g.
      FemSim_File_DET_1.5_LP01_core_diam_..._Taper_L_....m00
    """
    cwd = Path.cwd()
    stem = base_name[:-7]  # strip "_ex_.m00"
    discovered_files = []
    for f in fields:
        path_to_check = cwd / f"{stem}_{f}.m00"
        if not (path_to_check).exists():
            return False
        else:
            discovered_files.append(path_to_check)
    discovered_files = np.array(discovered_files)
    print(f"Found {len(discovered_files)} existing FemSIM files. Using these.")
    return True


def force_file_to_disk(path):
    with open(path, "a") as f:
        f.flush()
        os.fsync(f.fileno())
