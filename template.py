from Circuit_Properties import *
import random
'''
NOTE: Using values from 19CorePL_July2021_noOuter_MMtoSM_extraMM.ind
'''
fixed_params = {
    "core_sep": 80,
    "MCFCladd": 328,
    "cladding_delta": 0.0055,
    "Taper_L": 40000,
    "core_delta": 0.0122895, #0.015,
    # "core_diam": 8.2,
    "alpha": 0,
    "length_hyperparam": 0.01
    }

RSoft_params = {
    "Name": "MCF_Test",
    "cad_aspectratio_x": -1,
    "cad_aspectratio_y": -1,
    "cad_aspectratio_z": -1,
    "boundary_gap_x": 10,
    "boundary_gap_y": 10,
    "boundary_gap_z": 0,
    "dimension": 3,
    "eim": 0,
    "field_output_format": "OUTPUT_REAL_IMAG",
    "background_index": 1.4345,
    "free_space_wavelength": 1.55,
    "sim_tool": Sim_tool.BP,
    "slice_display_mode": "DISPLAY_CONTOURMAPXZ",
    "launch_align_file": 1,
    "launch_normalization": 1,
    "launch_type": LaunchType.MM,
    "grid_size": 1,
    "grid_size_y": 1,
    "step_size": 2,
    "structure": Struct_type.FIBRE,
    "width": 5,
    "height": 5,
    "slice_display_mode": "DISPLAY_NONE"
}
RSoft_params["lambda"] = RSoft_params["free_space_wavelength"]

Launch_params = {
    "monitor_type": Monitor_Prop.FIBRE_MODE_POWER,
    "cladding_monitor_type": Monitor_Prop.TOTAL_POWER,
    "comp": Monitor_comp.BOTH,
    "launch_tilt": 0,
    "launch_align_file": RSoft_params["launch_align_file"],
    "launch_type": RSoft_params["launch_type"],
    "launch_mode": 0,
    "launch_mode_radial": 1,
    "launch_normalization": 1,
    "core_delta": fixed_params["core_delta"],
    "cladding_delta": fixed_params["cladding_delta"]
}

Simulation_params = {
    "core_num": 7,
    "num_paras": 72,
    "batch_num": 6,
    "grid_type": "Hex",
    "Structure": "PL",
    "metric": "TH", # TH = throughput, MS = Mode Selective
    "mode_selective": 0, # 0 == False, 1 == True
    "core_to_monitor": 4
}
Launch_params["core_to_monitor"] = Simulation_params["core_to_monitor"]
Launch_params["launch_random_set"] = random.randint(0,Simulation_params["num_paras"]) # ensures that every simulation sees a different field
RSoft_params["random_set"] = Launch_params["launch_random_set"]

variable_params= {
    "taper": 17.393939393939398, # 10, taken from rough calculation of diameter required to support 7 modes
    # "Taper_L": 40000,
    "core_diam": 8.2,
}

core_params = {}

for i in range(1, Simulation_params["core_num"] + 1):
    if "core_diam" in fixed_params and "core_delta" in fixed_params:
        core_params[f"core_{i}"] = {
        "core_diam": fixed_params["core_diam"],
        "delta": fixed_params["core_delta"]
        }
    elif "core_diam" in variable_params:
        core_params[f"core_{i}"] = {
        "core_diam": variable_params["core_diam"],
        "delta": fixed_params["core_delta"]
        }