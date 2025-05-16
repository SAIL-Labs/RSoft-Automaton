import numpy as np
from Circuit_Properties import *

fixed_params = {
    "core_sep": 80,
    "MCFCladd": 300,
    "cladding_delta": 0.0015,
    "core_delta": 0.015,
    "core_diam": 8.2,
    "alpha": 0,
    "length_hyperparam": 0.01
    }

# fixed_params["k0"] = (2 * np.pi) / fixed_params["free_space_wavelength"]

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
    "background_index": 1.456,
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
    "height": 5
}
RSoft_params["lambda"] = RSoft_params["free_space_wavelength"]

Launch_params = {
    "monitor_type": Monitor_Prop.FIBRE_MODE_POWER,
    "cladding_monitor_type": Monitor_Prop.TOTAL_POWER,
    "comp": Monitor_comp.BOTH,
    "launch_tilt": 0,
    "launch_random_set": 0,
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
    "Structure": "PL"
}
variable_params= {
    "taper": 10,
    "Taper_L": 40000,
}
