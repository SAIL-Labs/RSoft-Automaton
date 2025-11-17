from Circuit_Properties import *
import random, numpy as np
'''
NOTE: Using values from 19CorePL_July2021_noOuter_MMtoSM_extraMM.ind
'''
fixed_params = {
    "core_sep": 35, #35, # 120
    "MCFCladd": 125, # 380 #125
    # "core_claddings": None, 
    "core_cladding_diam": None, # None #80,
    "cladding_neff": 1.44,#0.0055, #0.0095,
    "core_cladding_neff": None,
    "cen_core_cladding_neff": None,#0.00949,
    # "Taper_L": 50000, #45000
    # "core_neff": 0.0122895, #0.015,
    "taper": 7.33207692,
    # "core_diam": 8.2,
    "alpha": 0,
    "length_hyperparam": 0.01
    }

RSoft_params = {
    # "Name": "MCF_Test",
    "cad_aspectratio_x": -1,
    "cad_aspectratio_y": -1,
    "cad_aspectratio_z": -1,
    "boundary_gap_x": 10,
    "boundary_gap_y": 10,
    "boundary_gap_z": 0,
    "bpm_output_monitors": 1,
    "bpm_output_monitors_warned": 1,
    "dimension": 3,
    "eim": 0,
    "field_output_format": "OUTPUT_AMP_PHASE",
    "slice_output_format": "OUTPUT_AMP_PHASE",
    "slice_output_individual": "None", # OUTPUT_AMP_PHASE_3D
    "background_index": 1.4345,
    "free_space_wavelength": 1.5,
    "sim_tool": Sim_tool.BP,
    "launch_align_file": 1,
    "launch_normalization": 1,
    "launch_type": LaunchType.MM,
    "grid_size": 1,
    "grid_size_y": 1,
    "step_size": 2,
    "structure": Struct_type.FIBRE,
    "slice_display_mode": "DISPLAY_CONTOURMAPXZ",
    "fem_iterations": 1000,
    "fem_nev": 12
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
    "launch_phase": 0,
    "monitor_normalization": 0, # 0 == Input Power, 1 == Local Power
    "mon_type": "port_mon", # pathway_mon
    # "monitor_output": 0,
    # "monitor_step_size": 10,
    # "monitoroutputformat": "OUTPUT_AMP_PHASE",
    # "core_neff": fixed_params["core_neff"],
    "cladding_neff": fixed_params["cladding_neff"],
    # "core_cladding_neff": fixed_params["core_cladding_neff"],
    # "cen_core_cladding_neff": fixed_params["cen_core_cladding_neff"]
}

Simulation_params = {
    "core_num": 7,
    "num_paras": 72,
    "batch_num": 6,
    "hyp_param_b": 1,
    "hyp_param_c": 1,
    "grid_type": "Hex",
    "plot_centre_core": True,
    "Structure": "PL", # Fibre, PL, pigtail
    "metric": "TH", # TH = throughput, MS = Mode Selective, TF = Transfer Vector
    "add_cladding_to_cores": None, # This must be zero-indexed!
    "mode_selective": 0, # 0 == False, 1 == True
    "core_to_monitor": 4,
    "port_mon_file": None,
    "skip_core": None, # if you want to skip a certain core, set this to the 0-index core number
    "fixed_fem_file": False,
    # specify industry values
    "industry_neff_values": False,
    "industry_neff_file": None
}

variable_params= {
    "core_diam": 6.5, #8.3, np.array([30.0])
    "core_neff": 1.4467895,#0.0122895, #0.0157,#
    # "taper": 6.55789308, #22, #8.53
    "Taper_L": 50000,
}       

# Assign core_neffs here 
Launch_params["core_neff"] = variable_params.get("core_neff", fixed_params.get("core_neff"))
if Launch_params["core_neff"] is None:
    raise KeyError("core_neff missing from both variable_params and fixed_params")

for k in variable_params.keys():
    if k == "free_space_wavelength":
        RSoft_params[k] = variable_params[k]
RSoft_params["lambda"] = RSoft_params["free_space_wavelength"]

RSoft_params["width"] = variable_params["core_diam"]
RSoft_params["height"] = variable_params["core_diam"]
Launch_params["core_to_monitor"] = Simulation_params["core_to_monitor"]
Launch_params["launch_random_set"] = 0 #random.randint(0,Simulation_params["num_paras"]) # ensures that every simulation sees a different field
RSoft_params["random_set"] = Launch_params["launch_random_set"]
fixed_params["MMF_Taper"] = variable_params.get("taper", fixed_params.get("taper"))

core_params = {}

for i in range(1, Simulation_params["core_num"] + 1):
    if "core_diam" in fixed_params and "core_neff" in fixed_params:
        core_params[f"core_{i}"] = {
        "core_diam": fixed_params.get("core_diam"),
        "neff": fixed_params.get("core_neff") - RSoft_params["background_index"],
        "taper": variable_params.get("taper")
        }

    elif "core_diam" in variable_params and "core_neff" in variable_params:
        core_params[f"core_{i}"] = {
        "core_diam": 6.5,#variable_params["core_diam"],
        "neff": 1.4467895-RSoft_params["background_index"], #variable_params["core_neff"]
        "taper": variable_params.get("taper", fixed_params.get("taper"))
        }
    elif "core_neff" in fixed_params:
        core_params[f"core_{i}"] = {
        "core_diam": 6.5,#variable_params["core_diam"],
        "neff": fixed_params.get("core_neff") - RSoft_params["background_index"], #variable_params["core_neff"]
        "taper": fixed_params.get("taper")
        }