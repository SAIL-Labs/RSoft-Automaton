from Circuit_Properties import *
import random, numpy as np

fixed_params = {
    "core_sep":  125,
    "MCFCladd": 3*125,
    # "core_claddings": None, 
    "core_cladding_diam": None, #125  #80,,1.4402
    "cladding_neff": 1.44402, # this value gets replaced through the Sellmeier equation
    "core_cladding_neff": None, #1.4433510951304291734406199895015, #from using SMF-28 index (). Determined automatically using Sellmeier
    "cen_core_cladding_neff": None,#1.44,#0.00949,
    "other_core_diam": 8.3, # non-ms core diameter
    "silica_index": 0, #this is changed through the sellmeier values
    # "Taper_L": 50000, #45000
    # "core_neff": 0.0122895, #0.015,
    "taper": 25, #7 core pigtail gif #19.34984520123839, #19c pigtail #18.75, 7c pigtail
    # "core_diam": 8.2,
    "alpha": 0,
    "length_hyperparam": 0.01
    }

RSoft_params = {
    "cad_aspectratio_x": -1,
    "cad_aspectratio_y": -1,
    "cad_aspectratio_z": -1,
    "boundary_gap_x": 10,
    "boundary_gap_y": 10,
    "boundary_gap_z": 0,
    "femsim_boundary_gap_x": 40, # um, changes the domain size of the femsim calculations performed over the MS core
    "femsim_boundary_gap_y": 40, # um, changes the domain size of the femsim calculations performed over the MS core
    "bpm_output_monitors": 1,
    "bpm_output_monitors_warned": 1,
    "dimension": 3,
    "eim": 0,
    "field_output_format": "OUTPUT_AMP_PHASE",
    "slice_output_format": "OUTPUT_AMP_PHASE",
    "slice_output_individual": "None", # OUTPUT_AMP_PHASE_3D
    "background_index": 1.4345,#replace through sellmeier
    "background_index_offset": 0, # to be used when simulate_tf_metric = False. Default = 0, 0.001295136
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
    "use_profile": False, # determines whether user profiles are being used
    "num_profile": 1, # number of user profiles to use
    "profile": ["testing_1d.dat"], # name of the user profile
    "mode_selective": 0, # 0 == False, 1 == True
    "core_to_monitor": 4,
    "all_modes": False,
    "port_mon_file": None,
    "skip_core": None, # if you want to skip a certain core, set this to the 0-index core number
    "fixed_fem_file": False,
    # specify industry values
    "industry_neff_values": False,
    "industry_neff_file": None,
    "free_space_wavelength": None,

    # stuff for monitoring higher order mode indices in the MS core
    "higher_order_modes": [2, 4, 6, 8, 10],
    "higher_mode_indices": [2, 4, 6, 8, 10],
    # maximum l in LPln used by ofiber to calculate the total number of modes present in the fibre geometery 
    # e.g. max_ell = 3 --> calculate propagation constants up to LP31, or any LP3n
    "max_ell": 4,

    # optimiser settings
    "acq_type": "EI",
    "acq_hyperparam": 0.8,
    "acq_opt": "lbfgs",
    "n_init_points": 3,
    "use_previous_results": False, # decides whether to seed a new optimiser with previous results
    "previous_results": None, # name of the file containing the previous results

    # RSoft meta instance info
    "number_of_rsoft_instances": 2
}

# Polychromatic/monochromatic switch
if Simulation_params["free_space_wavelength"] is None:
    RSoft_params["free_space_wavelength"] = [1.5]
else:
    RSoft_params["free_space_wavelength"] = Simulation_params["free_space_wavelength"]
RSoft_params["lambda"] = RSoft_params["free_space_wavelength"]

bestvals = {         		            
    # "core_diam": 8.601, 		
    # "core_neff": 1.4579, 
    # "Taper_L": 45000 
}

bestval_limits = {         		            
    # "core_diam": (5.5, 20), 	
    # "core_neff": (1.4461, 1.465), 
    # "Taper_L": (30000,50000) 
}

# variable_params= {
#     "core_diam": 8.3, 
#     "core_neff": 1.4492, 
#     "Taper_L":  50000, 
# }   

# GIF settings     
variable_params= {
    "core_diam": 8.3, 
    "core_neff": 1.4492, 
    "Taper_L":  50000, 
}       
# variable_params= {
#     "core_diam": 125, 
#     "core_neff": 1.4578044315, 
#     "Taper_L":  50000, 
# }       

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
        "core_diam": fixed_params["other_core_diam"],#variable_params["core_diam"],
        # "neff": 1.4467895-RSoft_params["background_index"], #variable_params["core_neff"], 1.442906 2 mol% Ge, 1.4467895
        "taper": variable_params.get("taper", fixed_params.get("taper"))
        }
    elif "core_neff" in fixed_params:
        core_params[f"core_{i}"] = {
        "core_diam": fixed_params["other_core_diam"],#variable_params["core_diam"],
        "neff": fixed_params.get("core_neff") - RSoft_params["background_index"], #variable_params["core_neff"]
        "taper": fixed_params.get("taper")
        }
