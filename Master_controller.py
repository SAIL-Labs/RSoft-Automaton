import subprocess, os, shutil, numpy as np, csv, json

from Stats import *
from Functions import *
from rstools import RSoftUserFunction, RSoftCircuit
from HexProperties import generate_hex_grid, number_rows

'''
Now used inside the jupyter notebook file!
'''
# # name the output file
# name = 'MCF_Test'

# # Collection of symbols to use wihtin the RSoft CAD
# sym = {}

#############################################################################################################################################################################################
# ''' 
# Define the fibre and RSoft properties 
# '''

# # properties of the structure (units of um)
# with open("fibre_prop.json", "r") as f:
#     params = json.load(f)

# for key in params:
#     sym[key] = params[key]

# # creating the design file, load the seetings and add symbols
# c = RSoftCircuit()
# for key in sym:
#     c.set_symbol(key,sym[key])

#############################################################################################################################################################################################
# ''' 
# Generating the positional coordinates for each fibre 
# '''
# # this must be an odd number!!
# core_num = [sym['core_num']]

# for i in range(len(core_num)):
#     if core_num[i] % 2 == 0:
#         raise ValueError(f"The number of cores must be odd to perfectly fit inside the hex grid. Received:{core_num}")
    
# row_numbers = [number_rows(n) for n in core_num]

# for idx, row_num in enumerate(row_numbers):
#     hcoord, vcoord = generate_hex_grid(row_num, sym["Core_sep"])

#############################################################################################################################################################################################
# ''' 
# Generating the segments and assigning the positional coordinates found in 
# the previous sections 
# '''

# core_name = [f"core_{n+1:02}" for n in range(7)]

# cladding = c.add_segment(
#                 position=(0,0,0), offset=(0,0,'Length'), 
#                 dimensions = (sym['Claddiam']/sym['taper_ratio'],sym['Claddiam']/sym['taper_ratio']), # /sym['taper_ratio']
#                 dimensions_end = (('Claddiam', 'Claddiam'))
#                 )

# cladding.set_name("MMF Cladding")

# for j, (x, y) in enumerate(zip(hcoord, vcoord)):

#     core = c.add_segment(position=(x/sym['taper_ratio'],y/sym['taper_ratio'],0), offset=(x,y,'Length'), 
#                 dimensions = (sym['Corediam']/sym['taper_ratio'], sym['Corediam']/sym['taper_ratio']), 
#                 dimensions_end = (('Corediam', 'Corediam')))

#     core.set_name(core_name[j])

# c.write('%s.ind'%name)
#############################################################################################################################################################################################
# '''
# Controller
# '''
# # # This runs the original script with rspython and generates
# # subprocess.run(["rspython", "MCF_Test.py"], check=True)
# uf = RSoftUserFunction()

# # extract the number of cores from the template file
# core_num = int(Extract_params("core_num "))
# Corediam = Extract_params("Corediam ")
# fibre_length = Extract_params("Length")
# Core_delta = Extract_params("Core_delta") 

# with open("prior.json", "r") as f:
#     priors = json.load(f)
#     core_diam = priors["core_diam"]
###############################################################################################################################################
# '''
# This is the hacking part which adds all of the pathways, then the monitors and then the 
# launch fields. They are done individually to account for formatting issues that may exist
# within RSoft's .ind files.
# '''

# block_text = {
#     "pathway":'''
#     pathway {n}
#             {n}
#     end pathway
#     ''',
#     "monitor": '''
#     monitor {n}
#             pathway = {n}
#             monitor_type = MONITOR_WGMODE_POWER
#             monitor_tilt = 1
#             monitor_component = COMPONENT_BOTH
#             monitor_width = {Corediam}
#             monitor_height = {Corediam}
#     end monitor
#     ''',
#     "launch_field":'''
#     launch_field {n}
#             launch_pathway = {n}
#             launch_type = LAUNCH_WGMODE
#             launch_tilt = 1
#             launch_align_file = 1
#     end launch_field
#     '''
# }

# # appends the above text to the .ind file
# with open('%s.ind'%name, "a") as f:
#     for block_type in ["pathway","monitor","launch_field"]:
#         for i in range(1,int(core_num) + 1):

#             if block_type == "launch_field" and i != 1:
#                 continue

#             text = block_text[block_type].format(n=i,Corediam=Corediam)
#             f.write(text)

# # Add separate effective index for the cores.
# insert_param_into_file('%s.ind'%name,Core_delta)

###############################################################################################################################################
'''
FAIL SAFE: writes only the symbols listed in the template folder
'''
# # Append extracted block to pre-generated .ind file
# with open("MCF_Test.ind", "a") as f:
#     f.writelines((pathway_block,monitor_block,launchfield_block))

###############################################################################################################################################
# '''
# Manual setup to loop through a list of values
# Runs the terminal line that will initiate RSoft. 
# All output files will appear in a subfolder on the Desktop (windows)
# '''

# # initialise results dictionary
# results = {}
# l = 0

# # open and read the generated .ind file
# with open('%s.ind'%name, "r") as r:
#     lines = r.readlines()

#     # loop through list of parameters
#     '''
#     To add: make this universal so any list of parameters can be looped through, not just core diameter
#     '''
#     # for core in core_diam:
#     modified_line = []
#     for line in lines:
#         if line.strip().startswith("Corediam ="):
#             rep_line = f"Corediam = {core_diam}\n"
#             modified_line.append(rep_line)
#         else:
#             modified_line.append(line)

#     # generate separate .ind file based on parameters
#     with open(f"MCF_{core_diam}.ind", "w") as out:
#         out.writelines(modified_line)

#     # This runs the beamprop simulation and also allocates the results in their own folder on the desktop
#     filename = f"MCF_{core_diam}.ind"
#     prefix = f"prefix=Corediam_{core_diam}"
#     Folder_name = f"CoreSim_{core_diam}"

#     desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
#     results_folder_root = os.path.join(desktop_path,"Results")
#     os.makedirs(results_folder_root, exist_ok=True)

#     results_folder = os.path.join(results_folder_root,Folder_name)
#     os.makedirs(results_folder, exist_ok = True)

#     subprocess.run(["bsimw32", filename, prefix, "wait=0"], check=True)

#     '''
#     To do: make this test metric a choice. Incorporate other metric and make it an option to choose between metrics to test.
#     '''
#     # read pathway monitor file
#     uf.read(f'Corediam_{core_diam}.mon')
#     (x,y) = uf.get_arrays()
#     # x is the fibre length
#     # y is the pathway monitor power

#     average_throughput = (sum(y))/len(y)

#     results[l] = y
#     l += 1

#     for file in os.listdir():
#         if file.startswith(f"Corediam_{core_diam}"):
#             shutil.move(file, os.path.join(results_folder, file))
#         if file.startswith(filename):
#             shutil.move(file, os.path.join(results_folder, file))

# # using throughput as the test metric, saves pathway monitor results to a separate excel file
# source_path = os.path.join("C:\\Users\\justinvella\\OneDrive - The University of Sydney (Students)", "Apps", "New Caprica Stuff", "Python", "MCF", "Throughput_results.csv")
# destination_path = os.path.join("C:\\Users\\justinvella\\OneDrive - The University of Sydney (Students)", "Apps","VSCode", "Throughput", "Files", "Throughput_results.csv")

# with open("Throughput_results.csv", mode = "w", newline="") as file:
#     writer = csv.writer(file)
#     header = ["x"] + [f"{core_diam:.2f}"]
#     writer.writerow(header)

#     for i in range(len(x)):
#         row = [x[i]] + [results[j][i] for j in range(l)]
#         writer.writerow(row)
# shutil.copy(source_path,destination_path)
###############################################################################################################################################