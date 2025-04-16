from rstools import RSoftCircuit

import numpy as np, json
from HexProperties import generate_hex_grid, number_rows

# name the output file
name = 'MCF_Test'

# Collection of symbols to use wihtin the RSoft CAD
sym = {}

#############################################################################################################################################################################################
''' 
Define the fibre and RSoft properties 
'''

# properties of the structure (units of um)
with open("fibre_prop.json", "r") as f:
    params = json.load(f)

for key in params:
    sym[key] = params[key]

# creating the design file, load the seetings and add symbols
c = RSoftCircuit()
for key in sym:
    c.set_symbol(key,sym[key])

#############################################################################################################################################################################################
''' 
Generating the positional coordinates for each fibre 
'''
# this must be an odd number!!
core_num = [sym['core_num']]

for i in range(len(core_num)):
    if core_num[i] % 2 == 0:
        raise ValueError(f"The number of cores must be odd to perfectly fit inside the hex grid. Received:{core_num}")
    
row_numbers = [number_rows(n) for n in core_num]

for idx, row_num in enumerate(row_numbers):
    hcoord, vcoord = generate_hex_grid(row_num, sym["Core_sep"])

#############################################################################################################################################################################################
''' 
Generating the segments and assigning the positional coordinates found in 
the previous sections 
'''

core_name = [f"core_{n+1:02}" for n in range(7)]

cladding = c.add_segment(
                position=(0,0,0), offset=(0,0,'Length'), 
                dimensions = (sym['Claddiam']/sym['taper_ratio'],sym['Claddiam']/sym['taper_ratio']), # /sym['taper_ratio']
                dimensions_end = (('Claddiam', 'Claddiam'))
                )

cladding.set_name("MMF Cladding")

for j, (x, y) in enumerate(zip(hcoord, vcoord)):

    core = c.add_segment(position=(x/sym['taper_ratio'],y/sym['taper_ratio'],0), offset=(x,y,'Length'), 
                dimensions = (sym['Corediam']/sym['taper_ratio'], sym['Corediam']/sym['taper_ratio']), 
                dimensions_end = (('Corediam', 'Corediam')))

    core.set_name(core_name[j])

#############################################################################################################################################################################################

c.write('%s.ind'%name)