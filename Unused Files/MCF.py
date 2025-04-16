from rstools import RSoftCircuit
import numpy as np
# name the output file
name = 'McF_Test'

# Collection of symbols to use wihtin the RSoft CAD
sym = {}

# properties of the structure (units of um)
sym['Length'] = 1000
sym['Corediam'] = 8.2
sym['Claddiam'] = 125
sym['H'] = 'height'

# simulation settings
sym['Dx'] = 0.05
sym['Dy'] = sym['Dx']
sym['Dz'] = 0.2
sym['boundary_gap_x'] = 30
sym['boundary_gap_y'] = sym['boundary_gap_x']
sym['boundary_gap_z'] = 0
sym['bpm_pathway'] = 1
sym['bpm_pathway_monitor'] = sym['bpm_pathway']
sym['sym_tool'] = 'ST_BEAMPROP'
sym['delta'] = 0.012
sym['width'] = 5
sym['height'] = sym['width']
sym['background_index'] = 1.456
sym['grid_nonuniform'] = 0
sym['eim'] = 0
sym['polarization'] = 0
sym['free_space_wavelength'] = 1
sym['k0'] = 2*np.pi / sym['free_space_wavelength']
sym['slice_display_mode'] = 'DISPLAY_CONTOURMAPXZ'
sym['slice_position_z'] = 100

# launch properties
sym['launch_tilt'] = 1
sym['launch_port'] = 1
sym['launch_align_file'] = 1
sym['launch_type'] = 'LAUNCH_FIBREMODE'
sym['launch_mode'] = 0
sym['launch_mode_radial'] = 1
sym['launch_normalization'] = 1
sym['grid_size'] = 'Dx'
sym['grid_size_y'] = 'Dy'
sym['step_size'] = 'Dz'

# set the global segment structure, other use segment.structure( ‘FIBER’ )
sym['structure'] = 'STRUCT_FIBER'

# creating the design file, load the seetings and add symbols
c = RSoftCircuit()
for key in sym:
    c.set_symbol(key,sym[key])

## adding segments and attaching them
# starting vertex
central_core = c.add_segment(position=(0,0,0), offset=(0,0,'Length'), 
                    dimensions = ('Corediam', 'Corediam'))
left_upper_core = c.add_segment(position=(0,0,0),offset=(0,0,'Length'),
                                 dimensions=('Corediam', 'Corediam'))
left_lower_core = c.add_segment(position=(0,0,0),offset=(0,0,'Length'), 
                                dimensions=('Corediam', 'Corediam'))
bottom_core = c.add_segment(position=(0,0,0),offset=(0,0,'Length'),
                                dimensions=('Corediam', 'Corediam'))
right_lower_core = c.add_segment(position=(0,0,0),offset=(0,0,'Length'), 
                                 dimensions=('Corediam', 'Corediam'))
right_upper_core = c.add_segment(position=(0,0,0),offset=(0,0,'Length'), 
                                 dimensions=('Corediam', 'Corediam'))
upper_core = c.add_segment(position=(0,0,0),offset=(0,0,'Length'),
                                dimensions=('Corediam', 'Corediam'))

# Naming the segments
central_core.set_name("Central Core")
left_upper_core.set_name("Left Upper Core")
left_lower_core.set_name("Left Lower Core")
bottom_core.set_name("Bottom Core")
right_lower_core.set_name("Right Lower Core")
right_upper_core.set_name("Right Upper Core")
upper_core.set_name("Upper Core")

# attaching segments
c.attach(left_upper_core,central_core,offset=(-0.5*(sym['Claddiam']/2),
                                          0.25*(sym['Claddiam']/2),
                                           -sym['Length']))
c.attach(left_lower_core,central_core,offset=(-0.5*(sym['Claddiam']/2),
                                          -0.25*(sym['Claddiam']/2),
                                           -sym['Length']))
c.attach(bottom_core,central_core,offset=(0,-0.5*(sym['Claddiam']/2),-sym['Length']))
c.attach(right_lower_core,central_core,offset=(0.5*(sym['Claddiam']/2),
                                          -0.25*(sym['Claddiam']/2),
                                           -sym['Length']))
c.attach(right_upper_core,central_core,offset=(0.5*(sym['Claddiam']/2),
                                          0.25*(sym['Claddiam']/2),
                                           -sym['Length']))
c.attach(upper_core,central_core,offset=(0,0.5*(sym['Claddiam']/2),-sym['Length']))

c.write('%s.ind'%name)