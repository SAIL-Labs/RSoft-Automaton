import numpy as np
import subprocess, json

from Circuit_Properties import *
from Functions import *
from HexProperties import *
from rstools import RSoftUserFunction, RSoftCircuit

class RSoftSim:
    def __init__(self, name=""):
        self.name = name
        self.sym = {}
        self.core_log = {}

        # Generate Fibre Parameters
        self.length = 1000
        self.Corediam = 8.2
        self.Claddiam = 125
        self.Core_sep = 60
        self.taper_ratio = 1
        self.core_num = 1
        self.Core_delta = 0.01
        self.background_index = 1.456
        self.delta = 0.0036# 0.012
        self.grid_type = "Hex"

        # Multiprocessing Parameters
        self.num_paras = 288
        self.batch_number = 12

        # Simulation Setup
        self.Dx = 0.1
        self.Dy = self.Dx
        self.Dz = 0.2
        self.Phase = 0
        self.boundary_gap_x = 10
        self.boundary_gap_y = self.boundary_gap_x
        self.boundary_gap_z = 0
        self.bpm_pathway = 1
        self.bpm_pathway_monitor = self.bpm_pathway
        self.sim_tool = Sim_tool.BP
        self.width = 5
        self.height = self.width
        self.H = self.height
        self.grid_uniform = 0
        self.eim = 0
        self.polarization = 0
        self.free_space_wavelength = 1.55
        self.k0 = 2 * np.pi / self.free_space_wavelength
        self.slice_display_mode = "DISPLAY_CONTOURMAPXZ"
        self.slice_position_z = 100
        
        # CAD Stuff if you like the GUI or just want to check ind files
        self.cad_aspectratio_z = -1
        self.cad_aspectratio_y = -1
        self.cad_aspectratio_x = -1

        # Launch Properties
        self.monitor_type =  Monitor_Prop.FIBRE_MODE_POWER
        self.comp = Monitor_comp.BOTH
        self.launch_port = 1
        self.launch_type = LaunchType.SM
        self.launch_file = ""
        self.launch_random_set = 0 if self.launch_type == LaunchType.SM else 1
        self.launch_tilt = 1 if self.launch_type == LaunchType.SM else 0
        self.launch_align_file = 1
        self.launch_mode = 0 if self.launch_type == LaunchType.SM else "*"
        self.launch_mode_radial = 1 if self.launch_type == LaunchType.SM else "*"
        self.launch_normalization = 1
        self.grid_size = self.Dx
        self.grid_size_y = self.Dy
        self.step_size = self.Dz
        self.monitor_width = self.Corediam * 1.1
        self.monitor_height = self.monitor_width * 1.1

        self.structure = Struct_type.FIBRE
    
    def fixed_parameters_dict(self):
            return {
                "Dx": self.Dx,
                "Dy": self.Dy,
                "Dz": self.Dz,
                "Phase": self.Phase,
                "boundary_gap_x": self.boundary_gap_x,
                "boundary_gap_y": self.boundary_gap_y,
                "boundary_gap_z": self.boundary_gap_z,
                "bpm_pathway": self.bpm_pathway,
                "bpm_pathway_monitor": self.bpm_pathway_monitor,
                "sym_tool": self.sim_tool,
                "width": self.width,
                "height": self.height,
                "H": self.H,
                "grid_uniform": self.grid_uniform,
                "eim": self.eim,
                "polarization": self.polarization,
                "free_space_wavelength": self.free_space_wavelength,
                "k0": self.k0,
                "slice_display_mode": self.slice_display_mode,
                "slice_position_z": self.slice_position_z,
                "cad_aspectratio_z": self.cad_aspectratio_z,
                "cad_aspectratio_y": self.cad_aspectratio_y,
                "cad_aspectratio_x": self.cad_aspectratio_x,
                "grid_size": self.grid_size,
                "grid_size_y": self.grid_size_y,
                "step_size": self.step_size,
                "structure": self.structure,
                "num_paras": self.num_paras,
                "batch_number": self.batch_number
            }

    def variable_parameters_dict(self):
        return {
            "Name": self.name,
            "Length": self.length,
            "Corediam": self.Corediam,
            "Claddiam": self.Claddiam,
            "Core_sep": self.Core_sep,
            "taper_ratio": self.taper_ratio,
            "core_num": self.core_num,
            "Core_delta": self.Core_delta,
            "background_index": self.background_index,
            "delta": self.delta,
        }

    def fixed_parameters_dict(self):
        return {
            "Dx": self.Dx,
            "Dy": self.Dy,
            "Dz": self.Dz,
            "Phase": self.Phase,
            "boundary_gap_x": self.boundary_gap_x,
            "boundary_gap_y": self.boundary_gap_y,
            "boundary_gap_z": self.boundary_gap_z,
            "bpm_pathway": self.bpm_pathway,
            "bpm_pathway_monitor": self.bpm_pathway_monitor,
            "sym_tool": self.sim_tool,
            "width": self.width,
            "height": self.height,
            "H": self.H,
            "grid_uniform": self.grid_uniform,
            "eim": self.eim,
            "polarization": self.polarization,
            "free_space_wavelength": self.free_space_wavelength,
            "k0": self.k0,
            "slice_display_mode": self.slice_display_mode,
            "slice_position_z": self.slice_position_z,
            "cad_aspectratio_z": self.cad_aspectratio_z,
            "cad_aspectratio_y": self.cad_aspectratio_y,
            "cad_aspectratio_x": self.cad_aspectratio_x,
            "grid_size": self.grid_size,
            "grid_size_y": self.grid_size_y,
            "step_size": self.step_size,
            "structure": self.structure,
            "num_paras": self.num_paras,
            "batch_number": self.batch_number
        }

    def launch_parameters_dict(self):
        return {
            "monitor_type": self.monitor_type,
            "comp": self.comp,
            "launch_tilt": self.launch_tilt,
            "launch_port": self.launch_port,
            "launch_align_file": self.launch_align_file,
            "launch_mode": self.launch_mode,
            "launch_file": self.launch_file,
            "launch_type": self.launch_type,
            "launch_random_set": self.launch_random_set,
            "launch_mode_radial": self.launch_mode_radial,
            "launch_normalization": self.launch_normalization,
            "monitor_width": self.monitor_width,
            "monitor_height": self.monitor_height
        }

    def init_priors(self, custom = None):
        base_priors = {
            "Corediam": (1.0, 50.0)
            }
        if custom:
            base_priors.update(custom)
        self.prior_space = base_priors
        return base_priors
    
    def save_all_json(self):
        with open("variable_paras.json", "w") as g:
            json.dump(self.variable_parameters_dict(), g)
        with open("fibre_prop.json", "w") as f:
            json.dump(self.fixed_parameters_dict(), f)
        with open("launch_para.json", "w") as l:
            json.dump(self.launch_parameters_dict(), l)
        with open("prior_space.json", "w") as write:
            json.dump(self.prior_space, write)
    
    def load_json_parameters(self):
        """
        Load variable, fixed, and launch parameters from JSON and build symbol dictionary.
        """
        with open("variable_paras.json", "r") as g:
            param_v = json.load(g)
        with open("fibre_prop.json", "r") as f:
            params = json.load(f)
        with open("launch_para.json", "r") as l:
            launch_params = json.load(l)

        self.sym = {**param_v, **params}
        self.launch_params = launch_params

    def generate_core_positions(self):
        if self.grid_type == "Hex":
            """
            Generate hexagonal core coordinates and store internally.
            """
            core_num = self.sym["core_num"]
            if core_num % 2 == 0:
                raise ValueError(f"The number of cores must be odd to perfectly fit inside the hex grid. Received: {core_num}")

            row_numbers = [number_rows(core_num)]
            self.core_positions = []  # store tuples of (x, y)
            for row_num in row_numbers:
                hcoord, vcoord = generate_hex_grid(row_num, self.sym["Core_sep"])
                self.core_positions = list(zip(hcoord, vcoord))
        if self.grid_type == "Pent":
            estimated_radius = estimate_pentagon_radius(self.core_num,self.Core_sep)
            hcoord, vcoord= generate_filled_pentagon_grid(estimated_radius, self.Core_sep)
            self.core_positions = list(zip(hcoord, vcoord))


    def build_circuit(self):
        """
        Create the design file using the loaded symbols and write to .ind file.
        """
        self.circuit = RSoftCircuit()
        for key, val in self.sym.items():
            self.circuit.set_symbol(key, val)

        name = self.sym["Name"]
        core_beg_dims = (self.sym['Corediam'] / self.sym['taper_ratio'], self.sym['Corediam'] / self.sym['taper_ratio'])
        core_end_dims = (self.sym['Corediam'] , self.sym['Corediam'])
        cladding_beg_dims = (self.sym['Claddiam'] / self.sym['taper_ratio'], self.sym['Claddiam'] / self.sym['taper_ratio'])
        cladding_end_dims = (self.sym['Claddiam'] , self.sym['Claddiam'])
        core_name = [f"core_{n+1:02}" for n in range(self.sym['core_num'])]

        if self.sym['core_num'] == 1:

            for j, (x, y) in enumerate(self.core_positions):
                core = self.circuit.add_segment(
                    position=(x, y, 0),
                    offset=(x, y, 'Length'),
                    dimensions=('Corediam','Corediam'),
                    dimensions_end=('Corediam','Corediam')
                )
                core.set_name(core_name[j])

                # log individual core parameters
                self.core_log[core_name[j]] = {
                    "position": (x, y, 0),
                    "offset": (x, y, 'Length'),
                    "dimensions": ('Corediam','Corediam'),
                    "dimensions_end": ('Corediam','Corediam'),
                    "Core index": self.sym['Core_delta'],
                    }
                cladding = self.circuit.add_segment(
                    position=(0, 0, 0),
                    offset=(0, 0, 'Length'),
                    dimensions=('Claddiam','Claddiam'),
                    dimensions_end=('Claddiam','Claddiam')
                    )
                cladding.set_name("Super Cladding")
            
        if self.sym['core_num'] != 1:
            cladding = self.circuit.add_segment(
                position=(0, 0, 0),
                offset=(0, 0, self.length),
                dimensions=cladding_beg_dims,
                dimensions_end=cladding_end_dims
                )
            cladding.set_name("Super Cladding")
            for j, (x, y) in enumerate(self.core_positions):
                core = self.circuit.add_segment(
                    position=(x / self.sym['taper_ratio'], y / self.sym['taper_ratio'], 0),
                    offset=(x, y, 'Length'),
                    dimensions=core_beg_dims,
                    dimensions_end=core_end_dims
                )
                core.set_name(core_name[j])

                # log individual core parameters
                self.core_log[core_name[j]] = {
                    "position": (x / self.sym['taper_ratio'], y / self.sym['taper_ratio'], 0),
                    "offset": (x, y, self.length),
                    "dimensions": (self.Corediam,self.Corediam),
                    "dimensions_end": (self.Corediam,self.Corediam),
                    "Core index": self.sym['Core_delta'],
                    }

        self.circuit.write(f"{name}.ind")
        with open(f"{self.name}.ind", "r") as og:
            og_lines = og.readlines()
        # Insert cladding profile type after cladding segment start
        line_mod = insert_after_match(og_lines,"comp_name = Super Cladding", 
                                      f"\tprofile_type = PROF_INACTIVE\n")

        with open(f"{name}.ind", "w") as out:
            out.writelines(line_mod)  
        """
        Append all pathway, monitor, and launch field blocks based on launch parameters.
        """ 
        name = self.sym["Name"]
        core_num = self.sym["core_num"]
        background_index = self.sym["background_index"]
        AddHack(name, self.launch_params, int(core_num), background_index)
    
    def print_core_log(self):
        for core_name, core_info in self.core_log.items():
            print(f"{core_name}")
            for key, val in core_info.items():
                print(f"{key}: {val}")
            print("") 

    def RunRSoft(self):
        '''
        Multiprocessing must to be run outside of a Jupyter cell or it will silently 
        fail/infinitely loop on the first batch
        '''
        subprocess.run(["python", "Multiprocessing.py"], check=True)