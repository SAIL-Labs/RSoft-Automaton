# from rstool import RSoftCircuit

# class vec3d:
#     x = float
#     y = float
#     z = float

#     def __init__(self, x_pos, y_pos, z_pos):
#         self.x = x_pos
#         self.y = y_pos
#         self.z = z_pos

class Launch_Prop:
    FILE = "LAUNCH_FILE"
    COMPUTE = "LAUNCH_COMPMODE"
    WGMODE = "LAUNCH_WGMODE"
    GAUSSIAN = "LAUNCH_GAUSSIAN"
    MULTIMODE = "LAUNCH_MULTIMODE"
    PLANEWAVE = "LAUNCH_PLANEWAVE"

class Monitor_Prop:
    FILE_POWER = "MONITOR_FILE_POWER"
    FILE_PHASE = "MONITOR_FILE_PHASE"
    WGMODE_POWER = "MONITOR_WGMODE_POWER"
    WGMODE_PHASE = "MONITOR_WGMODE_PHASE"
    GAUSS_POWER = "MONITOR_GAUSS_POWER"
    GAUSS_PHASE = "MONITOR_GAUSS_PHASE"
    LAUNCH_POWER = "MONITOR_LAUNCH_POWER"
    LAUNCH_PHASE = "MONITOR_LAUNCH_PHASE"
    WG_POWER = "MONITOR_WG_POWER"
    TOTAL_POWER = "MONITOR_TOTAL_POWER"
    FIELD_NEFF = "MONITOR_FIELD_NEFF"
    FIELD_WIDTH = "MONITOR_FIELD_WIDTH"
    FIELD_HEIGHT = "MONITOR_FIELD_HEIGHT"

class Monitor_comp:
    MAJOR = "COMPONENT_MAJOR"
    MINOR = "COMPONENT_MINOR"

# class Circuits:
#     c = RSoftCircuit()
#     dimension_x = 0
#     dimension_y = 0
#     tap_ratio = 0
#     ind_file = str
#     out_file = str
#     eff_index = []
#     launch_field = ""
#     pathway = ""
#     monitor = ""

#     # launch field stuff
#     first_launch_field = True
#     first_launch_field_position = 0
#     first_launch_field_power = 0

#     def __init__(self, dim = int, dim_y = int, input_file_name = str, output_file_name = str, tap_rat = float):
#         self.ind_file = input_file_name + ".ind"
#         self.out_file = output_file_name 
#         self.tap_ratio = tap_rat
#         self.c.set_symbol("dimension", dim)
#         self.dimension_y = dim_y

#     def add_segments(self, pos=vec3d, offset=vec3d, dimension_x = float, dimension_y = float, index = float):
#         self.c.add_segment(position = (pos.x, pos.y, pos.z), offset = (offset.x, offset.y, offset.z),
#                            dimensions = (dimension_x/self.tap_ratio,self.dimension_y/self.tap_ratio),
#                            dimensions_end = (self.dimension_x,self.dimension_y)
#                            )
#         self.eff_index.append(index)

#     def add_pathways(self, segments = [str]):
#         for i in range(len(segments)):
#             self.pathway += ("\t" + segments[i] + "\n")
#         self.pathway += "/"
    
#     def add_monitors(self, pathway = int, mon_type = Monitor_Prop, tilt = str, mon_comp = Monitor_comp):
#         self.monitor += "pathway = " + str(pathway) + "\n\tmonitor_type = " + str(mon_type) + "\n\tmonitor_tilt = " + str(tilt) + "\n\tmonitor_component = " + str(mon_comp) + "\n/"
    
#     def add_launch_fields(self, x, power, launch_type, launch_pathway, launch_tilt):
#         # checks to see if this is the first launch field being added.
#         if self.first_launch_field:
#             self.first_launch_field = False
#             self.first_launch_field_position = x
#             self.first_launch_field_power = power
#         self.launch_field += "launch_pathway = " + str(launch_pathway) + "\n\tlaunch_type = " + str(launch_type) + "\n\tlaunch_tilt = " + str(launch_tilt) + "\n\tlaunch_position = " + str(x) + "\n\tlaunch_power = " + str(power) + "/"