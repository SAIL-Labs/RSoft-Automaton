class Monitor_Prop:
    FILE_POWER = "MONITOR_FILE_POWER"
    FILE_PHASE = "MONITOR_FILE_PHASE"
    FIBRE_MODE_POWER = "MONITOR_WGMODE_POWER"
    FIBRE_MODE_PHASE = "MONITOR_WGMODE_PHASE"
    GAUSS_POWER = "MONITOR_GAUSS_POWER"
    GAUSS_PHASE = "MONITOR_GAUSS_PHASE"
    LAUNCH_POWER = "MONITOR_LAUNCH_POWER"
    LAUNCH_PHASE = "MONITOR_LAUNCH_PHASE"
    PARTIAL_POWER = "MONITOR_WG_POWER"
    TOTAL_POWER = "MONITOR_TOTAL_POWER"
    EFF_IND = "MONITOR_FIELD_NEFF"
    FIELD_WIDTH = "MONITOR_FIELD_WIDTH"
    FIELD_HEIGHT = "MONITOR_FIELD_HEIGHT"

class Monitor_comp:
    BOTH = "COMPONENT_BOTH"
    MAJOR = "COMPONENT_MAJOR"
    MINOR = "COMPONENT_MINOR"

class Struct_type:
    FIBRE = "STRUCT_FIBER"

class Sim_tool:
    BP = "ST_BEAMPROP"

class LaunchType:
    LF = "LAUNCH_FILE"
    MM = "LAUNCH_MULTIMODE"
    SM = "LAUNCH_WGMODE"
    GAUSSIAN = "LAUNCH_GAUSSIAN"
    COMPUTE = "LAUNCH_COMPMODE"
    PLANEWAVE = "LAUNCH_PLANEWAVE"