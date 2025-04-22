# Function to extract parameters from the .ind file
def Extract_params(param=""):
    with open("MCF_Test.ind", "r") as r:
        for line in r:
            if param in line:
                key, value = line.split("=")
                key = key.strip()
                val = float(value.strip())
                break # <-- stop reading the file once found
    return val
#################################################################################
# def insert_param_into_file(file_path, param):
#     with open(file_path, "r") as f:
#         lines = f.readlines()

#     new_lines = []
#     for i, line in enumerate(lines):
#         new_lines.append(line)
#         if line.strip().startswith("begin.width"):
#             new_lines.append(f"\tbegin.delta = {param}\n")
#         elif line.strip().startswith("end.width"):
#             new_lines.append(f"\tend.delta = {param}\n")

#     with open(file_path, "w") as f:
#         f.writelines(new_lines)
#################################################################################
def has_another_line(file):
    current_pos = file.tell()
    read_line = bool(file.readline())
    file.seek(current_pos)
    return read_line
#################################################################################
def AddHack(file_name, core_num,background_index):
    block_text = { 
        "pathway":'''
        pathway {n}
                {n}
        end pathway
        ''',
        "monitor": '''
        monitor {n}
                pathway = {n}
                monitor_type = MONITOR_WGMODE_POWER
                monitor_tilt = 1
                monitor_component = COMPONENT_BOTH
                monitor_background_index = {background_index}
                monitor_width = 9.0
                monitor_height = 9.0
        end monitor
        ''',
        "launch_field":'''
        launch_field {n}
                launch_pathway = {n}
                launch_type = LAUNCH_WGMODE
                launch_tilt = 1
                launch_align_file = 1
        end launch_field
        '''
    }

    # appends the above text to the .ind file
    with open(f"{file_name}.ind", "a") as f:
        for block_type in ["pathway","monitor","launch_field"]:
            for i in range(1,int(core_num) + 1):

                if block_type == "launch_field" and i != 1:
                    continue

                text = block_text[block_type].format(n=i, background_index = background_index)
                f.write(text)
