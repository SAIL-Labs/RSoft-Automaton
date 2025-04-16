import numpy as np, warnings

def generate_hex_grid(row_num, grid_spacing):
    coord = []
    dx = grid_spacing
    dy = np.sqrt(3) * grid_spacing / 2
    mid_index = row_num // 2

    for row in range(row_num):
        row_offset = row - mid_index
        y = row_offset * dy
        points_in_row = row_num - abs(row_offset)

        for col in range(points_in_row):
            x_offset = col - (points_in_row - 1) / 2
            x = x_offset * dx
            coord.append([row, x, y, 0])

    hcoord = [c[1] for c in coord]
    vcoord = [c[2] for c in coord]
    return hcoord, vcoord

def number_rows(n_points):
    r = 0
    while True:
        # the total number of points that can fit within a hexagon = 1 + 3r(r+1)
        # add exception here when the number of cores cannot be placed neatly, idk
        total = 1 + 3 * r * (r+1)
        if total >= n_points:
            if total != n_points:
                warnings.warn(f"Warning: the requested hexagonal structure supports {total} cores, but {n_points} have been provided. \n Total number of unused cores: {total - n_points}")
            return 2*r+1
        r+=1