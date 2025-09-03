import numpy as np, warnings, pandas as pd
##############################################################################
'''
Functions to generate a circular grid
'''
def generate_filled_circle_grid(radius, grid_spacing):
    coord = []
    dy = grid_spacing * np.sqrt(3) / 2  # similar vertical step as hex grids
    y = -radius
    row_num = 0

    while y <= radius:
        if abs(y) > radius:
            y += dy
            continue

        # Circle intersection x-bounds
        x_bound = np.sqrt(radius**2 - y**2)
        x_left = -x_bound
        x_right = x_bound

        # Place points between x_left and x_right
        num_points = int(np.floor((x_right - x_left) / grid_spacing)) 

        if num_points > 0:
            for i in range(num_points):
                x = x_left + i * (x_right - x_left) / (num_points - 1) if num_points > 1 else (x_left + x_right) / 2
                coord.append([row_num, x, y, 0])
            row_num += 1

        y += dy

    # Output
    hcoord = [c[1] for c in coord]
    vcoord = [c[2] for c in coord]
    return hcoord, vcoord

def estimate_circle_radius_with_autofit(n_points, grid_spacing, tolerance=0):
    """
    Estimate the circle radius and auto-correct it to match desired n_points within a tolerance.

    Args:
        n_points (int): desired number of points
        grid_spacing (float): spacing between points (μm)
        tolerance (int): how close the result must be

    Returns:
        radius (float): adjusted radius
    """
    # Initial estimate
    radius = grid_spacing * np.sqrt(n_points / np.pi)

    # Auto-refine
    max_iterations = 50
    for _ in range(max_iterations):
        hcoord, vcoord = generate_filled_circle_grid(radius, grid_spacing)
        actual_points = len(hcoord)

        if abs(actual_points - n_points) <= tolerance:
            print(f"Matched {actual_points} points (target {n_points}) within ±{tolerance}")
            return radius

        if actual_points < n_points:
            radius *= 1.02  # grow slightly
        else:
            radius *= 0.98  # shrink slightly

    print(f"Max iterations reached: got {actual_points} points for target {n_points}")
    return radius
##############################################################################
'''
Functions to generate a pentagon grid
'''
def generate_pentagon_grid(max_radius, core_spacing, core_number, include_center=True):
    """
    Generate a pentagon-based ring grid of points. Note this does not completely fill the grid, rather just points along the vertices.
    
    Parameters:
        max_radius (float): Maximum radial extent of the grid (e.g., cladding radius).
        core_spacing (float): Spacing between concentric pentagonal rings.
        core_number (int): Desired number of total cores (approximate).
        include_center (bool): Whether to include a center core.

    Returns:
        Xarrs, Yarrs: Lists of x and y coordinates.
    """
    theta = 72  
    Xarrs, Yarrs = [], []

    # Estimate max number of rings based on radius and spacing
    num_rings = int(max_radius // core_spacing)

    # Adjust number of rings if fewer cores are requested
    if include_center:
        rings_required = min(num_rings, max(1, (core_number - 1) // 5))
    else:
        rings_required = min(num_rings, max(1, core_number // 5))

    def gen_ring(radius):
        xs, ys = [], []
        for i in range(5):
            angle = i * theta - 54  # rotate flat side down
            x = radius * np.cos(np.deg2rad(angle))
            y = radius * np.sin(np.deg2rad(angle))
            xs.append(x)
            ys.append(y)
        return xs, ys

    for j in range(1, rings_required + 1):
        radius = j * core_spacing
        xs, ys = gen_ring(radius)
        Xarrs.extend(xs)
        Yarrs.extend(ys)

    if include_center:
        Xarrs.append(0)
        Yarrs.append(0)

    return Xarrs, Yarrs

def old_generate_filled_pentagon_grid(radius, grid_spacing, y_shift_factor = 0.5):
    """
    Generate a filled pentagon grid, row-by-row.
    
    Args:
        radius (float): Distance from center to pentagon vertices
        grid_spacing (float): Desired spacing between points

    Returns:
        hcoord, vcoord, df: X, Y coordinate lists and DataFrame
    """
    # Define pentagon vertices
    vertices = []
    starting_angle = -54  # so the base is flat
    vertex = 5
    for i in range(vertex):
        angle_deg = starting_angle + i * (360 / vertex)
        angle_rad = np.deg2rad(angle_deg)
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        vertices.append((x, y))
    vertices.append(vertices[0])  # Close the pentagon

    # Define all edges
    edges = []
    for i in range(vertex):
        x1, y1 = vertices[i]
        x2, y2 = vertices[i+1]
        edges.append(((x1, y1), (x2, y2)))

    # Create points row by row
    ymin = min(v[1] for v in vertices)
    ymax = max(v[1] for v in vertices)

    coord = []
    row_num = 0
    y = ymin
    dy = grid_spacing * np.sqrt(3) / 2

    while y <= ymax:
        intersections = []

        # Find intersections of this horizontal line with pentagon edges
        for (x1, y1), (x2, y2) in edges:
            if (y1 - y) * (y2 - y) <= 0 and y1 != y2:  # Check if y is between y1 and y2
                # Linearly interpolate x at this y
                x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                intersections.append(x)

        if len(intersections) >= 2:
            x_left, x_right = sorted(intersections)[:2]
            num_points = int(np.floor((x_right - x_left) / grid_spacing)) 
            if num_points > 0:
                for i in range(num_points):
                    x = x_left + i * (x_right - x_left) / (num_points - 1) if num_points > 1 else (x_left + x_right) / 2
                    coord.append([row_num, x, y, 0])
            else:
                # No points in this row — skip adding anything
                pass

        y += dy
        row_num += 1

    y_shift = y_shift_factor * grid_spacing
    for c in coord:
        c[2] -= y_shift

    hcoord = [c[1] for c in coord]
    vcoord = [c[2] for c in coord]
    return hcoord, vcoord

# def estimate_pentagon_radius(n_points, grid_spacing, tolerance = 0):
#     distance_per_point = grid_spacing ** 2
#     pentagon_area_factor = 5/2 * np.sin(2*np.pi/5)

#     radius_squared = (n_points * distance_per_point) / pentagon_area_factor
#     radius = np.sqrt(radius_squared)

#     # Due to the imperfect geometry of a pentagon grid need to include a loop that fixes the number of positions reported, 
#     # otherwise the desired number of positions will always be less than the actual number of positions
#     max_iterations = 50
#     for _ in range(max_iterations):
#         hcoord, vcoord= generate_filled_pentagon_grid(radius, grid_spacing)
#         actual_points = len(hcoord)

#         if abs(actual_points - n_points) <= tolerance:
#             print(f"Matched points: {actual_points} points (within ±{tolerance})")
#             return radius

#         # Adjust radius based on whether we have too many or too few points
#         if actual_points < n_points:
#             radius *= 1.02  # Slightly expand
#         else:
#             radius *= 0.98  # Slightly contract

#     print(f"Warning: maximum iterations reached. Final points = {actual_points}")
#     return radius
##############################################################################
def generate_hex_ring_grid(max_radius, core_spacing, core_number, include_center=True):
    """
    Generate a hexagon-based ring grid of points (concentric hexagons).
    
    Parameters:
        max_radius (float): Maximum radial extent of the grid (e.g., cladding radius).
        core_spacing (float): Spacing between concentric hexagonal rings.
        core_number (int): Desired number of total cores (approximate).
        include_center (bool): Whether to include a center core.

    Returns:
        Xarrs, Yarrs: Lists of x and y coordinates.
    """
    n_sides = 6
    theta = 360 / n_sides
    Xarrs, Yarrs = [], []

    # Estimate the number of rings
    num_rings = int(max_radius // core_spacing)
    if include_center:
        rings_required = min(num_rings, max(1, (core_number - 1) // n_sides))
    else:
        rings_required = min(num_rings, max(1, core_number // n_sides))

    def gen_ring(radius):
        xs, ys = [], []
        for i in range(n_sides):
            angle = i * theta  # rotate so flat side is down
            x = radius * np.cos(np.deg2rad(angle))
            y = radius * np.sin(np.deg2rad(angle))
            xs.append(x)
            ys.append(y)
        return xs, ys

    for j in range(1, rings_required + 1):
        radius = j * core_spacing
        xs, ys = gen_ring(radius)
        Xarrs.extend(xs)
        Yarrs.extend(ys)

    if include_center:
        Xarrs.append(0)
        Yarrs.append(0)

    return Xarrs, Yarrs
##############################################################################
'''
Functions to generate a hex grid
'''
def old_generate_hex_grid(row_num, grid_spacing, include_centre = True):
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
    
    # Remove the central core at (0,0) if include_centre is False
    if not include_centre:
        # Find indices of (0,0) point(s), which can occur for even/odd row_num
        filtered = [(x, y) for x, y in zip(hcoord, vcoord) if not (np.isclose(x, 0) and np.isclose(y, 0))]
        hcoord = [x for x, y in filtered]
        vcoord = [y for x, y in filtered]
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
##############################################################################
'''
Generate Square Grid
'''
def generate_square_grid(core_num, spacing):
    # Calculate grid size: smallest n such that n^2 >= core_num
    side = int(np.ceil(np.sqrt(core_num)))
    x_vals = []
    y_vals = []

    for i in range(side):
        for j in range(side):
            if len(x_vals) < core_num:
                x = (i - (side - 1) / 2) * spacing
                y = (j - (side - 1) / 2) * spacing
                x_vals.append(x)
                y_vals.append(y)
    
    return x_vals, y_vals
##############################################################################