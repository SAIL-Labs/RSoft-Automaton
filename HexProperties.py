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

def generate_filled_pentagon_grid(radius, grid_spacing):
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

    hcoord = [c[1] for c in coord]
    vcoord = [c[2] for c in coord]
    return hcoord, vcoord

def estimate_pentagon_radius(n_points, grid_spacing, tolerance = 0):
    distance_per_point = grid_spacing ** 2
    pentagon_area_factor = 5/2 * np.sin(2*np.pi/5)

    radius_squared = (n_points * distance_per_point) / pentagon_area_factor
    radius = np.sqrt(radius_squared)

    # Due to the imperfect geometry of a pentagon grid need to include a loop that fixes the number of positions reported, 
    # otherwise the desired number of positions will always be less than the actual number of positions
    max_iterations = 50
    for _ in range(max_iterations):
        hcoord, vcoord= generate_filled_pentagon_grid(radius, grid_spacing)
        actual_points = len(hcoord)

        if abs(actual_points - n_points) <= tolerance:
            print(f"Matched points: {actual_points} points (within ±{tolerance})")
            return radius

        # Adjust radius based on whether we have too many or too few points
        if actual_points < n_points:
            radius *= 1.02  # Slightly expand
        else:
            radius *= 0.98  # Slightly contract

    print(f"Warning: maximum iterations reached. Final points = {actual_points}")
    return radius
##############################################################################
'''
Functions to generate a hex grid
'''
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
##############################################################################