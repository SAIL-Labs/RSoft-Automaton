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
def old_generate_pentagon_grid(max_radius, core_spacing, core_number, include_center=True):
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


"Stuff to do with pent grids"
def generate_pent_grid(row_num, grid_spacing, include_centre=True):
    """
    Generate coordinates on a pentagonal grid (built from concentric pentagon rings).

    Parameters
    ----------
    row_num : int
        Number of 'rows' (2*r + 1) obtained from number_rows(..., grid_type='pent').
        Internally determines how many rings of the pentagon are drawn.
    grid_spacing : float
        Spacing between consecutive rings.
    include_centre : bool, optional
        Whether to include the centre coordinate (0, 0). Default is True.

    Returns
    -------
    x, y : np.ndarray
        Arrays of x and y coordinates for each point in the pentagonal grid.
    """

    # infer number of rings r from number of rows
    r = (row_num - 1) // 2

    coords = []

    # include the central point if desired
    if include_centre:
        coords.append((0.0, 0.0))

    # generate ring by ring
    for ring in range(1, r + 1):
        # each ring has 5 * ring points (5 sides × ring subdivisions)
        n_side = ring  # number of divisions per side
        side_points = 5 * n_side
        radius = ring * grid_spacing

        # compute pentagon vertices for this radius
        # regular pentagon, vertex angle spacing = 72°
        angles = np.deg2rad(np.linspace(90, 450, 6))  # 6 so last vertex == first for closure
        vertices = np.column_stack((radius * np.cos(angles),
                                    radius * np.sin(angles)))

        # interpolate along each of the 5 edges
        for i in range(5):
            start = vertices[i]
            end = vertices[i + 1]
            # place `n_side` points per edge (excluding endpoint to avoid duplicates)
            for j in range(n_side):
                t = j / n_side
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                coords.append((x, y))

    coords = np.array(coords)
    x, y = coords[:, 0], coords[:, 1]
    return x, y
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
########################################################################
"Stuff to do with hex grids"
def old_generate_hex_grid(row_num, grid_spacing, include_centre = True):
    """
    Function to generate coordinates on a hexagonal grid.
    Parameters:
        - row_num: integer determining how many rows are plotted. This is determined from using the 
        number_rows() function,
        - grid_spaceing: float that determines the spacing between coordinates,
        include_centre: bool that determines if the centrer coordinate is plotted or not. True: the centre 
        coordinate is plotted, False: it is not.
    Returns:
        - Returns x and y coordinates as separate numpy arrays.
    """
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
        hcoord = [x for x, _ in filtered]
        vcoord = [y for _, y in filtered]
    return np.array(hcoord), np.array(vcoord)

########################################################################
"Number of rows needed for both grid types"
def number_rows(n_points, grid_type="hex"):
    """
    Determine how many 'rows' (i.e. rings → rows = 2*r+1) are needed to place
    `n_points` in a regular polygonal grid.

    Parameters
    ----------
    n_points : int
        number of cores / points requested
    grid_type : str
        "hex" for hexagonal (default),
        "pent" for pentagonal-style rings

    Returns
    -------
    n_rows : int
        2*r + 1, where r is the number of rings around the centre
    excess : int
        how many 'slots' in that pattern are left unused
    """
    r = 0
    while True:
        if grid_type.lower() == "hex":
            # hex pattern: 1 + 3 r (r+1)
            total = 1 + 3 * r * (r + 1)
        elif grid_type.lower() == "pent":
            # pent pattern: 1 + 5/2 * r (r+1)
            total = 1 + (5 * r * (r + 1)) // 2   # integer version
            # if you prefer exact math, do:
            # total = 1 + 5 * r * (r + 1) / 2
        else:
            raise ValueError(f"Unknown grid_type: {grid_type}")

        if total >= n_points:
            if total != n_points:
                excess = total - n_points
                warnings.warn(
                    f"Warning: the requested {grid_type} structure supports {total} cores, "
                    f"but {n_points} have been provided.\n"
                    f"Total number of unused cores: {excess}"
                )
            else:
                excess = 0
            return 2 * r + 1, excess

        r += 1

def plot_excess(excess, xcoord, ycoord, reorder_indices):
    if excess > 0:
        xcoord_og, ycoord_og = xcoord, ycoord
        xcoord_relist, ycoord_relist = xcoord_og[reorder_indices], ycoord_og[reorder_indices]
        xcoord_relist, ycoord_relist = xcoord_relist[:-excess], ycoord_relist[:-excess]

    else:
        xcoord_og, ycoord_og = xcoord, ycoord
        xcoord_relist, ycoord_relist = xcoord_og[reorder_indices], ycoord_og[reorder_indices]
    
    return xcoord_og, ycoord_og, xcoord_relist, ycoord_relist

def old_number_rows(n_points):
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