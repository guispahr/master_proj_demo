import numpy as np
import xml.etree.ElementTree as ET
from jax import jit, vmap
import jax.numpy as jnp

def rot_x(o):
    return jnp.array([
        [1, 0, 0],
        [0, jnp.cos(o), jnp.sin(o)],
        [0, -jnp.sin(o), jnp.cos(o)]])


def rot_y(p):
    return jnp.array([
        [jnp.cos(p), 0, -jnp.sin(p)],
        [0, 1, 0],
        [jnp.sin(p), 0, jnp.cos(p)]])


def rot_z(k):
    return jnp.array([
        [jnp.cos(k), jnp.sin(k), 0],
        [-jnp.sin(k), jnp.cos(k), 0],
        [0, 0, 1]])


def rot_zyx(o, p, k):
    return rot_z(k) @ rot_y(p) @ rot_x(o)

def rms_dict(_s):
    R = rot_zyx(_s['omega'], _s['phi'], _s['kappa']) # rotation matrix
    M = jnp.array([_s['X'], _s['Y'], _s['Z']])       # world coordinates (points)
    S = jnp.array([_s['Xs'], _s['Ys'], _s['Zs']])    # camera position
    RMS = R @ (M - S)                                # transformed into camera coordinates
    return RMS

# Projection onto image plane
def xy_frame(_s):
    RMS = rms_dict(_s)
    m = - RMS / RMS[2]
    x, y, _ = m
    z = -RMS[2]
    return x, y, z

# For the camera distortion correction
def corr_dist_agi(x, y, _s):
    rc = x ** 2 + y ** 2
    dr = 1 + _s['k1'] * rc + _s['k2'] * rc ** 2 + _s['k3'] * rc ** 3 + _s['k4'] * rc ** 4 + _s['k5'] * rc ** 5
    drx = x * dr
    dry = y * dr
    #
    dtx = _s['p1'] * (rc + 2 * x ** 2) + 2 * _s['p2'] * x * y * (1 + _s['p3'] * rc + _s['p4'] * rc ** 2)
    dty = _s['p2'] * (rc + 2 * y ** 2) + 2 * _s['p1'] * x * y * (1 + _s['p3'] * rc + _s['p4'] * rc ** 2)
    xp = drx + dtx
    yp = dry + dty
    #
    fx = _s['width'] * 0.5 + _s['cx'] + xp * _s['f'] + xp * _s['b1'] + yp * _s['b2']
    fy = _s['height'] * 0.5 + _s['cy'] + yp * _s['f']
    return fx, fy

def f_frame_agi(_s):
    x, y, z = xy_frame(_s)
    w_2 = _s['width'] / _s['f'] / 2
    h_2 = _s['height'] / _s['f'] / 2
    ins = (x >= -w_2) & (x < w_2) & (y >= -h_2) & (y < h_2) & (z > 0)
    y = -y # to match agisoft convention
    fx, fy = corr_dist_agi(x, y, _s)
    return fx, fy, z, ins

def parse_calibration_xml(file_path):
    """
    Parses Agisoft camera_calibration.xml.
    Returns: width, height, f, cx, cy, k1, k2, k3, k4, k5, 
             p1, p2, p3, p4, b1, b2
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize
    calibration_data = {
        #'projection': 'frame',
        'width': 0,
        'height': 0,
        'f': 0.0,
        'cx': 0.0,
        'cy': 0.0,
        'k1': 0.0,
        'k2': 0.0,
        'k3': 0.0,
        'k4': 0.0,
        'k5': 0.0,
        'p1': 0.0,
        'p2': 0.0,
        'p3': 0.0,
        'p4': 0.0,
        'b1': 0.0,
        'b2': 0.0,
        #'date': ''  # empty
    }

    for element in root:
        if element.tag in calibration_data:
            if element.tag in ['projection', 'date']:
                continue
            elif element.tag in ['width', 'height']:
                calibration_data[element.tag] = int(element.text)
            else:
                calibration_data[element.tag] = float(element.text)

    return calibration_data


def read_camera_file(filepath, offset):
    data = {}
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            values = line.strip().split()
            if len(values) < 16:
                continue
            photo_id = values[0]
            data[photo_id] = {
                "Xs": float(values[1]) - offset[0],
                "Ys": float(values[2]) - offset[1],
                "Zs": float(values[3]) - offset[2],
                "omega": np.radians(float(values[4])),
                "phi": np.radians(float(values[5])),
                "kappa": np.radians(float(values[6])),
            }
    return data

def get_pixel_values(image, i, j, in_bounds, nb_classes=10):
    """
    Retrieve pixel values from `image` at specified `coords`, marking out-of-bounds
    coordinates with `jnp.nan`. Uses only `jax.numpy` operations.
    Parameters:
        image (jnp.ndarray): 2D image array.
        fx, fy (jnp.ndarray): 2 arrays of shape (N, 1) representing i and j coordinates.
    Returns:
        jnp.ndarray: Array of pixel values with `jnp.nan` for out-of-bounds coordinates.
    """
    # Initialize pixel values with NaNs
    pixel_values = jnp.full((i.shape[0], nb_classes), jnp.nan)
    values = image[j[in_bounds], i[in_bounds]]
    pixel_values = pixel_values.at[in_bounds].set(image[j[in_bounds], i[in_bounds]])
    pixel_values = pixel_values.at[in_bounds].set(values)
    return pixel_values


def compute_depth_map(i, j, z, depth_map, buffer_size, threshold=0.05):
    height, width = depth_map.shape
    # Define buffer
    offsets = jnp.arange(-buffer_size, buffer_size + 1)  # Range from -3 to 3
    # Create all combinations of offsets to form a square neighborhood
    di, dj = jnp.meshgrid(offsets, offsets, indexing='ij')  # du, dv are (7, 7) arrays
    di = di.ravel()  # Flatten to 1D
    dj = dj.ravel()
    # Compute the neighborhood coordinates for each point
    neighbor_i = (i[:, None] + di).clip(0, width - 1)  # (N, 49), clipped to image bounds
    neighbor_j = (j[:, None] + dj).clip(0, height - 1)  # (N, 49), clipped to image bounds
    neighbor_depths = jnp.repeat(z[:, None], len(di), axis=1)  # Repeat depths to match neighborhood shape
    # Flatten everything for efficient indexing
    neighbor_i = neighbor_i.ravel()
    neighbor_j = neighbor_j.ravel()
    neighbor_depths = neighbor_depths.ravel()
    # Use scatter_min to update depth_image efficiently
    depth_map = depth_map.at[neighbor_j, neighbor_i].min(neighbor_depths)
    # Compute visibility map by comparing depth_map with z value
    visibility = jnp.abs(depth_map[j, i] - z) <= threshold
    return depth_map, visibility