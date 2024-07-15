import meshio
import numpy as np
import torch

from scipy.interpolate import griddata


def dump_points(filename: str):
    """
    Reads triangular mesh from file.

    Values are normalized with respect to x, to keep relation between dimensions.
    Flag z_relative_to_x determines whether z is scaled to (0, 1) separately or with respect to x.

    Returns vectors x, y, z where (x[i], y[i], z[i]) was the original point in mesh.
    """
    mesh = meshio.avsucd.read(filename)
    points = torch.tensor(mesh.points, dtype=torch.float32)

    x, y, z = points.transpose(0, 1)

    min_x, min_z = torch.min(x), torch.min(z)
    max_x, max_z = torch.max(x), torch.max(z)

    x = (x - min_x) / (max_x - min_x)
    y = (y - min_x) / (max_x - min_x)
    z = (z - min_z) / (max_z - min_z)

    return x, y, z


def interpolate_plane(x, y, z):
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    points = np.vstack((x.numpy(), y.numpy())).T
    values = z.numpy()

    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    if np.any(np.isnan(grid_z)):
        grid_z = griddata(points, values, (grid_x, grid_y), method='nearest')

    return grid_x, grid_y, grid_z


def calculate_derivatives(grid_x, grid_y, grid_z):
    dz_dx, dz_dy = np.gradient(grid_z, grid_x[1, 0] - grid_x[0, 0], grid_y[0, 1] - grid_y[0, 0])
    return dz_dx, dz_dy


def interpolate_derivatives_to_mesh_points(x, y, grid_x, grid_y, dz_dx, dz_dy):
    points = np.vstack((x.numpy(), y.numpy())).T
    dz_dx_mesh = griddata((grid_x.flatten(), grid_y.flatten()),
                          dz_dx.flatten(), points, method='cubic')
    dz_dy_mesh = griddata((grid_x.flatten(), grid_y.flatten()),
                          dz_dy.flatten(), points, method='cubic')

    dz_dx_tensor = torch.tensor(dz_dx_mesh, dtype=torch.float32)
    dz_dy_tensor = torch.tensor(dz_dy_mesh, dtype=torch.float32)

    return dz_dx_tensor, dz_dy_tensor


def calculate_partial_derivatives(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, device):
    grid_x, grid_y, grid_z = interpolate_plane(x, y, z)
    dz_dx, dz_dy = calculate_derivatives(grid_x, grid_y, grid_z)
    dz_dx_tensor, dz_dy_tensor = interpolate_derivatives_to_mesh_points(
        x, y, grid_x, grid_y, dz_dx, dz_dy)
    return dz_dx_tensor.to(device), dz_dy_tensor.to(device)
