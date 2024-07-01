from typing import Optional, Tuple
import torch
from domain.domain import Domain
from domain.utils import calculate_partial_derivatives, dump_points
from training.params import SimulationParameters

class MeshDomain(Domain):
    def __init__(self, mesh_filename: str, device=torch.device("cpu")) -> None:
        super().__init__()
        self.device = device
        self.params = SimulationParameters()
        self.x_raw, self.y_raw, self.z_raw = dump_points(mesh_filename)
        self.interior_points = self.get_interior_points()
        self.initial_points = self.get_initial_points(self.params.INITIAL_POINTS)
        self.boundary_points = self.get_boundary_points(self.params.BOUNDARY_POINTS)
        self.dzdx, self.dzdy = calculate_partial_derivatives(self.x_raw, self.y_raw, self.z_raw, device)

    def get_initial_points(self,
                           n_points: int = None,
                           requires_grad=True) -> Tuple[torch.Tensor,
                                                        torch.Tensor,
                                                        torch.Tensor]:
        """
        If no n_points is provided: then uses points from mesh and returns:
        - x_grid: [[x1], [x2], [x3], ...]
        - y_grid: [[y1], [y2], [y3], ...]
        - t0: [[0], [0], ...] -> number of points

        otherwise takes n_points and creates grids n_points x n_points:
        - x_grid: [[x1], [x2], [x3], ...] size: (n_points x n_points) x 1
        - y_grid: [[y1], [y2], [y3], ...] size: (n_points x n_points) x 1
        - t0: [[0], [0], ...] size: (n_points x n_points) x 1
        """
        if n_points:
            x_linspace, y_linspace, _ = self._generate_linespaces_n(n_points, requires_grad)
            x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
            x_grid = self._reshape_and_to_device(x_grid, requires_grad)
            y_grid = self._reshape_and_to_device(y_grid, requires_grad)
        else:
            x_grid = self._reshape_and_to_device(self.x_raw, requires_grad)
            y_grid = self._reshape_and_to_device(self.y_raw, requires_grad)

        t0 = torch.full_like(x_grid, self.params.T_DOMAIN[0], requires_grad=requires_grad)
        return (x_grid, y_grid, t0)

    def get_boundary_points(self, n_points=None, requires_grad=True):
        """
        Generates boundary points for the domain.

        Returns:
            Tuple of down, up, left, right boundary points.
        """
        x_linspace, y_linspace, t_linspace = self._generate_linespaces_n(n_points)

        x_grid, t_grid = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
        y_grid, _ = torch.meshgrid(y_linspace, t_linspace, indexing="ij")

        x_grid = self._reshape_and_to_device(x_grid, requires_grad)
        y_grid = self._reshape_and_to_device(y_grid, requires_grad)
        t_grid = self._reshape_and_to_device(t_grid, requires_grad)

        x0 = torch.full_like(t_grid, self.params.XY_DOMAIN[0], requires_grad=requires_grad)
        x1 = torch.full_like(t_grid, self.params.XY_DOMAIN[1], requires_grad=requires_grad)
        y0 = torch.full_like(t_grid, self.params.XY_DOMAIN[0], requires_grad=requires_grad)
        y1 = torch.full_like(t_grid, self.params.XY_DOMAIN[1], requires_grad=requires_grad)

        down = (x_grid, y0, t_grid)
        up = (x_grid, y1, t_grid)
        left = (x0, y_grid, t_grid)
        right = (x1, y_grid, t_grid)

        return down, up, left, right

    def get_interior_points(self, requires_grad=True):
        """
        Generates interior points for the domain.

        Returns:
            Tuple of x, y, z, t interior points.
        """
        # Vector with T_POINTS values, equally distributed in T_DOMAIN
        t_raw = torch.linspace(
            self.params.T_DOMAIN[0],
            self.params.T_DOMAIN[1],
            steps=self.params.T_POINTS, device=self.device)

        # x_grid: (n_points x T_POINTS) [[x1, x1, ...], [x2, x2, ...], ...]
        # y_grid: (n_points x T_POINTS) [[y1, y1, ...], [y2, y2, ...], ...]
        # z_grid: (n_points x T_POINTS) [[z1, z1, ...], [z2, z2, ...], ...]
        # t_grid: (n_points x T_POINTS) [[t1, t2, ...], [t1, t2, ...], ...]
        x_grid, t_grid = torch.meshgrid(self.x_raw.to(self.device), t_raw, indexing="ij")
        y_grid, _ = torch.meshgrid(self.y_raw.to(self.device), t_raw, indexing="ij")
        z_grid, _ = torch.meshgrid(self.z_raw.to(self.device), t_raw, indexing="ij")

        x = self._reshape_and_to_device(x_grid, requires_grad)
        y = self._reshape_and_to_device(y_grid, requires_grad)
        z = self._reshape_and_to_device(z_grid, requires_grad)
        t = self._reshape_and_to_device(t_grid, requires_grad)

        self.params.XY_DOMAIN = [x.min().item(), x.max().item()]
        self.params.INTERIOR_POINTS = x.size()[0] // self.params.T_POINTS

        # sizes: (n_points x T_POINTS) x 1, e.g. [[x1], [x1], ..., [x2], [x2], ...]
        return x, y, z, t

    def _reshape_and_to_device(self, tensor: torch.Tensor, requires_grad: Optional[bool] = None) -> torch.Tensor:
        """
        Reshapes the tensor to a 2D tensor and moves it to the specified device.
        """
        tensor = tensor.reshape(-1, 1).to(self.device)
        if requires_grad is not None:
            tensor.requires_grad = requires_grad
        return tensor

    def _generate_linespaces_n(self,
                               n_points: int,
                               requires_grad=False) -> Tuple[torch.Tensor,
                                                             torch.Tensor,
                                                             torch.Tensor]:
        """
        Generates linearly spaced points for the domain.

        Returns:
            Tuple of x, y, t linspaces.
        """
        n_points_linspace = n_points if n_points else self.params.INTERIOR_POINTS

        x_linspace = torch.linspace(
            self.params.XY_DOMAIN[0],
            self.params.XY_DOMAIN[1],
            steps=n_points_linspace,
            device=self.device,
            requires_grad=requires_grad)
        y_linspace = torch.linspace(
            self.params.XY_DOMAIN[0],
            self.params.XY_DOMAIN[1],
            steps=n_points_linspace,
            device=self.device,
            requires_grad=requires_grad)
        t_linspace = torch.linspace(
            self.params.T_DOMAIN[0],
            self.params.T_DOMAIN[1],
            steps=self.params.T_POINTS,
            device=self.device,
            requires_grad=requires_grad)

        return x_linspace, y_linspace, t_linspace
