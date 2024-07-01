from typing import Callable
import torch
from domain.domain import Domain
from equations.wave import dfdx, dfdy, f
from loss.loss import Loss
from pinn.pinn import PINN
from training.params import SimulationParameters

class WeightedLoss(Loss):
    def __init__(
        self,
        domain: Domain,
        initial_condition: Callable,
        wave_equation: Callable,
    ):
        self.domain = domain
        self.wave_equation = wave_equation
        self.initial_condition = initial_condition

        self.params = SimulationParameters()

    def residual_loss(self, pinn: PINN):
        """
        Calculates the residual loss for the given PINN.

        The residual loss is calculated as
        the mean squared error of the wave equation's output
        at the interior points.

        Args:
            pinn (PINN): The PINN for which to calculate the residual loss.

        Returns:
            torch.Tensor: The calculated residual loss.
        """
        x, y, z, t = self.domain.interior_points
        dzdx, dzdy = self.domain.dzdx, self.domain.dzdy
        loss = self.wave_equation(pinn, x, y, z, t, dzdx, dzdy)

        return loss.pow(2).mean()

    def initial_loss(self, pinn: PINN):
        """
        Calculates the initial loss for the given PINN.

        The initial loss is calculated as the difference between
        the PINN's output and the initial condition
        at the initial points.

        Args:
            pinn (PINN): The PINN for which to calculate the initial loss.

        Returns:
            torch.Tensor: The calculated initial loss.
        """
        x, y, t = self.domain.initial_points
        length = SimulationParameters().XY_DOMAIN[1]
        pinn_initial = self.initial_condition(x, y, length)

        return (f(pinn, x, y, t) - pinn_initial).pow(2).mean()

    def boundary_loss(self, pinn: PINN):
        """
        Calculates the boundary loss for the given PINN.

        The boundary loss is calculated as
        the mean squared error of the derivatives of the PINN's output
        with respect to x and y at the boundary points.

        Args:
            pinn (PINN): The PINN for which to calculate the boundary loss.

        Returns:
        torch.Tensor: The calculated boundary loss.
        """
        down, up, left, right = self.domain.boundary_points

        x_down, y_down, t_down = down
        x_up, y_up, t_up = up
        x_left, y_left, t_left = left
        x_right, y_right, t_right = right

        loss_down = dfdy(pinn, x_down, y_down, t_down)
        loss_up = dfdy(pinn, x_up, y_up, t_up)
        loss_left = dfdx(pinn, x_left, y_left, t_left)
        loss_right = dfdx(pinn, x_right, y_right, t_right)

        return sum(map(torch.Tensor.mean,
                       (loss_down.pow(2),
                        loss_up.pow(2),
                        loss_left.pow(2),
                        loss_right.pow(2))))

    def verbose(self, pinn: PINN):
        """
        Returns all parts of the loss function for the given PINN.

        This method is not used during training, only for checking the results later.

        Args:
            pinn (PINN): The PINN for which to calculate the loss components.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: The calculated residual loss and initial loss.
        """
        residual_loss = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn)
        boundary_loss = self.boundary_loss(pinn)

        final_loss = \
            self.params.INITIAL_WEIGHT_RESIDUAL * residual_loss + \
            self.params.INITIAL_WEIGHT_INITIAL * initial_loss + \
            self.params.INITIAL_WEIGHT_BOUNDARY * boundary_loss

        return final_loss, residual_loss, initial_loss, boundary_loss

    def __call__(self, pinn: PINN):
        """
        Allows you to use the instance of this class as if it were a function:

        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)
