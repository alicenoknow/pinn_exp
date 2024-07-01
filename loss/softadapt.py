import torch
from typing import Callable
from domain.domain import Domain
from equations.wave import dfdx, dfdy, f
from loss.loss import Loss
from softadapt import LossWeightedSoftAdapt

from pinn.pinn import PINN
from training.params import SimulationParameters

class SoftAdaptLoss(Loss):
    """
    SoftAdapt
    pip install git+https://github.com/dr-aheydari/SoftAdapt.git
    """

    def __init__(
        self,
        domain: Domain,
        initial_condition: Callable,
        wave_equation: Callable,
        beta: float = 0.1
    ):
        self.domain = domain
        self.wave_equation = wave_equation
        self.initial_condition = initial_condition

        self.params = SimulationParameters()

        self.softadapt_object = LossWeightedSoftAdapt(beta=beta)
        self.epochs_to_make_updates = 5
        self.initial_history = []
        self.residual_history = []
        self.boundary_history = []
        self.adaptive_weights = torch.tensor([self.params.INITIAL_WEIGHT_INITIAL,
                                              self.params.INITIAL_WEIGHT_RESIDUAL,
                                              self.params.INITIAL_WEIGHT_BOUNDARY])
        self.epoch = 0

    def residual_loss(self, pinn: PINN):
        x, y, z, t = self.domain.interior_points
        dzdx, dzdy = self.domain.dzdx, self.domain.dzdy
        loss = self.wave_equation(pinn, x, y, z, t, dzdx, dzdy)

        return loss.pow(2).mean()

    def initial_loss(self, pinn: PINN):
        x, y, t = self.domain.initial_points

        length = SimulationParameters().XY_DOMAIN[1]
        pinn_init = self.initial_condition(x, y, length)
        loss = f(pinn, x, y, t) - pinn_init

        return loss.pow(2).mean()

    def boundary_loss(self, pinn: PINN):
        down, up, left, right = self.environment.boundary_points

        x_down, y_down, t_down = down
        x_up, y_up, t_up = up
        x_left, y_left, t_left = left
        x_right, y_right, t_right = right

        loss_down = dfdy(pinn, x_down, y_down, t_down)
        loss_up = dfdy(pinn, x_up, y_up, t_up)
        loss_left = dfdx(pinn, x_left, y_left, t_left)
        loss_right = dfdx(pinn, x_right, y_right, t_right)

        return loss_down.pow(2).mean() + \
            loss_up.pow(2).mean() + \
            loss_left.pow(2).mean() + \
            loss_right.pow(2).mean()

    def verbose(self, pinn: PINN):
        """
        Returns all parts of the loss function

        Not used during training! Only for checking the results later.
        """
        residual_loss = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn)
        boundary_loss = self.boundary_loss(pinn)

        self.initial_history.append(initial_loss)
        self.residual_history.append(residual_loss)
        self.boundary_history.append(boundary_loss)

        if self.epoch % self.epochs_to_make_updates == 0 and self.epoch != 0:
            self.adaptive_weights = self.softadapt_object.get_component_weights(
                torch.tensor(
                    self.initial_history), torch.tensor(
                    self.residual_history), torch.tensor(
                    self.boundary_history), verbose=False)
            self.initial_history = []
            self.residual_history = []
            self.boundary_history = []

        final_loss = self.adaptive_weights[0] * initial_loss \
            + self.adaptive_weights[1] * residual_loss \
            + self.adaptive_weights[2] * boundary_loss

        self.epoch += 1

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
