from typing import Callable
import torch
import torch.nn.functional as F
from domain.domain import Domain
from equations.wave import dfdx, dfdy, f
from loss.loss import Loss
from pinn.pinn import PINN
from training.params import SimulationParameters


class ReloLoss(Loss):
    """
    Multi-Objective Loss Balancing for Physics-Informed Deep Learning
    https://github.com/rbischof/relative_balancing
    """

    def __init__(
        self,
        domain: Domain,
        initial_condition: Callable,
        wave_equation: Callable,
        alpha=0.999, temperature=0.1, rho=0.99
    ):
        self.domain = domain
        self.wave_equation = wave_equation
        self.initial_condition = initial_condition
        self.params = SimulationParameters()

        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        self.call_count = torch.tensor(0, requires_grad=False, dtype=torch.int16)

        self.lambdas = [torch.tensor(self.params.INITIAL_WEIGHT_INITIAL, requires_grad=False),
                        torch.tensor(self.params.INITIAL_WEIGHT_RESIDUAL, requires_grad=False),
                        torch.tensor(self.params.INITIAL_WEIGHT_BOUNDARY, requires_grad=False)]
        self.last_losses = [torch.tensor(1., requires_grad=False) for _ in range(3)]
        self.init_losses = [torch.tensor(1., requires_grad=False) for _ in range(3)]

    def residual_loss(self, pinn: PINN):
        x, y, z, t = self.domain.interior_points
        dzdx, dzdy = self.domain.dzdx, self.domain.dzdy
        loss = self.wave_equation(pinn, x, y, z, t, dzdx, dzdy)
        return loss.pow(2).mean()

    def initial_loss(self, pinn: PINN):
        x, y, t = self.domain.initial_points
        length = self.domain.domain.XY_DOMAIN[1]
        pinn_init = self.initial_condition(x, y, length)
        loss = f(pinn, x, y, t) - pinn_init
        return loss.pow(2).mean()

    def boundary_loss(self, pinn: PINN):
        down, up, left, right = self.domain.boundary_points

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
        losses = [residual_loss, initial_loss, boundary_loss]

        alpha = torch.where(self.call_count == 0,
                            torch.tensor(1.),
                            torch.where(self.call_count == 1,
                                        torch.tensor(0.),
                                        torch.tensor(self.alpha)))
        rho = torch.where(self.call_count == 0,
                          torch.tensor(1.),
                          torch.where(self.call_count == 1,
                                      torch.tensor(1.),
                                      (torch.rand(()) < self.rho).to(torch.float32)))

        EPS = 1e-5
        # compute new lambdas w.r.t. the losses in the previous iteration
        lambdas_hat = [losses[i] / (self.last_losses[i] * self.temperature + EPS)
                       for i in range(len(losses))]
        lambdas_hat = F.softmax(torch.tensor(lambdas_hat) -
                                torch.max(torch.tensor(lambdas_hat)), dim=0) * len(losses)

        # compute new lambdas w.r.t. the losses in the first iteration
        init_lambdas_hat = [losses[i] /
                            (self.init_losses[i] *
                             self.temperature +
                             EPS) for i in range(len(losses))]
        init_lambdas_hat = F.softmax(torch.tensor(init_lambdas_hat) -
                                     torch.max(torch.tensor(init_lambdas_hat)), dim=0) * len(losses)

        # use rho for deciding, whether a random look back should be performed
        new_lambdas = [(rho * alpha * self.lambdas[i]
                        + (1 - rho) * alpha * init_lambdas_hat[i]
                        + (1 - alpha) * lambdas_hat[i])
                       for i in range(len(losses))]
        self.lambdas = [lam.clone().detach().requires_grad_(False) for lam in new_lambdas]

        # compute weighted loss
        loss = torch.sum(torch.stack([lam * loss for lam, loss in zip(self.lambdas, losses)]))

        # store current losses in self.last_losses to be accessed in the next iteration
        self.last_losses = [loss.clone().detach().requires_grad_(False) for loss in losses]

        # in first iteration, store losses in self.init_losses to be accessed in next iterations
        first_iteration = (self.call_count < 1).to(torch.float32)
        self.init_losses = [(loss * first_iteration + init_loss * (1 - first_iteration)).clone(
        ).detach().requires_grad_(False) for init_loss, loss in zip(self.init_losses, losses)]

        self.call_count += 1

        return loss, residual_loss, initial_loss, boundary_loss

    def __call__(self, pinn: PINN):
        """
        Allows you to use the instance of this class as if it were a function:

        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)
