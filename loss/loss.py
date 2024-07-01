from abc import ABC, abstractmethod
from typing import Callable, Tuple
import torch
from domain.domain import Domain
from pinn.pinn import PINN
from training.params import SimulationParameters


class Loss(ABC):
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

    @abstractmethod
    def residual_loss(self, pinn: PINN) -> torch.Tensor:
        """
        Calculates the residual loss for the given PINN.

        Args:
            pinn (PINN): The PINN for which to calculate the residual loss.

        Returns:
            torch.Tensor: The calculated residual loss.
        """
        pass

    @abstractmethod
    def initial_loss(self, pinn: PINN) -> torch.Tensor:
        """
        Calculates the initial loss for the given PINN.

        Args:
            pinn (PINN): The PINN for which to calculate the initial loss.

        Returns:
            torch.Tensor: The calculated initial loss.
        """
        pass
    
    @abstractmethod
    def boundary_loss(self, pinn: PINN) -> torch.Tensor:
        """
        Calculates the boundary loss for the given PINN.

        Returns:
        torch.Tensor: The calculated boundary loss.
        """
        pass

    @abstractmethod
    def verbose(self, pinn: PINN) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns all parts of the loss function for the given PINN.

        This method is not used during training, only for checking the results later.

        Args:
            pinn (PINN): The PINN for which to calculate the loss components.

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Total loss and its components.
        """
        pass

    def __call__(self, pinn: PINN):
        """
        Allows you to use the instance of this class as if it were a function:

        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)
