

import logging
import os
from typing import Callable

import torch
from domain.domain import Domain
from pinn.pinn import PINN
from training.params import SimulationParameters
import numpy as np


def save_all(save_path: str,
             pinn: PINN,
             domain: Domain,
             initial_condition: Callable):
    path = os.path.join(save_path, "saved")
    os.makedirs(path, exist_ok=True)

    logging.info("Saving initial condition results")
    save_initial_condition(path, domain, pinn, initial_condition)

    logging.info("Saving simulation frames")
    save_frame(path, pinn, domain)


def save_initial_condition(save_path: str, domain: Domain, pinn: PINN, initial_condition: Callable):
    n_points_plot = SimulationParameters().POINTS_PLOT
    length = SimulationParameters().XY_DOMAIN[1]

    x, y, t = domain.get_initial_points(n_points_plot, requires_grad=False)
    z = initial_condition(x, y, length)
    np.save(os.path.join(save_path, "true_initial.npy"), z.detach().cpu().numpy())

    z = pinn(x, y, t)
    np.save(os.path.join(save_path, "pred_initial.npy"), z.detach().cpu().numpy())

def save_frame(save_path: str, pinn: PINN, domain: Domain, time_step: float = 0.01):
    t_max = SimulationParameters().T_DOMAIN[1]
    n_points_plot = SimulationParameters().POINTS_PLOT
    time_values = np.arange(0, t_max, time_step)

    for idx, t_value in enumerate(time_values):
        x, y, t = domain.get_initial_points(n_points_plot, requires_grad=False)
        t = torch.full_like(x, t_value)
        z = pinn(x, y, t)
        np.save(os.path.join(save_path, f"data_{idx}.npy"), z.detach().cpu().numpy())
