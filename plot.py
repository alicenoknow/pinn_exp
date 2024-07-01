import argparse
import logging
import sys
import os
import torch
from datetime import datetime

from domain.mesh_domain import MeshDomain
from domain.simple_domain import SimpleDomain
from equations.initial import make_initial_condition
from equations.wave import wave_equation, wave_equation_simplified
from loss.relo import ReloLoss
from loss.softadapt import SoftAdaptLoss
from loss.weighted import WeightedLoss
from pinn.pinn import PINN
from training.params import LossFunction, SimulationParameters
from training.train import Training
from visualization.plotting import plot_all_from_file

"""
Takes config file name.
Individual fields can be overridden by CLI arguments.
e.g. run.py --config base.json --layers 8
"""

logger = logging.getLogger()


def setup_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to data')

    args = parser.parse_args()
    params.set_json(os.path.join(args.path, 'config.json'))


def setup_logger(params):
    log_format = '[%(levelname)s] %(message)s'
    log_dir = os.path.join(params.DIR, f"run_{params.RUN_NUM}")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(
            log_dir,
            "run_plot.log"),
        "w+")
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)  # Set the logger's level to the lowest level (DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Logger set up successfully")


def setup_device():
    device = torch.device('cuda')
    logger.info(f"Running on device: {device}")
    return device


def setup_environment(params, device):
    if params.MESH:
        logger.info(f"Simulation environment: {params.MESH}")
        return MeshDomain(params.MESH, device), wave_equation

    logger.info("Simulation environment: no mesh")
    return SimpleDomain(device), wave_equation_simplified

def run():
    params = SimulationParameters()
    setup_params(params)
    setup_logger(params)
    device = setup_device()
    environment, _ = setup_environment(params, device)

    pinn = PINN(params.LAYERS, params.NEURONS_PER_LAYER, device).to(device)

    path = os.path.join(params.DIR, f"run_{params.RUN_NUM}")
    plot_all_from_file(path, pinn, environment, limit=(0,2),
                       limit_wave=(params.BASE_HEIGHT - params.PEAK_HEIGHT, params.BASE_HEIGHT + params.PEAK_HEIGHT))


if __name__ == "__main__":
    start_time = datetime.now()
    run()
    time_elapsed = datetime.now() - start_time
    print('Time elapsed: {}'.format(time_elapsed))
