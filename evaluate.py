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
from visualization.plotting import plot_all

"""
Takes config file name.
Individual fields can be overridden by CLI arguments.
e.g. run.py --config base.json --layers 8
"""

logger = logging.getLogger()


def setup_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config file name')
    parser.add_argument('--run', help='Run number')

    args = parser.parse_args()
    params.set_json(args.config)
    params.set_run_num(args.run)
    params.save_params()


def setup_logger(params):
    log_format = '[%(levelname)s] %(message)s'
    log_dir = os.path.join(params.DIR, f"run_{params.RUN_NUM}")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(
            log_dir,
            "run.log"),
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


def setup_loss(environment,
               params,
               initial_condition,
               wave_equation,):
    logger.info(f"Loss: {params.LOSS}")

    if params.LOSS == LossFunction.RELO:
        return ReloLoss(environment, initial_condition, wave_equation)
    elif params.LOSS == LossFunction.SOFTADAPT:
        return SoftAdaptLoss(environment, initial_condition, wave_equation)
    return WeightedLoss(environment, initial_condition, wave_equation)


def run():
    params = SimulationParameters()
    setup_params(params)
    setup_logger(params)
    device = setup_device()

    environment, wave_equation = setup_environment(params, device)
    initial_condition = make_initial_condition(params.BASE_HEIGHT,
                                               params.DECAY_RATE,
                                               params.PEAK_HEIGHT,
                                               params.X_DIVISOR,
                                               params.Y_DIVISOR)

    logger.info(f'Loading model: {params.MODEL_PATH}')

    pinn = PINN(params.LAYERS, params.NEURONS_PER_LAYER, device)
    state_dict = torch.load(params.MODEL_PATH)
    pinn.load_state_dict(state_dict)
    pinn = pinn.cuda()

    loss = setup_loss(
        environment,
        params,
        initial_condition,
        wave_equation,
    )

    logger.info('Evaluating model')
    total_loss, initial_loss, residual_loss, boundary_loss = loss.verbose(pinn)

    logger.info(f'Total loss: \t{total_loss:.5f} ({total_loss:.3E})')
    logger.info(f'Interior loss: \t{initial_loss:.5f} ({initial_loss:.3E})')
    logger.info(f'Initial loss: \t{residual_loss:.5f} ({residual_loss:.3E})')
    logger.info(f'Boundary loss: \t{boundary_loss:.5f} ({boundary_loss:.3E})')

    logger.info('Visualizing results')
    visualize_results(params, pinn, environment, initial_condition)

def visualize_results(params, model, environment, initial_condition):
    save_dir = os.path.join(params.DIR, f"run_{params.RUN_NUM}")

    print("Saving images: ", save_dir)
    plot_all(save_dir,
             model,
             environment,
             initial_condition,
             limit=(0,2),
             limit_wave=(params.BASE_HEIGHT - params.PEAK_HEIGHT, params.BASE_HEIGHT + params.PEAK_HEIGHT))

if __name__ == "__main__":
    start_time = datetime.now()
    run()
    time_elapsed = datetime.now() - start_time
    print('Time elapsed: {}'.format(time_elapsed))
