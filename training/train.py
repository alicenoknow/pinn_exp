from typing import Callable, List, Tuple
import numpy as np
import os
import logging
import time
import torch
import signal
from domain.domain import Domain
from loss.loss import Loss
from pinn.pinn import PINN
from training.params import SimulationParameters
from visualization.plotting import plot_all, plot_running_average

logger = logging.getLogger()

class Training:
    def __init__(self, model: PINN,
                 loss: Loss,
                 domain: Domain,
                 initial_condition: Callable) -> None:
        self.model = model
        self.loss = loss
        self.domain = domain
        self.initial_condition = initial_condition
        self.params = SimulationParameters()
        self.best_loss = float("inf")
        self.interrupted = False

        # Set up signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signal, frame):
        self.interrupted = True
        logger.info("Training interrupted by user")

    def start(self):
        self.create_run_directory()
        
        logging.info(f"Starting training run: {self.params.RUN_NUM}")
        start = time.time()

        loss_total, loss_r, loss_i, loss_b = self.train()
        loss_total_lbfgs, loss_r_lbfgs, loss_i_lbfgs, loss_b_lbfgs = self.train_lbfgs()
        loss_total = np.concatenate((loss_total, loss_total_lbfgs))
        loss_r = np.concatenate((loss_r, loss_r_lbfgs))
        loss_i = np.concatenate((loss_i, loss_i_lbfgs))
        loss_b = np.concatenate((loss_b, loss_b_lbfgs))

        logging.info(f"Finished training in: {time.time() - start}")
        logging.info("Visualizing results")
        
        self.print_summary(loss_total[-1], loss_r[-1], loss_i[-1], loss_b[-1])
        self.visualize_results((loss_total, loss_r, loss_i, loss_b))

        return self.model, loss_total, loss_r, loss_i, loss_b

    def train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.LEARNING_RATE)
        loss_values, residual_loss_values, initial_loss_values, boundary_loss_values = [], [], [], []

        for epoch in range(self.params.EPOCHS):
            if self.interrupted:
                break
            try:
                self.model.train()
                loss = self.loss(self.model)
                optimizer.zero_grad()
                loss[0].backward()
                optimizer.step()

                loss_values.append(loss[0].item())
                residual_loss_values.append(loss[1].item())
                initial_loss_values.append(loss[2].item())
                boundary_loss_values.append(loss[3].item())

                self.save_best_callback(loss[0].item())

                if (epoch + 1) % 1000 == 0:
                    self.print_epoch_report(epoch, loss)

            except KeyboardInterrupt:
                self.interrupted = True
                break

        return (np.array(loss_values), 
                np.array(residual_loss_values), 
                np.array(initial_loss_values), 
                np.array(boundary_loss_values))

    def train_lbfgs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        loss_values, residual_loss_values, initial_loss_values, boundary_loss_values = [], [], [], []
    
        lbfgs_optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1.0, history_size=100, line_search_fn="strong_wolfe", max_iter=20)

        def closure():
            self.model.train()
            loss = self.loss(self.model)
            lbfgs_optimizer.zero_grad()
            loss[0].backward()
            return loss[0]
        
        for epoch in range(self.params.LBFGS_EPOCHS):
            if self.interrupted:
                break
            try:
                total_loss = lbfgs_optimizer.step(closure).item()
                loss = self.loss(self.model)
                
                loss_values.append(total_loss)
                residual_loss_values.append(loss[1].item())
                initial_loss_values.append(loss[2].item())
                boundary_loss_values.append(loss[3].item())

                if (epoch + 1) % 1000 == 0:
                    self.print_epoch_report(epoch, loss)

            except KeyboardInterrupt:
                self.interrupted = True
                break
            
        return (np.array(loss_values), 
                np.array(residual_loss_values), 
                np.array(initial_loss_values), 
                np.array(boundary_loss_values))

    def visualize_results(self, losses: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        save_path = os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}")
        self.plot_averages(losses)

        os.makedirs(os.path.join(save_path, "img"), exist_ok=True)
        plot_all(save_path, self.model, self.domain, self.initial_condition, 
                 limit=(0, 2), limit_wave=(self.params.BASE_HEIGHT - self.params.PEAK_HEIGHT, self.params.BASE_HEIGHT + self.params.PEAK_HEIGHT))

    def plot_averages(self, losses: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        save_path = os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}")
        plot_running_average(save_path, losses[0], "Loss function (running average)", "total_loss")
        plot_running_average(save_path, losses[1], "Residual loss function (running average)", "residual_loss")
        plot_running_average(save_path, losses[2], "Initial loss function (running average)", "initial_loss")
        plot_running_average(save_path, losses[3], "Boundary loss function (running average)", "boundary_loss")

    def print_summary(self, total_loss: float, initial_loss: float, residual_loss: float, boundary_loss: float):
        logger.info(f'Total loss: \t{total_loss:.5f} ({total_loss:.3E})')
        logger.info(f'Interior loss: \t{initial_loss:.5f} ({initial_loss:.3E})')
        logger.info(f'Initial loss: \t{residual_loss:.5f} ({residual_loss:.3E})')
        logger.info(f'Boundary loss: \t{boundary_loss:.5f} ({boundary_loss:.3E})')

    def print_epoch_report(self, epoch: int, loss: List[torch.Tensor]):
        logger.info(f"Epoch: {epoch + 1} - "
                    f"Loss: {loss[0].item():.7f}, "
                    f"Residual Loss: {loss[1].item():.7f}, "
                    f"Initial Loss: {loss[2].item():.7f}, "
                    f"Boundary Loss: {loss[3].item():.7f}")

    def create_run_directory(self):
        try:
            os.makedirs(os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}"), exist_ok=True)
            logger.info("Run directory created successfully")
        except OSError as error:
            logger.error(f"Run directory creation failed: {error}")

    def save_best_callback(self, loss: float):
        if loss < self.best_loss:
            torch.save(self.model.state_dict(), os.path.join(self.params.DIR, f"run_{self.params.RUN_NUM}", f"best_{self.params.RUN_NUM}.pt"))
            self.best_loss = loss
