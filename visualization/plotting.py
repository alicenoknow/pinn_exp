import imageio
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import torch
import os

from typing import Callable, Tuple

from domain.domain import Domain
from domain.mesh_domain import MeshDomain
from pinn.pinn import PINN
from training.params import SimulationParameters


def plot_all(save_path: str,
             pinn: PINN,
             domain: Domain,
             initial_condition: Callable,
             limit: Tuple[float, float],
             limit_wave: Tuple[float, float]):
    os.makedirs(os.path.join(save_path, "img"), exist_ok=True)

    logging.info("Plotting initial condition results")
    plot_initial_condition(save_path, domain, pinn, initial_condition,
                           limit=limit, limit_wave=limit_wave)

    logging.info("Plotting simulation frames")
    plot_simulation_by_frame(save_path, pinn, domain, limit=limit, limit_wave=limit_wave)

    logging.info("Creating GIFs")
    create_all_gifs(save_path, SimulationParameters().T_DOMAIN[1])


def plot_all_from_file(save_path: str,
             domain: Domain,
             limit: Tuple[float, float],
             limit_wave: Tuple[float, float]):
    os.makedirs(os.path.join(save_path, "img"), exist_ok=True)
    
    logging.info("Plotting initial condition results")
    plot_initial_condition_from_file(save_path, domain, limit=limit, limit_wave=limit_wave)

    logging.info("Plotting simulation frames")
    plot_simulation_by_frame_from_file(save_path, domain, limit=limit, limit_wave=limit_wave)

    logging.info("Creating GIFs")
    create_all_gifs(save_path, SimulationParameters().T_DOMAIN[1])


def create_all_gifs(save_path: str,
                    total_time: float,
                    step: float = 0.01,
                    duration: float = 0.1):
    create_gif(save_path, "img_top", total_time, step, duration)
    create_gif(save_path, "img_side", total_time, step, duration)
    create_gif(save_path, "img_color", total_time, step, duration)


def create_gif(save_path: str,
               img_path: str,
               total_time: float,
               step: float = 0.01,
               duration: float = 0.1) -> None:
    time_values = np.arange(0, total_time, step)
    frames = []
    for idx in range(len(time_values)):
        image = imageio.v2.imread(os.path.join(save_path, "img", f"{img_path}_{idx:03d}.png"))
        frames.append(image)

    imageio.mimsave(os.path.join(save_path, f"tsunami_{img_path}.gif"), frames, duration=duration)


def plot_initial_condition(save_path: str,
                           domain: Domain,
                           pinn: PINN,
                           initial_condition: Callable,
                           limit: Tuple[float, float],
                           limit_wave: Tuple[float, float]) -> None:

    n_points_plot = SimulationParameters().POINTS_PLOT
    length = SimulationParameters().XY_DOMAIN[1]
    x, y, t = domain.get_initial_points(n_points_plot, requires_grad=False)

    z_true = initial_condition(x, y, length)
    z_pred = pinn(x, y, t)
    z_true, z_pred = convert_to_numpy(z_true, n_points_plot), convert_to_numpy(z_pred, n_points_plot)
    _plot_initial_condition(z_true, z_pred, save_path, domain, limit, limit_wave)


def plot_initial_condition_from_file(save_path: str,
                           domain: Domain,
                           limit: Tuple[float, float],
                           limit_wave: Tuple[float, float]) -> None:
    n_points_plot = SimulationParameters().POINTS_PLOT

    z_true = np.load(os.path.join(save_path, "saved", "true_initial.npy"))
    z_pred = np.load(os.path.join(save_path, "saved", "pred_initial.npy"))
    z_true, z_pred = z_true.reshape(n_points_plot, n_points_plot), z_pred.reshape(n_points_plot, n_points_plot)
    _plot_initial_condition(z_true, z_pred, save_path, domain, limit, limit_wave)

def _plot_initial_condition(z_true: np.ndarray,
                           z_pred: np.ndarray, 
                           save_path: str,
                           domain: Domain,
                           limit: Tuple[float, float],
                           limit_wave: Tuple[float, float]) -> None:

    title = "Initial condition"
    n_points_plot = SimulationParameters().POINTS_PLOT

    x, y, _ = domain.get_initial_points(n_points_plot, requires_grad=False)
    x, y = convert_to_numpy(x, n_points_plot), convert_to_numpy(y, n_points_plot)

    fig1 = plot_cmap(x, y, z_true, f"{title} - exact", limit=limit_wave)
    fig2 = plot_3D_matplotlib(x, y, z_true, domain,
                   f"{title} - exact", limit=limit, limit_wave=limit_wave)
    fig3 = plot_cmap(x, y, z_pred, f"{title} - PINN", limit=limit_wave)
    fig4 = plot_3D_matplotlib(x, y, z_pred, domain,
                   f"{title} - PINN", limit=limit, limit_wave=limit_wave)

    c1, c2, c3, c4 = fig1.canvas, fig2.canvas, fig3.canvas, fig4.canvas

    c1.draw()
    c2.draw()
    c3.draw()
    c4.draw()

    a1, a2, a3, a4 = np.array(c1.buffer_rgba()), np.array(c2.buffer_rgba()), \
        np.array(c3.buffer_rgba()), np.array(c4.buffer_rgba())
    a12, a34 = np.vstack((a1, a2)), np.vstack((a3, a4))
    a = np.hstack((a12, a34))

    fig, ax = plt.subplots(figsize=(100, 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.matshow(a)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "initial.png"))

    plt.close(fig)
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

def plot_cmap(x: np.ndarray,
               y: np.ndarray,
               z: np.ndarray,
               title: str,
               figsize=(8, 6), dpi=100, cmap="viridis", limit=(0,1)):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

    c = ax.pcolormesh(x, y, z, cmap=cmap, vmin=limit[0], vmax=limit[1])
    fig.colorbar(c, ax=ax)

    return fig

def plot_3D_matplotlib(x: np.ndarray, 
                        y: np.ndarray, 
                        z: np.ndarray,
                        domain: Domain,
                        title: str, figsize=(8, 6),
                        limit: Tuple[float] = (0, 1),
                        limit_wave: Tuple[float] = (0, 1)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.axes.set_zlim3d(bottom=limit[0], top=limit[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    surf = ax.plot_surface(x, y, z, alpha=0.9, vmin=limit_wave[0], vmax=limit_wave[1], cmap="Blues_r")

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.05)
    cbar.set_label('Water altitude', fontsize=14)

    if isinstance(domain, MeshDomain):
        trisurf = ax.plot_trisurf(domain.x_raw,
                                  domain.y_raw,
                                  domain.z_raw,
                                  linewidth=0.2, alpha=0.8, cmap="plasma")

        cbar_trisurf = fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=5, pad=0.15, location='left')
        cbar_trisurf.set_label('Terrain altitude', fontsize=14)
        cbar_trisurf.ax.yaxis.set_label_position('left')
        cbar_trisurf.ax.yaxis.set_ticks_position('left')

    return fig

def plot_3D_top_view(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     domain: Domain,
                     title: str,
                     limit: Tuple[float, float],
                     limit_wave: Tuple[float, float]):

   top = np.maximum(limit[1], limit_wave[1]) + 0.1
   return plot_3D(x, y, z, domain, title, limit, limit_wave, eye=dict(x=0, y=0, z=top))


def plot_3D_side_view(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      domain: Domain,
                      title: str,
                      limit: Tuple[float, float],
                      limit_wave: Tuple[float, float]):

   return plot_3D(x, y, z, domain, title, limit, limit_wave, eye=dict(x=1.3, y=2.8, z=0.8))

def plot_3D(x: np.ndarray, y: np.ndarray, z: np.ndarray,    
            domain: Domain,
            title: str,
            limit: Tuple[float, float],
            limit_wave: Tuple[float, float],
            eye: dict):

    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=z, opacity=1,
        cmin=limit_wave[0], cmax=limit_wave[1], colorscale="Blues_r",
        colorbar=dict(title=dict(text='Water altitude', font=dict(size=16)))
    )])

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis=dict(title=dict(text="x", font=dict(size=16)), tickvals=[]),
            yaxis=dict(title=dict(text="y", font=dict(size=16)), tickvals=[]),
            zaxis=dict(range=[limit[0], limit[1]], tickvals=[], title=dict(text="z", font=dict(size=16))),
            camera=dict(eye=eye),
            aspectratio=dict(x=1, y=1, z=0.5)
        )
    )

    if isinstance(domain, MeshDomain):
        fig.add_trace(go.Mesh3d(
            x=domain.x_raw,
            y=domain.y_raw,
            z=domain.z_raw,
            opacity=1,
            intensity=domain.z_raw,
            colorscale='Plasma',
            colorbar=dict(title=dict(text='Terrain altitude', font=dict(size=16)), x=-0.2),
        ))

    return fig

def plot_frame(x: np.ndarray,
               y: np.ndarray,
               z: np.ndarray,
               save_path: str,
               domain: Domain,
               idx: int,
               t_value: float,
               limit: float,
               limit_wave: float) -> None:
    title = f"PINN for t = {t_value:.3f}"

    fig1 = plot_cmap(x, y, z, title, limit=limit_wave)
    plt.savefig(os.path.join(save_path, "img", "img_color_{:03d}.png".format(idx)))
    plt.close(fig1)

    fig2 = plot_3D_top_view(x, y, z, domain,
                            title, limit=limit, limit_wave=limit_wave)
    fig3 = plot_3D_side_view(x, y, z, domain,
                             title, limit=limit, limit_wave=limit_wave)
    fig2.write_image(os.path.join(save_path, "img", "img_top_{:03d}.png".format(idx)))
    fig3.write_image(os.path.join(save_path, "img", "img_side_{:03d}.png".format(idx)))

def plot_simulation_by_frame(save_path: str,
                             pinn: PINN,
                             domain: Domain,
                             time_step: float = 0.01,
                             limit: Tuple[float, float] = (0, 1),
                             limit_wave: Tuple[float, float] = (0, 1)) -> None:
    t_max = SimulationParameters().T_DOMAIN[1]
    n_points_plot = SimulationParameters().POINTS_PLOT
    time_values = np.arange(0, t_max, time_step)
    x, y, t = domain.get_initial_points(n_points_plot, requires_grad=False)
    x_np, y_np = convert_to_numpy(x, n_points_plot), convert_to_numpy(y, n_points_plot)

    for idx, t_value in enumerate(time_values):
        t = torch.full_like(x, t_value)
        z = convert_to_numpy(pinn(x, y, t), n_points_plot)

        plot_frame(x=x_np, y=y_np, z=z,
                   save_path=save_path,
                   domain=domain,
                   idx=idx,
                   t_value=t_value,
                   limit=limit,
                   limit_wave=limit_wave)
        
def plot_simulation_by_frame_from_file(save_path: str,
                             domain: Domain,
                             time_step: float = 0.01,
                             limit: Tuple[float, float] = (0, 1),
                             limit_wave: Tuple[float, float] = (0, 1)) -> None:
    t_max = SimulationParameters().T_DOMAIN[1]
    n_points_plot = SimulationParameters().POINTS_PLOT
    time_values = np.arange(0, t_max, time_step)
    x, y, _ = domain.get_initial_points(n_points_plot, requires_grad=False)
    x_np, y_np = convert_to_numpy(x, n_points_plot), convert_to_numpy(y, n_points_plot)

    for idx, t_value in enumerate(time_values):
        t = torch.full_like(x, t_value)
        z = np.load(os.path.join(save_path, f"saved/data_{idx}.npy"))
        z = z.reshape(n_points_plot, n_points_plot)

        plot_frame(x=x_np, y=y_np, z=z,
                   save_path=save_path,
                   domain=domain,
                   idx=idx,
                   t_value=t_value,
                   limit=limit,
                   limit_wave=limit_wave)


def running_average(y, window: int = 100):
    cumsum = np.cumsum(np.insert(y, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def plot_running_average(save_path: str, loss_values, title: str, path: str):
    average_loss = running_average(loss_values, window=100)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.plot(average_loss)
    ax.set_yscale('log')

    fig.savefig(os.path.join(save_path, f"{path}.png"))

def convert_to_numpy(tensor: torch.Tensor, n_points: int) -> np.array:
    return tensor.detach().cpu().numpy().reshape(n_points, n_points)
