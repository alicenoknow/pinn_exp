import torch
from typing import Union

G = 9.81

def f(pinn: "PINN",  # noqa: F821 # type: ignore
      x: torch.Tensor,
      y: torch.Tensor,
      t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return pinn(x, y, t)

def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Compute the nth-order derivative of a tensor with respect to the input tensor.

    Args:
        output (torch.Tensor): The output tensor, usually a result of some computation.
        input (torch.Tensor): The input tensor with respect to which the derivative is computed.
        order (int): The order of the derivative to compute. Default is 1 (first derivative).

    Returns:
        torch.Tensor: The computed nth-order derivative tensor.
    """
    df_value = output
    for _ in range(order):
        df_value, = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(df_value),
            create_graph=True,
            retain_graph=True,
        )
    return df_value

def dfdt(pinn: "PINN",  # noqa: F821 # type: ignore
         x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
         u: Union[torch.Tensor, None] = None, order: int = 1) -> torch.Tensor:
    """Compute the time derivative of the solution"""
    u = u if u is not None else f(pinn, x, y, t)
    return df(u, t, order=order)

def dfdx(pinn: "PINN",  # noqa: F821 # type: ignore
         x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
         u: Union[torch.Tensor, None] = None, order: int = 1) -> torch.Tensor:
    """Compute the x derivative of the solution"""
    u = u if u is not None else f(pinn, x, y, t)
    return df(u, x, order=order)

def dfdy(pinn: "PINN",  # noqa: F821 # type: ignore
         x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
         u: Union[torch.Tensor, None] = None, order: int = 1) -> torch.Tensor:
    """Compute the y derivative of the solution"""
    u = u if u is not None else f(pinn, x, y, t)
    return df(u, y, order=order)

def wave_equation_simplified(
        pinn: "PINN",  # noqa: F821 # type: ignore
        x: torch.Tensor,
        y: torch.Tensor,
        _z: torch.Tensor,
        t: torch.Tensor,
        _dzdx: torch.Tensor,
        _dzdy: torch.Tensor) -> torch.Tensor:
    """Simplified wave equation"""
    u = f(pinn, x, y, t)
    d2u_dt2 = dfdt(pinn, x, y, t, u, order=2)
    du_dx = dfdx(pinn, x, y, t, u)
    du_dy = dfdy(pinn, x, y, t, u)
    d2u_dx2 = dfdx(pinn, x, y, t, u, order=2)
    d2u_dy2 = dfdy(pinn, x, y, t, u, order=2)
    return d2u_dt2 - G * (du_dx ** 2 + du_dy ** 2 + u * (d2u_dx2 + d2u_dy2))

def wave_equation(
        pinn: "PINN",  # noqa: F821 # type: ignore
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        dzdx: torch.Tensor,
        dzdy: torch.Tensor) -> torch.Tensor:
    """Wave equation with full complexity"""
    u = f(pinn, x, y, t)
    u_minus_z = u - z
    X1 = torch.maximum(u_minus_z, torch.tensor(0.0, device=u.device))
    d2u_dx2 = dfdx(pinn, x, y, t, u, order=2)
    d2u_dy2 = dfdy(pinn, x, y, t, u, order=2)
    X2 = d2u_dx2 + d2u_dy2

    du_dx = dfdx(pinn, x, y, t, u)
    du_dy = dfdy(pinn, x, y, t, u)

    val_x = (du_dx - dzdx) * du_dx
    val_y = (du_dy - dzdy) * du_dy

    X3 = torch.where(u_minus_z >= 0, val_x, torch.tensor(0.0, device=u.device))
    X4 = torch.where(u_minus_z >= 0, val_y, torch.tensor(0.0, device=u.device))

    d2u_dt2 = dfdt(pinn, x, y, t, u, order=2)
    return d2u_dt2 - G * (X1 * X2 + X3 + X4)
