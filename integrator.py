import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, List, Tuple

def main_loop(f, xi, step_size):
    k1 = f(xi)
    k2 = f(xi + 0.5 * step_size * k1)
    k3 = f(xi + 0.5 * step_size * k2)
    k4 = f(xi + step_size * k3)
    return step_size / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def rk4_solver(f: Callable[[jnp.ndarray], jnp.ndarray], x0 : jnp.ndarray, t: List[jnp.ndarray], step_size: float = 0.001):
    '''
    Runga-Kutta 4th order solver for ODEs
    Args:
    f: function that defines the ODE
    x0: initial condition of size [num_traj, d]
    t: time points  List of num_traj elements, the j^th element is a 1D array of size [num_time_steps_j]
    step_size: step size.  A float for the step size of the solver
    Returns:
    x: solution of the ODE at time points t as a list of n arrays of size [num_time_steps_j, d]
    '''
    num_traj = x0.shape[0]
    d = x0.shape[1]
    if num_traj != len(t):
        raise ValueError('Number of trajectories in x0 and t must be the same')
    left = [t[i][0] for i in range(num_traj)]
    right = [t[i][-1] for i in range(num_traj)]
    num_steps = int(max([jnp.ceil((right[i] - left[i]) / step_size) + 1 for i in range(num_traj)]))  # type: ignore
    tstar = [jnp.linspace(left[i], left[i] + (num_steps-1)*step_size, num_steps) for i in range(num_traj)]
    tindex = [jnp.argmin(jnp.abs(tstar[i][:, None] - t[i]), axis=0) for i in range(num_traj)]
    # x = jnp.zeros((num_traj, num_steps, d))
    # x.at[:, 0, :].set(x0)
    x = x0[:, None, :]
    main_loop_jit = jax.jit(main_loop, static_argnums=0)
    for i in range(num_steps-1):
        #k1 = f(x[:, i])
        #k2 = f(x[:, i] + 0.5 * step_size * k1)
        #k3 = f(x[:, i] + 0.5 * step_size * k2)
        #k4 = f(x[:, i] + step_size * k3)
        #delta_x = step_size / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        delta_x = main_loop_jit(f, x[:, i], step_size)
        x = jnp.concatenate([x, x[:, i, None, :] + delta_x[:, None, :]], axis=1)    
    traj = [x[i, tindex[i]] for i in range(num_traj)] 
    return traj
    