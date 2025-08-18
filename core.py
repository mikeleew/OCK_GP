import jax
import jax.numpy as jnp
from typing import List, Tuple, Union, Callable

def get_delta_t(t: List[jnp.ndarray]) -> jnp.ndarray:
    '''
    Get the difference in the time points of the trajectories.
    Args:
        t: List of arrays, each array is the time points of the trajectory.
    Returns:
        jnp.ndarray: Array of the difference in the time points of the trajectories.
    '''
    t_diff =  [t[k][1:] - t[k][:-1] for k in range(len(t))]
    return jnp.concatenate(t_diff, axis=0)

def get_delta_t_list(t: List[jnp.ndarray]) -> List[jnp.ndarray]:
    '''
    Get the difference in the time points of the trajectories.
    Args:
        t: List of arrays, each array is the time points of the trajectory.
    Returns:
        List[jnp.ndarray]: List of the difference in the time points of the trajectories.
    '''
    t_diff =  [t[k][1:] - t[k][:-1] for k in range(len(t))]
    return t_diff

def get_delta_y(traj: List[jnp.ndarray]) -> jnp.ndarray:
    '''
    Get the difference in the y values of the trajectories.
    Args:
        traj: List of arrays, each array is a trajectory.
    Returns:
        jnp.ndarray: Array of the difference in the y values of the trajectories.
    '''
    y_diff = [traj[k][1:] - traj[k][:-1] for k in range(len(traj))]
    return jnp.concatenate(y_diff, axis=0)


def get_traj_idx(traj: List[jnp.ndarray], include_first=False) -> jnp.ndarray:
    '''Get the indices of the trajectories for use in various computations.
    Store in array format
    
    Args:
        traj: List of arrays, each array is a trajectory.
        include_first: Boolean, whether to include the first point in each trajectory.
    Returns:
        jnp.ndarray: Array of indices for the trajectories.
    '''
    if not include_first:
        traj_len = [len(traj[k]) for k in range(len(traj))]
        traj_len = [0] + traj_len
        traj_len = jnp.cumsum(jnp.array(traj_len))
        return jnp.setdiff1d(jnp.arange(traj_len[-1]), traj_len)
    else:
        traj_len = [len(traj[k]) for k in range(len(traj))]
        traj_len = jnp.cumsum(jnp.array(traj_len))-1
        return jnp.setdiff1d(jnp.arange(traj_len[-1]), traj_len)


def get_traj_idx_list(traj: List[jnp.ndarray], include_first: float = False) -> List[jnp.ndarray]:
    '''Get the indices of the trajectories for use in various computations.
    Store in list format
    
    Args:
        traj: List of arrays, each array is a trajectory.
    Returns:
        List[jnp.ndarray]: List of indices for the trajectories.
    '''
    if not include_first:
        traj_len = [len(traj[k]) for k in range(len(traj))]
        traj_len = [0] + traj_len
        traj_len = jnp.cumsum(jnp.array(traj_len))
        return [jnp.arange(traj_len[k], traj_len[k+1]) for k in range(len(traj))]
    else:
        traj_len = [len(traj[k])-1 for k in range(len(traj))]
        traj_len = [0] + traj_len
        traj_len = jnp.cumsum(jnp.array(traj_len))
        return [jnp.arange(traj_len[k], traj_len[k+1]) for k in range(len(traj))]
