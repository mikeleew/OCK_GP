import jax
import jax.numpy as jnp
from typing import Callable

def exp_dot_kernel(x, y, bw=1.0):
    '''Compute the Exponential Dot Product kernel between two sets of vectors.'''
    dot_product = x @ jnp.swapaxes(y, -1, -2)
    return jnp.exp(dot_product / bw**2)

def gaussian_kernel(x, y, bw=1.0):
    '''Compute the Gaussian kernel between two sets of vectors.'''
    norm2x = jnp.linalg.norm(x, axis=-1) ** 2
    norm2y = jnp.linalg.norm(y, axis=-1) ** 2
    cross = x @ jnp.swapaxes(y, -1, -2)
    return jnp.exp(-0.5 * (norm2x[:, None] + norm2y[None, :] - 2 * cross) / bw**2)

def gaussian_kernel_len_scales(x, y, len_scales):
    '''Compute the Gaussian kernel between two sets of vectors with length scales.'''
    diff = (x[:, None, :] - y[None, :, :]) / len_scales[None, None, :]
    norm2 = jnp.sum(diff ** 2, axis=-1)
    return jnp.exp(-0.5 * norm2)

def grad_gaussian_kernel(x, y, bw=1.0):
    '''Compute the gradient of the Gaussian kernel in the first argument.'''
    norm2x = jnp.linalg.norm(x, axis=-1) ** 2
    norm2y = jnp.linalg.norm(y, axis=-1) ** 2
    cross = x @ jnp.swapaxes(y, -1, -2)
    kern = jnp.exp(-0.5 * (norm2x[:, None] + norm2y[None, :] - 2 * cross) / bw**2)
    return -kern[:, :, None] * (x[:, None, :] - y[None, :, :])/bw**2


def gaussian_kernel_safe(x, y, bw=1.0):
    '''Compute the Gaussian kernel between two sets of vectors.
    Attempts to conserve memory.'''
    norm2x = jnp.linalg.norm(x, axis=-1) ** 2
    norm2y = jnp.linalg.norm(y, axis=-1) ** 2
    cross = x @ jnp.swapaxes(y, -1, -2)
    n_x = x.shape[0]
    n_y = y.shape[0]
    out = jnp.zeros((n_x, n_y))
    for k in range(n_y):
        out.at[:, k].set(jnp.exp(-0.5 * (norm2x + norm2y[k] - 2 * cross[:, k]) / bw**2))
    return out

def gaussian_kernel_multi(x, y, bw=1.0):
    '''Compute the Gaussian kernel between two sets of multiple vectors.
    
    Args:
        x: [n, d] or [n, M, d]
        y: [m, d] or [m, M, d]
        bw: float: The bandwidth of the kernel.
    Returns:
        [n, M, m] or [n, M, m, M]'''
    len_x = len(x.shape)
    len_y = len(y.shape)
    if len_x == 2 and len_y == 2:
        return gaussian_kernel(x, y, bw=bw)  # [n, m]
    if len_x == 3:
        M_x = x.shape[1]
    else:
        M_x = 1
    if len_y == 3:
        M_y = y.shape[1]
    else:
        M_y = 1
    d = x.shape[-1]
    n_x = x.shape[0]
    n_y = y.shape[0]
    x = x.reshape(-1, d)
    y = y.reshape(-1, d)
    out = gaussian_kernel(x, y, bw=bw)  # [n*M, m] or [n, m*M] or [n*M, m*M]
    out = out.reshape(n_x, M_x, n_y, M_y)
    return jnp.squeeze(out)  # [n, M, m] or [n, m, M] or [n, M, m, M]


def RBF_multi_safe(x, y, bw=1.0):
    '''Compute the Gaussian kernel between two sets of multiple vectors
    Attempts to convserve memory.
    
    Args:
        x: [n, d] or [n, M, d]
        y: [m, d] or [m, M, d]
        bw: float: The bandwidth of the kernel.
    Returns:
        [n, M, m] or [n, M, m, M]'''
    len_x = len(x.shape)
    len_y = len(y.shape)
    if len_x == 2 and len_y == 2:
        return gaussian_kernel(x, y, bw=bw)  # [n, m]
    if len_x == 3:
        M_x = x.shape[1]
    else:
        M_x = 1
    if len_y == 3:
        M_y = y.shape[1]
    else:
        M_y = 1
    d = x.shape[-1]
    n_x = x.shape[0]
    n_y = y.shape[0]
    # x = x.reshape(-1, d)
    # y = y.reshape(-1, d)
    if len(x.shape) == 2:
        x = jnp.expand_dims(x, axis=1)
    if len(y.shape) == 2:
        y = jnp.expand_dims(y, axis=1)
    out = jnp.zeros((n_x, M_x, n_y, M_y))
    for i in range(M_x):
        # print(i)
        for j in range(M_y):
            out.at[:, i, :, j].set(gaussian_kernel_safe(x[:, i], y[:, j], bw=bw))
    # out = gaussian_kernel(x, y, bw=bw)  # [n*M, m] or [n, m*M] or [n*M, m*M]
    # out = out.reshape(n_x, M_x, n_y, M_y)
    return jnp.squeeze(out)  # [n, M, m] or [n, m, M] or [n, M, m, M]

def linear_kernel(x, y):
    return x @ jnp.swapaxes(y, -1, -2)  # [n, m]

def grad_linear_kernel(x, y):
    return jnp.ones(x.shape)[:, None, :]*y[None, :, :]  # [n, 1, d] * [1, m, d] = [n, m, d]

def linear_kernel_multi(x, y):
    '''Compute the Gaussian kernel between two sets of multiple vectors.
    
    Args:
        x: [n, d] or [n, M, d]
        y: [m, d] or [m, M, d]
        bw: float: The bandwidth of the kernel.
    Returns:
        [n, M, m] or [n, M, m, M]'''
    len_x = len(x.shape)
    len_y = len(y.shape)
    if len_x == 2 and len_y == 2:
        return linear_kernel(x, y)  # [n, m]
    if len_x == 3:
        M_x = x.shape[1]
    else:
        M_x = 1
    if len_y == 3:
        M_y = y.shape[1]
    else:
        M_y = 1
    d = x.shape[-1]
    n_x = x.shape[0]
    n_y = y.shape[0]
    x = x.reshape(-1, d)
    y = y.reshape(-1, d)
    out = linear_kernel(x, y)  # [n*M, m] or [n, m*M] or [n*M, m*M]
    out = out.reshape(n_x, M_x, n_y, M_y)
    return out  # [n, M, m]

def multi_kernel(x, y, kernel: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], **kwargs):
    '''
    Compute the kernel between two sets of multiple vectors.
    Pass in any kernel function as the kernel argument.
    
    Args:
        x: [n, d] or [n, M, d]
        y: [m, d] or [m, M, d]
        kernel: callable: The kernel function to use.
        **kwargs: The keyword arguments for the kernel.
    Returns:
        [n,m] or [n, M, m] or [n, M, m, M]'''
    bw = kwargs.get('bw')
    if bw is not None:
        kernel = lambda x, y: kernel(x, y, bw=bw)
    len_x = len(x.shape)
    len_y = len(y.shape)
    squeeze_x = False
    squeeze_y = False
    if len_x == 2 and len_y == 2:
        return kernel(x, y, bw=bw)  # [n, m]
    if len_x == 3:
        M_x = x.shape[1]
    else:
        M_x = 1
        squeeze_x = True
    if len_y == 3:
        M_y = y.shape[1]
    else:
        M_y = 1
        squeeze_y = True
    d = x.shape[-1]
    n_x = x.shape[0]
    n_y = y.shape[0]
    x = x.reshape(-1, d)
    y = y.reshape(-1, d)
    out = kernel(x, y)  # [n*M, m] or [n, m*M] or [n*M, m*M]
    out = out.reshape(n_x, M_x, n_y, M_y)
    if squeeze_x:
        out = jnp.squeeze(out, 1)
    if squeeze_y:
        out = jnp.squeeze(out, -1)
    return out


def polynomial_kernel(x, y, degree=3):
    return (x @ jnp.swapaxes(y, -1, -2) + 1) ** degree

def affine_kernel(x, y, c=1.0):
    return x @ jnp.swapaxes(y, -1, -2) + c

def constant_kernel(x, y, c=1.0):
    return jnp.ones((x.shape[0], y.shape[0])) * c  # [n, m]

class FeatureMap:
    """The feature map for a kernel.
    
    Kernels implemented: Gaussian, linear, polynomial, constant, implicit

    Currently only supports kernels for one-dimenisonal input data
    
    Args:
    kernel_type: str: The type of kernel to use. Default is Gaussian.
    **kwargs:   The keyword arguments for the kernel. For Gaussian, this includes bandwidth, num_features, and input_dim
                For Polynomial, this includes degree.
    """
    def __init__(self, kernel_type: str = 'Gaussian', **kwargs):
        self.kernel_type = kernel_type
        if kernel_type.upper() == 'GAUSSIAN':
            bandwidth = kwargs.get('bandwidth', 1.0)
            num_features = kwargs.get('num_features', 100)
            input_dim = kwargs.get('input_dim', 2)
            if num_features is None:
                raise ValueError('num_features must be provided.')
            if input_dim is None:
                raise ValueError('input_dim must be provided.')
            if bandwidth is None:
                print('Warning: bandwidth not provided. Defaulting to 1.0.')
                bandwidth = 1.0
            self.bandwidth = bandwidth
            self.num_features = num_features
            self.input_dim = input_dim
            rng = jax.random.PRNGKey(0)
            self.features = jax.random.normal(rng, (self.input_dim, self.num_features)) / bandwidth  # [input_dim, num_features]
            rng, _ = jax.random.split(rng)
            self.uniform_features = jax.random.uniform(rng, minval=0.0, maxval=2*jnp.pi, shape=(self.num_features))  # [input_dim, num_features]
            def fourier_features(x):  # Use the lower-dimensional Fourier features
                feat_x = x @ self.features + self.uniform_features # [num_points, num_features]
                cos_feat_x = jnp.cos(feat_x)  # [num_points, num_features]
                # sin_feat_x = jnp.sin(feat_x / bandwidth)
                # return 1/jnp.sqrt(self.features.shape[1])*jnp.concatenate([cos_feat_x, sin_feat_x], axis=-1)
                return jnp.sqrt(2.0/self.features.shape[1])*cos_feat_x
            self.feature_map = lambda x: fourier_features(x)
        elif kernel_type.upper() == 'LINEAR':
            self.feature_map = lambda x: x[:, None] if len(x.shape) == 1 else x
        elif kernel_type.upper() == 'POLYNOMIAL':
            degree = kwargs.get('degree')
            if degree is None:
                print('Warning: degree not provided. Defaulting to 3.')
                degree = 3
            def poly_feature_map(x, degree=degree):
                powx = jnp.ones((x.shape[0], 1))
                for k in range(degree):
                    powx = jnp.concatenate([powx, x ** (k+1)], axis=-1)
                return powx
            self.feature_map = lambda x: poly_feature_map(x, degree=degree)
        elif kernel_type.upper() == 'CONSTANT':
            c = kwargs.get('c', 1.0)
            if c is None:
                c = 1.0
            self.feature_map = lambda x: jnp.ones((x.shape[0], 1)) * c
        elif kernel_type.upper() == 'IMPLICIT':
            traj = kwargs.get('traj')
            kernel = kwargs.get('kernel')
            if traj is None:
                raise ValueError('traj must be provided for implicit kernel.')
            traj_data = jnp.concatenate(traj, axis=0)
            K_mat = kernel(traj_data, traj_data)
            eigvals, _ = jnp.linalg.eigh(K_mat)
            l_max = jnp.max(eigvals)
            l_min = jnp.min(eigvals)
            cond_num = 1e5
            kappa = (l_max - cond_num*l_min) / (cond_num - 1)  # Fix the condition number of the kernel matrix
            R = jax.scipy.linalg.cholesky(K_mat + kappa*jnp.eye(K_mat.shape[0]))  # [num_points, num_points]  possibly need to add small diagonal perturbation
            full_rank = False
            if jnp.linalg.matrix_rank(K_mat) == K_mat.shape[0]:
                full_rank = True
            def implicit_feature_map(x, full_rank=full_rank, R=R, kernel=kernel, traj_data=traj_data):
                K_x = kernel(x, traj_data)
                
                if full_rank:
                    return jnp.linalg.solve(R.T, K_x.T).T  # [num_points, num_features]
                else:
                    return jnp.linalg.lstsq(R.T, K_x.T, rcond=None)[0].T
            self.feature_map = lambda x: implicit_feature_map(x)
        else:
            raise ValueError('Invalid kernel type. Must be Gaussian, linear, polynomial, constant, or implicit.')