from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple, Union
from diffrax._custom_types import FloatScalarLike

@jax.jit
def ftle(
        DF: Float[Array, "... 2 2"], 
        T: FloatScalarLike, 
        reshape: Union[None, Tuple] = None
) -> Float[Array, "..."]:
    """
    Batched function for computing FTLE from derivative of flowmap DF for integration time T. Can
    reshape the output into shape=reshape.

    Parameters
    ----------
    DF : jnp.array
        array containing a single, or all DFs for computing FTLE.
    T : float
        integration time.
    reshape : None or tuple, optional
        desired shape of output. If None passed, ftle will be returned as a vector. 
        The default is None.

    Returns
    -------
    jnp.array
        ftle value(s).

    """

    dxdx = DF[..., 0, 0]
    dxdy = DF[..., 0, 1]
    dydx = DF[..., 1, 0]
    dydy = DF[..., 1, 1]
    
    a1 = dxdx**2 + dydx**2
    a2 = dxdx*dxdy + dydx*dydy
    a3 = dxdy**2 + dydy**2    
    
    eigval_max = 0.5*(a1 + a3 + jnp.sqrt((a1 - a3)**2 + 4*a2**2))
    
    safe_eigval = jnp.maximum(eigval_max, 1.0)
    
    ftle_ = (1 / (2 * jnp.abs(T))) * jnp.log(safe_eigval)
    
    if reshape is not None:
        return ftle_.reshape(reshape)
    else:
        return ftle_


@partial(jax.jit, static_argnames=['reshape'])
def DF_svd(
        DF: Float[Array, "... 2 2"], 
        reshape: Union[None, Tuple] = None
) -> Tuple[Float[Array, "... 2"], Float[Array, "... 2 2"]]:
    """
    Wrapper for jnp.linalg.svd, computed singular values and singular vectors for DF. Can reshape
    the leading dims of output into shape=reshape/
    

    Parameters
    ----------
    DF : jnp.array
        array containing a single, or all DFs/
    reshape : None or Tuple, optional
        desired shape of output. If None passed, ftle will be returned as a vector. 
        The default is None.

    Returns
    -------
    s : jnp.array
        singular values of DFs.
    vh : jnp.array
        right singular vectors of DFs.

    """
    
    _, s, vh = jnp.linalg.svd(DF)
    
    if reshape is not None:
        s = s.reshape(reshape, + (2,))
        vh = vh.reshape(reshape, + (2, 2))
        
    
    return s, vh

def neighbor_views(arr, mode='constant'):
    """
    Get neighbor views of batched array for fast finite differencing.

    Parameters
    ----------
    arr : jnp.array
        array we wish to perform finite differencing on.
    mode : str, optional
        pad mode. The default is 'constant'.

    Returns
    -------
    nhbr_views : dict (pytree)
        dict containing views to slices of array for finite differencing.

    """
    
    pad_width = [(0, 0)] * (arr.ndim - 2) + [(1, 1)] * 2
    
    pad_arr = jnp.pad(arr, pad_width, mode=mode)
    
    nhbr_views = {
        'center': pad_arr[..., 1:-1, 1:-1],
        'west': pad_arr[..., :-2, 1:-1],
        'east': pad_arr[..., 2:, 1:-1],
        'north': pad_arr[..., 1:-1, 2:],
        'south': pad_arr[..., 1:-1, :-2],
        'nw': pad_arr[..., :-2, 2:],
        'ne': pad_arr[..., 2:, 2:],
        'sw': pad_arr[..., :-2, :-2],
        'se': pad_arr[..., 2:, :-2]
    }
    
    return nhbr_views

@jax.jit
def ftle_ridge_pts(
        s_max: Float[Array, "... 2"], 
        v_max: Float[Array, "... 2 2"],
        X: Float[Array, "..."],
        Y: Float[Array, "... 2 2"],
        dx: float,
        dy: float,
        sdd_thresh: float = 0.0, 
        percentile: float = 0.0, 
        mode: str = 'constant'
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Batched function for computing ftle ridge points using largest singular value (s_max) and
    corresponding right singular vector (v_max) over the grid defined by (X, Y) with spacing
    (dx, dy). Second directional derivative threshold given by sdd_thresh (should be nonnegative)
    percentile of s_max will skip all points with s_max less than percentile (s_max). Mode will
    stipulate if data wraps around or not (periodic). We do simple finite differencing for 
    derivatives of the singular value as these do not need to be as accurate so the very large
    increase in copmutational graph size/complexity is not worth the marginal gain in accuracy.

    Parameters
    ----------
    s_max : jnp.array
        largest singular values.
    v_max : jnp.array
        right singular vector corresponding the largest singular values.
    X : jnp.array
        meshgrid of x points ('ij' indexing).
    Y : jnp.array
        meshgrid of y points ('ij' indexing).
    dx : float
        spacing in x direction.
    dy : float
        spacing in y direction.
    sdd_thresh : float, optional
        second directional derivative threshold. The default is 0.0.
    percentile : float, optional
        percentile of s_max values above which points will be checked. The default is 0.0.
    mode : str, optional
        padding mode. The default is 'constant'.

    Returns
    -------
    ridge_pts : jnp.array
        points satisfying ridge conditions.
    ridge_mask : jnp.array
        boolean array indicating which grid points led to nearby ridge points.

    """
        
    vx = v_max[..., 0]
    vy = v_max[..., 1]
    
    f = neighbor_views(s_max, mode=mode)
    
    fx = (f['east'] - f['west']) / (2 * dx)
    fy = (f['north'] - f['south']) / (2 * dy)
    fxx = (f['east'] - 2 * f['center'] + f['west']) / (dx**2)
    fyy = (f['north'] - 2 * f['center'] + f['south']) / (dy**2)
    fxy = (f['ne'] - f['se'] - f['nw'] + f['sw']) / (4 * dx * dy)
    
    c2 = vx * (fxx * vx + fxy * vy) + vy * (fxy * vx + fyy * vy)
    numerator = -(fx * vx + fy * vy)
    t = jnp.where(c2 != 0, numerator / c2, 0.0)
    tvx = t * vx
    tvy = t * vy
    ridge_pts_ = jnp.stack(X, Y, axis=-1) + jnp.stack(tvx, tvy, axis=-1)
    
    ridge_mask = (f['center'] > percentile) \
                 & (c2 < -sdd_thresh) \
                 & (jnp.abs(tvx) <= dx / 2) \
                 & (jnp.abs(tvy) <= dy / 2)
                 
    ridge_pts = jnp.where(ridge_mask[..., None], ridge_pts_, jnp.nan)               
    
    return ridge_pts, ridge_mask
    
    
    
    
    
