# This module largely re-implements interpax code for our specific use cases so we can 
# compute memory efficient jacobians of cubic interpolants in jax. When tracing through cubic
# interpolant logic embedded in ode solves for performing AD, generating and evaluating the
# computational graph can get out of hand. To get around this, we implement custom_jvp and
# custom_vjp rules in the _vector_field module so the interpolant logic is not part of the
# computational graph.

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float
from typing import Tuple
from diffrax._custom_types import FloatScalarLike
from interpax._coefs import A_TRICUBIC
from interpax.utils import asarray_inexact
from interpax._spline import AbstractInterpolator
from collections import OrderedDict


def _calculate_coefficients(f, derivs, i, j, k, dx, dy, dz):
    """Calculates the 64 tricubic spline coefficients for a grid cell."""
    fs = OrderedDict()
    fs["f"] = f
    fs["fx"] = derivs["fx"]
    fs["fy"] = derivs["fy"]
    fs["fz"] = derivs["fz"]
    fs["fxy"] = derivs["fxy"]
    fs["fxz"] = derivs["fxz"]
    fs["fyz"] = derivs["fyz"]
    fs["fxyz"] = derivs["fxyz"]
    
    fsq = OrderedDict()
    for ff in fs.keys():
        for kk in [0, 1]:
            for jj in [0, 1]:
                for ii in [0, 1]:
                    s = ff + str(ii) + str(jj) + str(kk)
                    fsq[s] = fs[ff][i - 1 + ii, j - 1 + jj, k - 1 + kk]
                    if "x" in ff:
                        fsq[s] = (dx * fsq[s].T).T
                    if "y" in ff:
                        fsq[s] = (dy * fsq[s].T).T
                    if "z" in ff: 
                        fsq[s] = (dz * fsq[s].T).T
    
    F = jnp.stack(list(fsq.values()), axis=0).T
    coef = jnp.vectorize(jnp.matmul, signature="(n,n),(n)->(n)")(A_TRICUBIC, F).T
    return jnp.moveaxis(coef.reshape((4, 4, 4, *coef.shape[1:]), order="F"), 3, 0)

@jax.jit
def _fast_cubic_jac_kernel(xq, yq, zq, x, y, z, fx0, derivs0, fx1, derivs1):
    """Kernal function to compute interpolant values and jacobian for custom_vjp and custom_jvp."""
    xq, yq, zq, x, y, z, fx0, fx1 = map(asarray_inexact, (xq, yq, zq, x, y, z, fx0, fx1))
    
    xq, yq, zq = jnp.broadcast_arrays(xq, yq, zq)


    # Promote scalar query points to 1D array.
    # Note this is done after the computation of outshape
    # to make jax.grad work in the scalar case.
    xq, yq, zq = map(jnp.atleast_1d, (xq, yq, zq))

    i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
    j = jnp.clip(jnp.searchsorted(y, yq, side="right"), 1, len(y) - 1)
    k = jnp.clip(jnp.searchsorted(z, zq, side="right"), 1, len(z) - 1)

    dx = x[i] - x[i - 1]
    deltax = xq - x[i - 1]
    dxi = jnp.where(dx == 0, 0, 1 / dx)
    tx = deltax * dxi

    dy = y[j] - y[j - 1]
    deltay = yq - y[j - 1]
    dyi = jnp.where(dy == 0, 0, 1 / dy)
    ty = deltay * dyi

    dz = z[k] - z[k - 1]
    deltaz = zq - z[k - 1]
    dzi = jnp.where(dz == 0, 0, 1 / dz)
    tz = deltaz * dzi
    
    coef0 = _calculate_coefficients(fx0, derivs0, i, j, k, dx, dy, dz)
    coef1 = _calculate_coefficients(fx1, derivs1, i, j, k, dx, dy, dz)
    
    ttx = _get_t_der(tx, 0, dxi)
    
    tty = _get_t_der(ty, 0, dyi)
    ttdy = _get_t_der(ty, 1, dyi)
    
    ttz = _get_t_der(tz, 0, dzi)
    ttdz = _get_t_der(tz, 1, dzi)
    
    primal0 = jnp.einsum("lijk...,li,lj,lk->l...", coef0, ttx, tty, ttz)
    primal1 = jnp.einsum("lijk...,li,lj,lk->l...", coef1, ttx, tty, ttz)
    
    primal_out = jnp.array([primal0, primal1]).squeeze()
    
    J00 = jnp.einsum("lijk...,li,lj,lk->l...", coef0, ttx, ttdy, ttz)
    J01 = jnp.einsum("lijk...,li,lj,lk->l...", coef0, ttx, tty, ttdz)
    J10 = jnp.einsum("lijk...,li,lj,lk->l...", coef1, ttx, ttdy, ttz)
    J11 = jnp.einsum("lijk...,li,lj,lk->l...", coef1, ttx, tty, ttdz)
    
    jac = jnp.array([[J00, J01], [J10, J11]]).squeeze()

    return jnp.atleast_1d(primal_out), jnp.atleast_2d(jac)

@jax.jit
def _fast_cubic_curl_kernel(xq, yq, zq, x, y, z, fx0, derivs0, fx1, derivs1):
    """Kernal function to compute interpolant values and curl for custom_vjp and custom_jvp."""
    xq, yq, zq, x, y, z, fx0, fx1 = map(asarray_inexact, (xq, yq, zq, x, y, z, fx0, fx1))
    
    xq, yq, zq = jnp.broadcast_arrays(xq, yq, zq)


    # Promote scalar query points to 1D array.
    # Note this is done after the computation of outshape
    # to make jax.grad work in the scalar case.
    xq, yq, zq = map(jnp.atleast_1d, (xq, yq, zq))

    i = jnp.clip(jnp.searchsorted(x, xq, side="right"), 1, len(x) - 1)
    j = jnp.clip(jnp.searchsorted(y, yq, side="right"), 1, len(y) - 1)
    k = jnp.clip(jnp.searchsorted(z, zq, side="right"), 1, len(z) - 1)

    dx = x[i] - x[i - 1]
    deltax = xq - x[i - 1]
    dxi = jnp.where(dx == 0, 0, 1 / dx)
    tx = deltax * dxi

    dy = y[j] - y[j - 1]
    deltay = yq - y[j - 1]
    dyi = jnp.where(dy == 0, 0, 1 / dy)
    ty = deltay * dyi

    dz = z[k] - z[k - 1]
    deltaz = zq - z[k - 1]
    dzi = jnp.where(dz == 0, 0, 1 / dz)
    tz = deltaz * dzi
    
    coef0 = _calculate_coefficients(fx0, derivs0, i, j, k, dx, dy, dz)
    coef1 = _calculate_coefficients(fx1, derivs1, i, j, k, dx, dy, dz)
    
    ttx = _get_t_der(tx, 0, dxi)
    
    tty = _get_t_der(ty, 0, dyi)
    ttdy = _get_t_der(ty, 1, dyi)
    
    ttz = _get_t_der(tz, 0, dzi)
    ttdz = _get_t_der(tz, 1, dzi)
    
    J01 = jnp.einsum("lijk...,li,lj,lk->l...", coef0, ttx, tty, ttdz)
    J10 = jnp.einsum("lijk...,li,lj,lk->l...", coef1, ttx, ttdy, ttz)
    
    curl = (J10 - J01).squeeze()

    return jnp.atleast_1d(curl)

@jax.jit
def _get_t_der(t: jax.Array, derivative: int, dxi: jax.Array):
    """Get arrays of [1,t,t^2,t^3] for cubic interpolation."""
    t0 = jnp.zeros_like(t)
    t1 = jnp.ones_like(t)
    dxi = jnp.atleast_1d(dxi)[:, None]
    # derivatives of monomials
    d0 = lambda: jnp.array([t1, t, t**2, t**3]).T * dxi**0
    d1 = lambda: jnp.array([t0, t1, 2 * t, 3 * t**2]).T * dxi

    return jax.lax.switch(derivative, [d0, d1])


def fast_cubic_jac(
    tq: Float[Array, " Nq"],
    xq: Float[Array, " Nq"],
    yq: Float[Array, " Nq"],
    cubic_obj_x: type[AbstractInterpolator],
    cubic_obj_y: type[AbstractInterpolator]
) -> Tuple[Float[Array, " Nq 2"], Float[Array, " Nq 2 2"]]:
    """
    Computes the jacobian of the function defined by f = (cubic_obj_x, cubic_obj_y) @ (tq, xq, yq)
    where derivatives are computed with respect to the spatial coordinates (x, y). Wrapper for the 
    function _fast_cubic_jac_kernel that unpacks the interpolant objects so the core logic can be 
    compiled.

    Parameters
    ----------
    tq : float or jnp.array
        t query point(s).
    xq : float or jnp.array
        x query point(s).
    yq : float or jnp.array
        y query point(s).
    cubic_obj_x : AbstractInterpolator
        x-cubic interp object.
    cubic_obj_y : AbstractInterpolator
        y-cubic interp object.

    Returns
    -------
    tuple
        tuple containing primal and jacobian.

    """
    t = cubic_obj_x.x
    x = cubic_obj_x.y
    y = cubic_obj_x.z
    fx = cubic_obj_x.f
    derivs_x = cubic_obj_x.derivs
    fy = cubic_obj_y.f
    derivs_y = cubic_obj_y.derivs

    return _fast_cubic_jac_kernel(tq, xq, yq, t, x, y, fx, derivs_x, fy, derivs_y)

def fast_cubic_curl(
    tq: Float[Array, " Nq"],
    xq: Float[Array, " Nq"],
    yq: Float[Array, " Nq"],
    cubic_obj_x: type[AbstractInterpolator],
    cubic_obj_y: type[AbstractInterpolator]
) -> Float[Array, " Nq"]: 
    """
    Computes the curl of the function defined by f = (cubic_obj_x, cubic_obj_y) @ (tq, xq, yq)
    Wrapper for the function _fast_cubic_curl_kernel that unpacks the interpolant objects so the 
    core logic can be compiled.

    Parameters
    ----------
    tq : float or jnp.array
        t query point(s).
    xq : float or jnp.array
        x query point(s).
    yq : float or jnp.array
        y query point(s).
    cubic_obj_x : AbstractInterpolator
        x-cubic interp object.
    cubic_obj_y : AbstractInterpolator
        y-cubic interp object.

    Returns
    -------
    jnp.array
        curl value(s).

    """
    t = cubic_obj_x.x
    x = cubic_obj_x.y
    y = cubic_obj_x.z
    fx = cubic_obj_x.f
    derivs_x = cubic_obj_x.derivs
    fy = cubic_obj_y.f
    derivs_y = cubic_obj_y.derivs

    return _fast_cubic_curl_kernel(tq, xq, yq, t, x, y, fx, derivs_x, fy, derivs_y)