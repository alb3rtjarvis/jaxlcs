#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 17:53:53 2025

@author: ajarvis
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

def _ensure_continuity(current_vec, reference_vec):
    """Ensure current_vec is continious with reference_vec"""
    return lax.cond(
        jnp.dot(current_vec, reference_vec) > 0.0,
        lambda vec: vec,
        lambda vec: -vec,
        current_vec
        )

def _in_bounds(point, x_bounds, y_bounds):
    """Check if point is in bounds (x_bounds, y_bounds)"""
    xq, yq = point
    x_in = (xq >= x_bounds[0]) & (xq <= x_bounds[1])
    y_in = (yq >= y_bounds[0]) & (yq <= y_bounds[1])
    
    return x_in & y_in

def eigvec_interp(x, y, eigvec_grid):
    """
    Compute linear interpolant for eigenvectors defined on a grid (x, y) and ensure local continuity
    is enforced.

    Parameters
    ----------
    x : jnp.array, shape=(nx,)
        x values.
    y : jnp.array, shpae=(ny,)
        y values.
    eigvec_grid : jnp.array, shape=(nx, ny, 2)
        eigenvectors on grid defined by (x, y).

    Returns
    -------
    Callable
        interpolator function for eigvec_grid.

    """
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # interpolator function
    def _interpolator(query_point):
        xq, yq = query_point
        
        # find bottom-left index of grid-cell that (xq, yq) is in
        i = jnp.clip(jnp.searchsorted(x, xq, side='right') - 1,  0, len(x) - 2)
        j = jnp.clip(jnp.searchsorted(y, yq, side='right') - 1,  0, len(y) - 2)
        
        # bottom-left point of grid-cell
        x0 = x[i]
        y0 = y[j]
        
        # get eigenvector values at each gridpoint of the cell
        v00 = eigvec_grid[i, j]
        v01 = eigvec_grid[i, j + 1]
        v10 = eigvec_grid[i + 1, j]
        v11 = eigvec_grid[i + 1, j + 1]
        
        # make sure all grid eigenvectors are continuous
        v01 = _ensure_continuity(v01, v00)
        v10 = _ensure_continuity(v10, v00)
        v11 = _ensure_continuity(v11, v00)
        
        # fractional distance (xq, yq) is from (x0, y0)
        tx = (xq - x0) / dx
        ty = (yq - y0) / dy
        
        # x and y interpolants
        v_bottom = (1 - tx) * v00 + tx * v10
        v_top = (1 - tx) * v01 + tx * v11
        interp_vec = (1 - ty) * v_bottom + ty * v_top
        
        # ensure result has norm 1 or 0 (if the vector is the zero vector).
        norm = jnp.linalg.norm(interp_vec)
        safe_norm = jnp.where(norm == 0, 1.0, norm)
        return interp_vec / safe_norm
    
    return _interpolator


@partial(jax.jit, static_argnames=['eigvec_fun', 'alpha_scaling', 'eigval_max_interp'])
def rk4_tensorlines(
        eigvec_fun, 
        y0, 
        t0, 
        tf, 
        num_steps, 
        x_bounds, 
        y_bounds, 
        alpha_scaling=True, 
        eigval_max_interp=None
):
    
    if alpha_scaling:
        if eigval_max_interp is None:
            raise ValueError("'eigval_max_interp' must be provided when alpha_scaling=True")
        
        # scaled eigvec function
        def _scaled_eigvecs(point):
            eigval_max = eigval_max_interp(point)
            eigval_min = 1 / eigval_max
            
            alpha = ((eigval_max - eigval_min) / (eigval_max + eigval_min))**2
            
            return alpha * eigvec_fun(point)
        
        f = _scaled_eigvecs
        
    else:
        
        f = eigvec_fun
        
    h = (tf - t0) / num_steps
    
    # while loop condition
    def cond_fun(state):
        
        _, _, step, is_valid, _ = state
        return is_valid & (step < num_steps)
    
    # while loop body
    def body_fun(state):
        
        # unpack previous state
        y_prev, k_prev, step, _, incoming_buffer = state
        
        # rk4 logic
        k1 = _ensure_continuity(f(y_prev), k_prev)
        
        yk2 = y_prev + 0.5 * h * k1
        k2 = _ensure_continuity(f(yk2), k1)
        
        yk3 = y_prev + 0.5 * h * k2
        k3 = _ensure_continuity(f(yk3), k2)
        
        yk4 = y_prev + h * k3
        k4 = _ensure_continuity(f(yk4), k3)
        
        # rk4 step
        y_next = y_prev + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # make sure y_next is in domain
        is_valid = _in_bounds(y_next, x_bounds, y_bounds)
        
        # update buffer
        outgoing_buffer = incoming_buffer.at[step + 1].set(y_next)
        
        return y_next, k4, step + 1, is_valid, outgoing_buffer
    
    # initialize buffer and initial vector
    initial_buffer = jnp.full((num_steps + 1, 2), y0)
    k0 = f(y0)
    
    # define initial state for integrating "forward"
    init_state = (y0, k0, 0, True, initial_buffer)
    _, _, final_step_fwd, _, final_results_buffer_fwd = lax.while_loop(
        cond_fun, body_fun, init_state
    )

    # define initial state for integrating "backward"
    init_state = (y0, -k0, 0, True, initial_buffer)
    _, _, final_step_bwd, _, final_results_buffer_bwd = lax.while_loop(
        cond_fun, body_fun, init_state
    )
    
    inds = jnp.arange(num_steps + 1)
    
    # set array values to nan if integration terminated due leaving the domain 
    mask_fwd = inds > final_step_fwd
    final_results_fwd = jnp.where(mask_fwd[:, None], jnp.nan, final_results_buffer_fwd)

    mask_bwd = inds > final_step_bwd
    final_results_bwd = jnp.where(mask_bwd[:, None], jnp.nan, final_results_buffer_bwd)[::-1]
    
    # combine results into full tensorline
    final_results = jnp.concatenate((final_results_bwd[:-1], final_results_fwd))
    return final_results
           


# @partial(jax.jit, static_argnames='f')
# def rk4_tensorlines(f, t0, y0, tf, num_steps):
    
#     h = (tf - t0) / num_steps

    
#     def scan_body(carry, _):
        
#         yi, ki = carry
        
#         k1 = _ensure_continuity(f(yi), ki)
        
#         yk2 = yi + 0.5 * h * k1
#         k2 = _ensure_continuity(f(yk2), k1)
        
#         yk3 = yi + 0.5 * h * k2
#         k3 = _ensure_continuity(f(yk3), k2)
        
#         yk4 = yi + h * k3
#         k4 = _ensure_continuity(f(yk4), k3)
        
#         y_next = yi + (h / 6) * (k1 + 2 * k2 + 2* k3 + k4)
        
#         return (y_next, k4), y_next

#     init_carry = (y0, f(y0))
#     _, ys = lax.scan(scan_body, init_carry, length=num_steps)
#     ts = jnp.linspace(t0, tf, num_steps)
    
#     return ts, ys
           
        