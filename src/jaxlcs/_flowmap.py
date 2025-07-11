# This module wraps diffrax.diffeqsolve for convinience, with simple flags for autodiff, vmapping
# (over t0 and y0), and just-in-time compilation.
from typing import Literal, Union
from collections.abc import Callable
from jaxtyping import Float, Array, ArrayLike
from diffrax._custom_types import FloatScalarLike
import jax
import jax.numpy as jnp
import diffrax
from diffrax import AbstractTerm, AbstractSolver, AbstractStepSizeController

def flowmap(
    term: AbstractTerm,
    solver: AbstractSolver = diffrax.Dopri8(),
    stepsize_controller: AbstractStepSizeController = diffrax.PIDController(rtol=1e-3, atol=1e-5), 
    autodiff: Literal[-1, 0, 1] = 1,
    val_and_jac: bool = False,
    vmap_y0: bool = True,
    chunk_t0: bool = True,
    jit_compile: bool = True
) -> Callable:
    """
    A flexible wrapper for diffrax.diffeqsolve, returns a function for computing either the final 
    position of trajectory initialized at t0, y0 for time T (if autodiff == 0), 
    or computes the Jacobian of the that trajectory with respect to y0 (if autodiff!=0).
    Can return a function that is vmapped over y0 (if vmap_y0 == True), 
    vmapped over t0 (if chunck_t0 == True, only recommended if vmap_y0 == True), 
    jit compiled (if jit_compile == True), or any combination of these options.

    Parameters
    ----------
    term : diffrax.AbstractTerm
        term representing the differential equation to be solved. 
        Consult diffrax docs for more info.
    solver : diffrax.AbstractSolver, optional
        ODE solver used. Consult diffrax docs for more info. The default is diffrax.Dopri8().
    stepsize_controller : diffrax.AbstractStepSizeController, optional
        stepsize controller used for solver. Consult diffrax docs for more info. 
        The default is diffrax.PIDController(rtol=1e-3, atol=1e-5).
    autodiff : int, optional
        int determining if autodiff will be performed. 
        1 for forward-mode AD, -1 for reverse-mode AD, and 0 for no AD. The default is 1.
    val_and_jac : bool, optional
        flag determining if jacobian AND value will be returned, only relevant if autodiff != 0. 
        This functionality will be added later. The default is False.        
    vmap_y0 : bool, optional
        flag to determine if returned function will be vmapped over y0. The default is True.
    chunck_t0 : bool, optional
        flag to determine if returned function will be vmapped over t0. The default is True.
    jit_compile : bool, optional
        flag to determine if the returned function will be jit compiled. The default is True.

    Raises
    ------
    ValueError
        error raised if autodiff not in (1, 0, -1).

    Returns
    -------
    callable
        a function that takes args t0, y0, T, dt0 and returns either final position or jacobian.

    """
    if autodiff == 1:
        def _fm(t0, y0, T, dt0):
            sol = diffrax.diffeqsolve(
                term, 
                solver, 
                t0, 
                t0 + T, 
                dt0, 
                y0, 
                stepsize_controller=stepsize_controller,
                adjoint=diffrax.ForwardMode()
            )
            return sol.ys[-1]
        
        solve_func = jax.jacfwd(_fm, argnums=1)        
        
    elif autodiff == -1:
        def _fm(t0, y0, T, dt0):
            sol = diffrax.diffeqsolve(
                term, solver, t0, t0 + T, dt0, y0, stepsize_controller=stepsize_controller
            )
            return sol.ys[-1]
        
        solve_func = jax.jacrev(_fm, argnums=1)
        
    elif autodiff == 0:    
        def _fm(t0, y0, T, dt0):
            sol = diffrax.diffeqsolve(
                term, solver, t0, t0 + T, dt0, y0, stepsize_controller=stepsize_controller
            )
            return sol.ys[-1]
        
        solve_func = _fm        
    
    else:
        raise ValueError(
            "autodiff must be set to 1 (for forward AD), -1 (for backward AD), or 0 (for no AD)"
        )
        
    match (vmap_y0, chunk_t0):
        case (True, True):
            vmap_over_y0 = jax.vmap(solve_func, in_axes=(None, 0, None, None))
            chunk_over_t0 = jax.vmap(vmap_over_y0, in_axes=(0, None, None, None))
            
            return jax.jit(chunk_over_t0) if jit_compile else chunk_over_t0
        
        case (True, False):
            vmap_over_y0 = jax.vmap(solve_func, in_axes=(None, 0, None, None))
            
            return jax.jit(vmap_over_y0) if jit_compile else vmap_over_y0
        
        case (False, True):
            # This case is not recommended in general! This will essentially vmap over only t0.
            # It is recommended if only vmapping over one arg, it should be the larger batch, y0.
            vmap_over_t0 = jax.vmap(solve_func, in_axes=(0, None, None, None))
            
            return jax.jit(vmap_over_t0) if jit_compile else vmap_over_t0
        
        case (False, False):
            
            return jax.jit(solve_func) if jit_compile else solve_func
        
def flowmap_n(
    term: AbstractTerm,
    solver: AbstractSolver = diffrax.Dopri8(),
    stepsize_controller: AbstractStepSizeController = diffrax.PIDController(rtol=1e-3, atol=1e-5), 
    val_and_jac: bool = False,
    vmap_y0: bool = True,
    chunk_t0: bool = True,
    jit_compile: bool = True
) -> Callable:
    """
    A flexible wrapper for diffrax.diffeqsolve, returns a function for computing the 
    full trajectory initialized at t0, y0 for time T, evaluated at ts. 
    Can return a function that is vmapped over y0 (if vmap_y0 == True), 
    vmapped over t0 (if chunck_t0 == True, only recommended if vmap_y0 == True), 
    jit compiled (if jit_compile == True), or any combination of these options.
    Automatic differentiation is not supported for the returned function.

    Parameters
    ----------
    term : diffrax.AbstractTerm
        term representing the differential equation to be solved. 
        Consult diffrax docs for more info.
    solver : diffrax.AbstractSolver, optional
        ODE solver used. Consult diffrax docs for more info. The default is diffrax.Dopri8().
    stepsize_controller : diffrax.AbstractStepSizeController, optional
        stepsize controller used for solver. Consult diffrax docs for more info. 
        The default is diffrax.PIDController(rtol=1e-3, atol=1e-5).
    n : int, optional
        int determining how many points to return the trajectory at. 
    val_and_jac : bool, optional
        flag determining if jacobian AND value will be returned, only relevant if autodiff != 0. 
        This functionality will be added later. The default is False.        
    vmap_y0 : bool, optional
        flag to determine if returned function will be vmapped over y0. The default is True.
    chunck_t0 : bool, optional
        flag to determine if returned function will be vmapped over t0. The default is True.
    jit_compile : bool, optional
        flag to determine if the returned function will be jit compiled. The default is True.

    Raises
    ------
    ValueError
        error raised if autodiff not in (1, 0, -1).

    Returns
    -------
    callable
        a function that takes args t0, y0, T, dt0, ts and returns the trajectory evaluated @ ts.

    """
    
    def _fm(t0, y0, T, dt0, ts):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0, 
            t0 + T,
            dt0,
            y0,
            stepsize_controller=stepsize_controller,
            saveat=diffrax.SaveAt(ts=ts)
        )
        return sol.ys
        
    match (vmap_y0, chunk_t0):
        case (True, True):
            vmap_over_y0 = jax.vmap(_fm, in_axes=(None, 0, None, None, None))
            chunk_over_t0 = jax.vmap(vmap_over_y0, in_axes=(0, None, None, None, None))
            
            return jax.jit(chunk_over_t0) if jit_compile else chunk_over_t0
        
        case (True, False):
            vmap_over_y0 = jax.vmap(_fm, in_axes=(None, 0, None, None, None))
            
            return jax.jit(vmap_over_y0) if jit_compile else vmap_over_y0
        
        case (False, True):
            # This case is not recommended in general! This will essentially vmap over only t0.
            # It is recommended if only vmapping over one arg, it should be the larger batch, y0.
            vmap_over_t0 = jax.vmap(_fm, in_axes=(0, None, None, None, None))
            
            return jax.jit(vmap_over_t0) if jit_compile else vmap_over_t0
        
        case (False, False):
            
            return jax.jit(_fm) if jit_compile else _fm        
           
def flowmap_loop(
    fm_func: Callable[
        [Float[ArrayLike, "..."], 
         Float[Array, "... 2"], 
         FloatScalarLike, 
         FloatScalarLike], 
        Float[Array, "..."]
    ],
    t0vals: Float[Array, "..."], 
    y0: Float[Array, "... 2"], 
    T: FloatScalarLike, 
    dt0: FloatScalarLike, 
    chunk_t0: bool = True
) -> Union[Float[Array, "... 2"], Float[Array, "... 2 2"]]:
    """
    A simple function for looping over the flowmap function fm_func for a series of t0vals.
    This will work if t0vals are chuncked (i.e., t0vals.shape=(c, nt0 // c), 
    in this case you would set chuck_t0=True in the flowmap function)
    or if you are simply looping over a jnp.array of t0vals (i.e., t0vals.shape=(nt0,)).
    In both cases, the returned array will have shape=(nt0, ny0, 2) if no AD is performed,
    and shape=(nt0, ny0, 2, 2) if AD is performed.

    Parameters
    ----------
    fm_func : callable
        a function returned by `jlcs.flowmap`.
        Expected to accept:
          - `t0`: scalar float or JAX array, shape=(nt0,)
          - `y0`: JAX array of shape=(ny0, 2)
          - `T`: scalar float (integration time)
          - `dt0`: scalar float (initial timestep)
        Returns a JAX array of shape `(nt0, ny0, 2)` or `(nt0, ny0, 2, 2)`.     
    t0vals : jnp.array, shape=(nt0, ) or shape=(c, nt0 / c)
        array of t0 values to be looped over.
    y0 : jnp.array, shape=(ny0, 2)
        array of y0 values.
    T : float
        scalar float (integration time).
    dt0 : float
        scalar float (initial timestep).

    Returns
    -------
    jnp.array, shape=(nt0, ny0, 2) or shape=(nt0, ny0, 2, 2) 
        array containing flowmap or jacobian values for all t0vals.

    """
    
    flowmaps = []
    for t0 in t0vals:
        flowmaps.append(fm_func(t0, y0, dt0, T))
    
    flowmaps_arr = jnp.stack(flowmaps)
    
    if chunk_t0:
        fm_shape = flowmaps_arr.shape
        flowmaps_arr = flowmaps_arr.reshape((fm_shape[0] * fm_shape[1], ) + fm_shape[2:])
        
    return flowmaps_arr
        
            