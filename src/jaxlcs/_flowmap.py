import jax
import diffrax

def flowmap(
        term,
        solver=diffrax.Dopri8(),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5), 
        autodiff=1,
        val_and_jac=False,
        vmap_y0=True,
        chunck_t0=False,
        jit_compile=True
    ):
    """
    A flexible wrapper for Diffrax.diffeqsolve, returns a function for computing either the final 
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
    TYPE
        DESCRIPTION.

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
        
    match (vmap_y0, chunck_t0):
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
        
def flowmap_d(
        term,
        solver=diffrax.Dopri8(),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5),
        n=100,
        val_and_jac=False,
        vmap_y0=True,
        chunck_t0=False,
        jit_compile=True
    ):
    """
    A flexible wrapper for Diffrax.diffeqsolve, returns a function for computing the 
    full trajectory initialized at t0, y0 for time T that
    returns the dense solution which can be evaluated with sol.evaluate. 
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
        a function that takes args t0, y0, T, dt0 and returns either final position of jacobian.

    """
    
    def _fm(t0, y0, T, dt0):
        saveat = diffrax.SaveAt(ts = jax.numpy.linspace(t0, t0 + T, n))
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0, 
            t0 + T,
            dt0,
            y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat
        )
        return sol
        
    match (vmap_y0, chunck_t0):
        case (True, True):
            vmap_over_y0 = jax.vmap(_fm, in_axes=(None, 0, None, None))
            chunk_over_t0 = jax.vmap(vmap_over_y0, in_axes=(0, None, None, None))
            
            return jax.jit(chunk_over_t0) if jit_compile else chunk_over_t0
        
        case (True, False):
            vmap_over_y0 = jax.vmap(_fm, in_axes=(None, 0, None, None))
            
            return jax.jit(vmap_over_y0) if jit_compile else vmap_over_y0
        
        case (False, True):
            # This case is not recommended in general! This will essentially vmap over only t0.
            # It is recommended if only vmapping over one arg, it should be the larger batch, y0.
            vmap_over_t0 = jax.vmap(_fm, in_axes=(0, None, None, None))
            
            return jax.jit(vmap_over_t0) if jit_compile else vmap_over_t0
        
        case (False, False):
            
            return jax.jit(_fm) if jit_compile else _fm        
            
            
            