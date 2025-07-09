import jax
import jax.numpy as jnp
from interpax import Interpolator3D
from functools import partial
from ._cubic import fast_cubic_jac
    
def vector_field(t, x, y, u, v, autodiff=1):
    """
    

    Parameters
    ----------
    t : jnp.array
        array containing t-values.
    x : jnp.array
        array containing x-values.
    y : jnp.array
        array containing y-values.
    u : jnp.array
        array containing u-values.
    v : jnp.array
        array containing v-values.
    autodiff : int, optional
        int determining if autodiff will be performed. 
        1 for forward-mode AD, -1 for reverse-mode AD, and 0 for no AD. The default is 1.

    Raises
    ------
    ValueError
        error raised if autodiff not in (1, 0, -1).

    Returns
    -------
    callable
        returns function for interpolant of velocity field.

    """
    
    ui = Interpolator3D(t, x, y, u, 'cubic')
    vi = Interpolator3D(t, x, y, v, 'cubic')
    
    if autodiff == 1:        
        @partial(jax.custom_jvp, nondiff_argnums=(3, 4))
        def _cubic_interp_custom_jvp(t, x, y, fx, fy):
            uq = fx(t, x, y)
            vq = fy(t, x, y)
        
            return jnp.array([uq, vq])
        
        def _cubic_jvp(nondiff_arg0, nondiff_arg1, primals, tangents):
            
            fx, fy = nondiff_arg0, nondiff_arg1
            t, x, y = primals
            _, x_dot, y_dot = tangents
            
            primal_out, J = fast_cubic_jac(t, x, y, fx, fy)
            
            tangent_out = jnp.matmul(J, jnp.array([x_dot, y_dot]))
            
            return primal_out, tangent_out
        
        _cubic_interp_custom_jvp.defjvp(_cubic_jvp)
    
        def cubic_eval_fwd(t, y, args=None):
            
            return _cubic_interp_custom_jvp(t, y[0], y[1], ui, vi)
        
        return cubic_eval_fwd
    
    elif autodiff == -1:      
        @partial(jax.custom_vjp, nondiff_argnums=(3, 4))
        def _cubic_interp_custom_vjp(t, x, y, fx, fy):
            uq = fx(t, x, y)
            vq = fy(t, x, y)
            return jnp.array([uq, vq])
        
        def _cubic_fwd(t, x, y, fx, fy):
            
            primal_out, jac = fast_cubic_jac(t, x, y, fx, fy)
            residuals = jac
            
            return primal_out, residuals
        
        def _cubic_rev(nondiff_arg0, nondiff_arg1, residuals, g):
            
            J = residuals            
            grad = jnp.matmul(g, J)
            
            return (None, grad[0], grad[1])
        
        _cubic_interp_custom_vjp.defvjp(_cubic_fwd, _cubic_rev)
        
        def cubic_eval_rev(t, y, args=None):
            
            return _cubic_interp_custom_vjp(t, y[0], y[1], ui, vi)
        
        return cubic_eval_rev
    
    elif autodiff == 0:        
        def cubic_eval(tq, yq, args=None):
            uq = ui(tq, yq[0], yq[1])
            vq = vi(tq, yq[0], yq[1])
            return jnp.array([uq, vq])  
        
        return cubic_eval
    
    else:
        raise ValueError(
            "autodiff must be set to 1 (for forward AD), -1 (for backward AD), or 0 (for no AD)"
        )
        
def vector_field_linear(t, x, y, u, v):
    ui = jax.scipy.interpolate.RegularGridInterpolator((t, x, y), u, fill_value=0.0)
    vi = jax.scipy.interpolate.RegularGridInterpolator((t, x, y), v, fill_value=0.0)
    
    def interp_eval(t, y, args=None):
        uq = ui(jnp.array([t, y[0], y[1]]))[0]
        vq = vi(jnp.array([t, y[0], y[1]]))[0]
        return jnp.array([uq, vq]) 
    
    return interp_eval