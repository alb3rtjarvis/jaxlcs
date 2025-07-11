from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from diffrax._custom_types import FloatScalarLike

@jax.jit
def ftle(DF: Float[Array, "... 2 2"], T: FloatScalarLike) -> Float[Array, "..."]:

    dxdx = DF[..., 0, 0]
    dxdy = DF[..., 0, 1]
    dydx = DF[..., 1, 0]
    dydy = DF[..., 1, 1]
    
    a1 = dxdx**2 + dydx**2
    a2 = dxdx*dxdy + dydx*dydy
    a3 = dxdy**2 + dydy**2    
    
    eigval_max = 0.5*(a1 + a3 + jnp.sqrt((a1 - a3)**2 + 4*a2**2))
    
    safe_eigval = jnp.maximum(eigval_max, 1.0)
    
    return (1 / (2 * jnp.abs(T))) * jnp.log(safe_eigval)
