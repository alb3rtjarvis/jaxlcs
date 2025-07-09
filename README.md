# JAXLCS

JAXLCS is a Python package for accurately and efficiently computing Lagrangian Coherent Structures by leveraging the power of JAX for automatic differentiation, just-in-time compilation, and auto-vectorization on GPUs. 

## The idea

At the heart of most coherent structure methods is obtaining derivatives of trajectories with respect to initial conditions. Traditionally, this is done using finite differencing. In some cases, this is okay (FTLE), but in others, these derivatives need to be very accurate (variational LCS, shape coherent sets, etc.). In these cases, finite differencing can lead to unacceptable errors, causing the user to re-run the very expensive particle integration with different auxiliary grid spacing to find a suitable value. By using automatic differentiation and leveraging methods made popular in the neural ODE world, we can accurate derivatives of trajectories (up to machine precision and the accuracy of our solver) without the need to tune this potentially expensive parameter. For more information, refer to chapter 5 of my dissertation, [*Numerical and Theoretical Developments for Coherent Structures*](https://hdl.handle.net/10919/134980). A dedicated paper is coming soon.

## Basic Usage
```python
import jax.numpy as jnp
import jaxlcs as jlcs
import diffrax

# Create data for a vector field, we could just use the analytical vector field but create
# data for demonstration

def dg(tq, xq, yq):
    A, eps, omega = jnp.array([0.1, 0.25, 0.2*jnp.pi])
    a = eps*jnp.sin(omega*tq)
    y = jnp.array([xq, yq])
    b = 1 - 2*a
    f = a*y[0]**2 + b*y[0]
    dfdx = 2*a*y[0] + b
    dx = -jnp.pi*A*jnp.sin(jnp.pi*f)*jnp.cos(jnp.pi*y[1])
    dy = jnp.pi*A*jnp.cos(jnp.pi*f)*jnp.sin(jnp.pi*y[1])*dfdx
    return jnp.array([dx, dy])

t = jnp.linspace(0.0, 10.0, 51)
x = jnp.linspace(0.0, 2.0, 201)
y = jnp.linspace(0.0, 1.0, 101)

Tp, Xp, Yp = jnp.meshgrid(t, x, y, indexing='ij')

up, vp = dg(Tp, Xp, Yp)

ad = 1   # we will use forward-mode AD
ode = jlcs.vector_field(t, x, y, up, vp, ad=ad)   # obtain the interpolated vector field
term = diffrax.ODETerm(ode)   # convert to diffrax.ODETerm
solver = diffrax.Dopri8()
t0, T = 0.0, 5.0   # initial time and integration time
dt0 = 1e-2   # initial time spacing for solver
# tolerances for adaptive solver, these are the defaults 
stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-5)   

y0 = jnp.array([0.25, 0.75])   # an initial point

# get the flowmap function
jac_fn = jlcs.flowmap(term, solver, stepsize_controller, ad=ad)

# compute the jacobian at and w.r.t. y0
J = jac_fn(t0, y0, T, dt0)

# vectorize and compute for whole grid
batched_jac_fn = jax.vmap(jac_fn)
ic = jnp.array([X.ravel(), Y.ravel()])
J_ic = batched_jac_fn(ic)

# Compute FTLE
ftle = jlcs.ftle_from_DF(J_ic)



```