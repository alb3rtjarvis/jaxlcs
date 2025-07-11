# JAXLCS

JAXLCS is a Python package for accurately and efficiently computing Lagrangian Coherent Structures by leveraging the power of JAX for automatic differentiation, just-in-time compilation, and auto-vectorization on GPUs (and TPUs). This package is meant to have some overlap with, and eventually grow into the GPU version of [Numbacs](https://github.com/alb3rtjarvis/numbacs). 

**_NOTE:_** JAXLCS is in early stages of development with plans to add more functionality, change much of the structure, and further optimize many methods.

## The idea

At the heart of most coherent structure methods is differentiating final positions of trajectories with respect to initial conditions. Traditionally, this is done using finite differencing. In some cases, this is okay (FTLE), but in others, these derivatives need to be very accurate (variational LCS, shape coherent sets, etc.). In these cases, finite differencing can lead to unacceptable errors, forcing the user to re-run the very expensive particle integration with different auxiliary grid spacing to find a suitable value. By using automatic differentiation and leveraging methods made popular in the neural ODE world, we can obtain accurate derivatives of trajectories (up to machine precision and the accuracy of our solver) without the need to tune this potentially expensive parameter. For more information, refer to chapter 5 of my dissertation, [*Numerical and Theoretical Developments for Coherent Structures*](https://hdl.handle.net/10919/134980). A dedicated paper is coming soon.

## Installation

Coming soon...

## Basic Usage
```python
import jax.numpy as jnp
import jaxlcs as jlcs
import diffrax

# create data for a vector field, we could just use the analytical vector field but create
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
ode = jlcs.vector_field(t, x, y, up, vp, autodiff=ad)   # obtain the interpolated vector field
term = diffrax.ODETerm(ode)   # convert to diffrax.ODETerm
solver = diffrax.Dopri8()   # use dopri8 solver, this is the default
t0, T = 0.0, 5.0   # initial time and integration time
dt0 = 1e-2   # initial time spacing for solver

# tolerances for adaptive solver, these are the defaults 
stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-5)   

# get the flowmap function, since autodiff=1, it is actually the Jacobian of the flowmap
jac_fn = jlcs.flowmap(
    term, solver, stepsize_controller, autodiff=ad, vmap_y0=True, chunk_t0=False
)

# create initial conditions
Nt, Nx, Ny = 25, 101, 51
t0span = jnp.linspace(0.0, 0.5, 25)
xq = jnp.linspace(0.0, 2.0, Nx)
yq = jnp.linspace(0.0, 1.0, Ny)

# create batched initial spatial conditions
Xq, Yq = jnp.meshgrid(xq, yq, indexing='ij')
Y0 = jnp.array([Xq.ravel(), Yq.ravel()]).T

# can compute for a single t0
DF_t0 = jac_fn(t0, Y0, T, dt0)

# and compute FTLE
ftle_t0 = jlcs.ftle(DF_t0, T)

# can also get a vmapped version over t0
jac_fn_chunk = jlcs.flowmap(
    term, solver, stepsize_controller, autodiff=ad, vmap_y0=True, chunk_t0=True
)

# reshape t0span for chunking
t0_chunk = t0span.reshape(5, -1)

# can compute series of DF's with either version, one may be faster depending on hardware and
# problem size

# will loop over t0span
DF_t0span = jlcs.flowmap_loop(jac_fn, t0span, Y0, T, dt0, chunk_t0=False)
# will loop over t0span chunks
DF_t0_chunk = jlcs.flowmap_loop(
    jac_fn_chunk, t0_chunk, Y0, T, dt0, chunk_t0=True
)
```

##  Similartities and differences with NumbaCS

As mentioned, the plan is for JAXLCS to eventually grow into the GPU version of NumbaCS, though each will have some functionality the other does not. 

**Similarities**: 

- Both are highly performant Python packages making heavy use of just-in-time compilation and parallelization, though how this is tackled under the hood differs.
- Both (eventually JAXLCS will) implement a variety of finite-time coherent structure methods (FTLE, LAVD, FTLE, ridges, variational hyperbolic and elliptic LCS, LAVD-based LCS).
- Both can compute derivatives of trajectories using finite difference methods. 

**Differences**: 

- There are no plans to implement any instantaneous methods in JAXLCS that are available in NumbaCS (iLE, IVD), though this could easily be done if there is demand for it.
- While both can compute derivatives of trajectories using finite differencing, JAXLCS can also compute derivatives using automatic differentiation by taking advantage of the power of JAX and Diffrax. Refer to the cited work to see why this is advantageous for certain methods.
- The most obvious difference, NumbaCS currently only supports CPUs while JAXLCS supports CPUs, GPUs and TPUs (though it is recommended to use NumbaCS on the CPU as it will be faster). 