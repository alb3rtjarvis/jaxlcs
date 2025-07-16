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

# can compute series of DF's with either version, one may be faster depending on hardware,
# problem size, and chunk size

# will loop over t0span
DF_t0span = jlcs.flowmap_loop(jac_fn, t0span, Y0, T, dt0, chunk_t0=False)
# will loop over t0span chunks
DF_t0_chunk = jlcs.flowmap_loop(
    jac_fn_chunk, t0_chunk, Y0, T, dt0, chunk_t0=True
)

# compute FTLE for full t0span
ftle_t0span = jlcs.ftle(DF_t0span, T)

# same for chunked version
ftle_t0_chunk = jlcs.ftle(DF_t0_chunk, T)