import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, Kvaerno3, Kvaerno4, Kvaerno5, ODETerm, SaveAt, PIDController
from pylinger_jax_utils import get_grho_and_adotrad

def dtauda(a, tau, args):
    grhom, grhog, grhor, adotrad, Omegam, OmegaL = args
    grho2 = grhom * Omegam * a + (grhog + 3.0 * grhor) + grhom * OmegaL * a ** 4
    return jnp.sqrt(3.0 / grho2)


def compute_tau(*, Omegam: float, OmegaL: float, H0: float, Tcmb: float,
                rtol: float = 1e-5, atol: float = 1e-5, order: int = 5):
    grhom, grhog, grhor, adotrad = get_grho_and_adotrad(H0, Tcmb)

    amin = 0.0
    amax = 1.1
    a = jnp.geomspace(1e-9, amax, 1000)
    a = a.at[0].set(amin)
    tauini = 0.0
    term = ODETerm(dtauda)
    if order == 5:
        solver = Kvaerno5()
    elif order == 4:
        solver = Kvaerno4()
    elif order == 3:
        solver = Kvaerno3()
    else:
        raise NotImplementedError
    saveat = SaveAt(ts=a)
    stepsize_controller = PIDController(rtol=rtol, atol=atol)
    sol = diffeqsolve(term, solver, t0=amin, t1=amax, dt0=1e-5 * (a[1] - a[0]), y0=tauini, saveat=saveat,
                       stepsize_controller=stepsize_controller, args=(grhom, grhog, grhor, adotrad, Omegam, OmegaL))
    return sol.ts, sol.ys

