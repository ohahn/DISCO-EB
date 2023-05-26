import jax.numpy as jnp
from diffrax import diffeqsolve, Kvaerno3, Kvaerno4, Kvaerno5, ODETerm, SaveAt, PIDController


class cosmo:
    def dtauda(self, a, tau, args):
        grho2 = self.grhom * self.Omegam * a + (self.grhog + 3.0 * self.grhor) + self.grhom * self.OmegaL * a**4
        return jnp.sqrt(3.0 / grho2)

    def __init__(self, *, Omegam: float, Omegab: float, OmegaL: float, H0: float, Tcmb: float, YHe: float,
                 rtol: float=1e-5, atol: float=1e-5, order: int=5):
        self.Omegam = Omegam
        self.Omegab = Omegab
        self.OmegaL = OmegaL
        self.H0 = H0
        self.Tcmb = Tcmb
        self.YHe = YHe
        self.grhom = 3.3379e-11 * H0**2
        self.grhog = 1.4952e-13 * Tcmb**4
        self.grhor = 3.3957e-14 * Tcmb**4
        self.adotrad = 2.8948e-7 * Tcmb**2
        self.Omegac = Omegam - Omegab

        amin = 0.0
        amax = 1.1
        a = jnp.geomspace(1e-9, amax, 1000)
        a = a.at[0].set(amin)
        tauini = 0.0
        term = ODETerm(self.dtauda)
        if order == 5:
            solver = Kvaerno3()
        elif order == 4:
            solver = Kvaerno4()
        elif order == 3:
            solver = Kvaerno3()
        else:
            raise NotImplementedError
        saveat = SaveAt(ts=a)
        stepsize_controller = PIDController(rtol=rtol, atol=atol)
        sol = diffeqsolve(term, solver, t0=amin, t1=amax, dt0=1e-5 * (a[1]-a[0]), y0=tauini, saveat=saveat,
                          stepsize_controller=stepsize_controller)
        self.a = sol.ts
        self.tau = sol.ys

    def get_tau(self, a: float):
        return jnp.interp(a, self.a, self.tau)

    def get_a(self, tau: float):
        return jnp.interp(tau, self.tau, self.a)
