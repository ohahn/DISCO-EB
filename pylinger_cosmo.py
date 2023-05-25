import numpy as np
from scipy.integrate import solve_ivp


class cosmo:
    def dtauda(self, a, tau):
        grho2 = self.grhom * self.Omegam * a + (self.grhog + 3.0 * self.grhor) + self.grhom * self.OmegaL * a**4
        return np.sqrt(3.0 / grho2)

    def d2tauda2(self, a, tau):
        jac = np.zeros((1, 1))
        grho2 = self.grhom * self.Omegam * a + (self.grhog + 3.0 * self.grhor) + self.grhom * self.OmegaL * a**4
        jac[0, 0] = -0.5 * self.grhom * (4 * a**3 * self.OmegaL + self.Omegam) * np.sqrt(3.0) / grho2**1.5
        return jac

    def __init__(self, *, Omegam: float, Omegab: float, OmegaL: float, H0: float, Tcmb: float, YHe: float):
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
        a = np.geomspace(1e-9, amax, 1000)
        a[0] = amin
        tauini = 0.0
        sol = solve_ivp(self.dtauda, (amin, amax), [tauini], jac=self.d2tauda2, method="BDF", t_eval=a)

        self.a = np.array(sol["t"])
        self.tau = np.array(sol["y"][0])

    def get_tau(self, a: float):
        return np.interp(a, self.a, self.tau)

    def get_a(self, tau: float):
        return np.interp(tau, self.tau, self.a)
