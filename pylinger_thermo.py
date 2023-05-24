import numpy as np
from numba import njit

from pylinger_cosmo import cosmo


class thermal_history:
    # @njit
    def ionize(self, tempb: float, a: float, adot: float, dtau: float, xe: float) -> float:
        # ... switch for fully implicit (switch=1.0) or semi implicit (switch=0.5);
        iswitch = 0.5

        tion = 1.5789e5  # ionization temperature
        beta0 = 43.082  # ionizatio coefficient (?)
        dec2g = 8.468e14  # two photon decay rate (in 1/Mpc)
        # recombination coefficient (in sqrt(K)/Mpc).
        alpha0 = 2.3866e-6 * (1.0 - self.cp.YHe) * self.cp.Omegab * self.cp.H0**2
        # coefficient for correction of radiative decay (dimensionless)
        crec = 8.0138e-26 * (1.0 - self.cp.YHe) * self.cp.Omegab * self.cp.H0**2
        # recombination and ionization rates.
        phi2 = np.maximum(0.448 * np.log(tion / tempb), 0.0)
        alpha = alpha0 / np.sqrt(tempb) * phi2 / a**3
        beta = tempb * phi2 * np.exp(beta0 - tion / tempb)
        # ... Peebles' correction factor
        cpeebles = 1.0
        if tempb >= 200.0:
            cp1 = crec * dec2g * (1.0 - xe) / (a * adot)
            cp2 = crec * tempb * phi2 * np.exp(beta0 - 0.25 * tion / tempb) * (1.0 - xe) / (a * adot)
            cpeebles = (1.0 + cp1) / (1.0 + cp1 + cp2)
        # ... integrate dxe=bb*(1-xe)-aa*xe*xe by averaging rhs at current tau
        # ... (fraction 1-iswitch) and future tau (fraction iswitch).
        aa = a * dtau * alpha * cpeebles
        bb = a * dtau * beta * cpeebles
        b1 = 1.0 + iswitch * bb
        bbxe = bb + xe - (1.0 - iswitch) * (bb * xe + aa * xe * xe)
        rat = iswitch * aa * bbxe / (b1 * b1)
        if rat < 1e-6:
            xe = bbxe / b1 * (1.0 - rat)
        else:
            xe = b1 / (2.0 * iswitch * aa) * (np.sqrt(4.0 * rat + 1.0) - 1.0)

        return xe

    # @njit
    def ionHe(self, tempb: float, a: float, x0: float, x1: float, x2: float) -> tuple[float, float]:
        """Compute the helium ionization fractions using the Saha equation

        Args:
            tempb (float): baryon temperature
            a (float):  scale factor
            x0 (float): hydrogen ionization fraction n(H+)/n(H) (input)
            x1 (float): helium first ionization fraction n(He+)/n(He) (in&out)
            x2 (float): helium second ionization fraction n(He++)/n(He) (in&out)

        Returns:
            tuple[float,float]: x1,x2
        """
        tion1 = 2.855e5
        tion2 = 6.313e5

        # ... constant for electron partition function per baryon
        b0 = 2.150e24 / ((1.0 - self.cp.YHe) * self.cp.Omegab * self.cp.H0**2)
        # ... electron partition function per baryon
        b = b0 * a**3 * tempb * np.sqrt(tempb)
        # ... dimensionless right-hand sides in Saha equations
        r1 = 4.0 * b * np.exp(-tion1 / tempb)
        r2 = b * np.exp(-tion2 / tempb)

        # ... solve coupled equations iteratively
        c = 0.25 * self.cp.YHe / (1.0 - self.cp.YHe)
        err = 1.0
        niter = 0

        while err > 1e-12:
            xe = x0 + c * (x1 + 2.0 * x2)
            x2new = r1 * r2 / (r1 * r2 + xe * r1 + xe * xe)
            x1 = xe * r1 / (r1 * r2 + xe * r1 + xe * xe)
            err = np.fabs(x2new - x2)
            x2 = x2new
            niter += 1

        return (x1, x2)

    # @njit
    def compute(self, taumin: float, taumax: float, nthermo: int):
        thomc0 = 5.0577e-8 * self.cp.Tcmb**4

        tauminn = taumin
        dlntau = np.log(taumax / taumin) / (nthermo - 1)

        # set up fields
        tb = np.zeros((nthermo))
        xe = np.zeros((nthermo))
        xH = np.zeros((nthermo))
        xHeI = np.zeros((nthermo))
        xHeII = np.zeros((nthermo))

        cs2 = np.zeros((nthermo))
        ttau = np.zeros((nthermo))

        # ... initial codnitions : assume radiation-dominated universe
        tau0 = taumin
        adot0 = self.cp.adotrad
        a0 = self.cp.adotrad * taumin

        # ... assume entropy generated before taumin.
        tb[0] = self.cp.Tcmb / a0

        # assume fully ionized initial state
        xe0 = 1.0
        x1 = 0.0
        x2 = 1.0
        xH[0] = xe0
        xHeI[0] = x1
        xHeII[0] = x2
        xe[0] = xe0 + 0.25 * self.cp.YHe / (1.0 - self.cp.YHe) * (x1 + 2.0 * x2)
        barssc0 = 9.1820e-14
        barssc = barssc0 * (1.0 - 0.75 * self.cp.YHe + (1.0 - self.cp.YHe) * xe[0])
        cs2[0] = 4.0 / 3.0 * barssc * tb[0]
        ttau[0] = tau0

        tau_neglect_rad = -1.0
        a_neglect_rad = 10.0 * (3.0 * self.cp.grhor + self.cp.grhog) / self.cp.grhom

        for i in range(1, nthermo):
            tau = taumin * np.exp(i * dlntau)
            dtau = tau - tau0

            ttau[i] = tau

            # integrate Friedmann equation using inverse trapezoidal rule.
            a = a0 + adot0 * dtau

            grho = (
                self.cp.grhom * self.cp.Omegam / a
                + (3.0 * self.cp.grhor + self.cp.grhog) / a**2
                + self.cp.grhom * self.cp.OmegaL * a**2
            )
            adot = np.sqrt(grho / 3.0) * a

            a = a0 + 2 * dtau / (1.0 / adot0 + 1.0 / adot)

            # ... check if radiation can be neglected
            if (tau_neglect_rad < 0.0) and (a > a_neglect_rad):
                tau_neglect_rad = tau

            # ... baryon temperature evolution: adiabatic except for Thomson cooling
            # ... use quadratic solution.
            tg0 = self.cp.Tcmb / a0
            ahalf = 0.5 * (a0 + a)
            adothalf = 0.5 * (adot0 + adot)

            # ... fe = number of free electrons divided by total number of free baryon
            # ... partilces (e+p+H+He). Evaluate at timestep i-1 for convenience; if
            # ... more accuracy is required (unlikely) then this can be iterated with
            # ... the solution of the ionization equation
            fe = (1.0 - self.cp.YHe) * xe[i - 1] / (1.0 - 0.75 * self.cp.YHe + (1.0 - self.cp.YHe) * xe[i - 1])
            thomc = thomc0 * fe / adothalf / (ahalf * ahalf * ahalf)
            etc = np.exp(-thomc * (a - a0))
            a2t = a0 * a0 * (tb[i - 1] - tg0) * etc - self.cp.Tcmb / thomc * (1.0 - etc)
            tb[i] = self.cp.Tcmb / a + a2t / (a * a)

            # ... integrate ionization equation
            tbhalf = 0.5 * (tb[i - 1] + tb[i])
            xe0 = self.ionize(tbhalf, ahalf, adothalf, dtau, xe0)
            x1, x2 = self.ionHe(tb[i], a, xe0, x1, x2)

            xe[i] = xe0 + 0.25 * self.cp.YHe / (1.0 - self.cp.YHe) * (x1 + 2.0 * x2)

            xH[i] = xe0
            xHeI[i] = x1
            xHeII[i] = x2

            # ... baryon sound speed squared (over c^2)
            # double adotoa=adot/a;
            dtbdla = -2.0 * tb[i] - thomc * a2t / a
            barssc = barssc0 * (1.0 - 0.75 * self.cp.YHe + (1.0 - self.cp.YHe) * xe[i])
            cs2[i] = barssc * tb[i] * (1.0 - dtbdla / tb[i] / 3.0)

            # ... use old values for next loop
            a0 = a
            tau0 = tau
            adot0 = adot

        return ttau, tb, cs2, xe, xH, xHeI, xHeII

    def evaluate_at_tau(self, tau):
        tempb = np.interp(tau, self.tau, self.tb)
        cs2 = np.interp(tau, self.tau, self.cs2)
        xe = np.interp(tau, self.tau, self.xe)

        return tempb, cs2, xe

    def __init__(self, *, taumin: float, taumax: float, cp: cosmo, N: int):
        self.cp = cp

        self.tau, self.tb, self.cs2, self.xe, self.xH, self.xHeI, self.xHeII = self.compute(taumin, taumax, N)
