import numpy as np

from pylinger_cosmo import cosmo
from pylinger_thermo import thermal_history

from scipy.integrate import solve_ivp


class pt_synchronous:

    def model(self, tau: float, yin: np.ndarray) -> np.ndarray:
        y = np.copy(yin).reshape((self.nmodes, self.nvar))
        f = np.zeros_like(y)

        # print(y[0,0])

        # ... metric
        a = y[:, 0]
        ahdot = y[:, 1]
        eta = y[:, 2]

        # ... cdm
        deltac = y[:, 3]
        thetac = y[:, 4]

        # ... baryons
        deltab = y[:, 5]
        thetab = y[:, 6]

        # ... photons
        deltag = y[:, 7]
        thetag = y[:, 8]
        shearg = y[:, 9] / 2.0

        # ... polarization term
        polter = y[:, 9] + y[:, 8 + self.lmax] + y[:, 10 + self.lmax]

        # ... massless neutrinos
        deltar = y[:, 9 + 2 * self.lmax]
        thetar = y[:, 10 + 2 * self.lmax]
        shearr = y[:, 11 + 2 * self.lmax] / 2.0

        tempb, cs2, xe = self.th.evaluate_at_tau(tau)

        # ... Thomson opacity coefficient
        akthom = 2.3048e-9 * (1.0 - self.cp.YHe) * self.cp.Omegab * self.cp.H0**2

        # ... Thomson opacity
        opac = xe * akthom / a**2

        # ... Photon mass density over baryon mass density
        photbar = self.cp.grhog / (self.cp.grhom * self.cp.Omegab * a)
        pb43 = 4.0 / 3.0 * photbar

        # ... compute expansion rate
        grho = (
            self.cp.grhom * self.cp.Omegam / a
            + (self.cp.grhog + 3.0 * self.cp.grhor) / a**2
            + self.cp.grhom * self.cp.OmegaL * a**2
        )
        adotoa = np.sqrt(grho / 3.0)

        f[:, 0] = adotoa * a

        gpres = ((self.cp.grhog + 3.0 * self.cp.grhor) / 3.0) / a**2 - self.cp.grhom * self.cp.OmegaL * a**2

        # ... evaluate metric perturbations
        dgrho = (
            self.cp.grhom * (self.cp.Omegac * deltac + self.cp.Omegab * deltab) / a
            + (self.cp.grhog * deltag + self.cp.grhor * 3.0 * deltar) / a**2
        )
        dgpres = (self.cp.grhog * deltag + self.cp.grhor * 3.0 * deltar) / a**2 / 3.0

        dahdotdtau = -(dgrho + 3.0 * dgpres) * a
        f[:, 1] = dahdotdtau

        # ... force energy conservation
        hdot = (2.0 * self.ak**2 * eta + dgrho) / adotoa

        dgtheta = (
            self.cp.grhom * (self.cp.Omegac * thetac + self.cp.Omegab * thetab) / a
            + 4.0 / 3.0 * (self.cp.grhog * thetag + 3.0 * self.cp.grhor * thetar) / a**2
        )
        etadot = 0.5 * dgtheta / self.ak**2
        f[:, 2] = etadot

        dgshear = 4.0 / 3.0 * (self.cp.grhog * shearg + 3.0 * self.cp.grhor * shearr) / a**2

        # ... cdm equations of motion
        deltacdot = -thetac - 0.5 * hdot
        f[:, 3] = deltacdot
        thetacdot = -adotoa * thetac
        f[:, 4] = thetacdot

        # ... baryon equations of motion
        deltabdot = -thetab - 0.5 * hdot
        f[:, 5] = deltabdot
        # ... need photon perturbation for first-order correction to tightly-coupled baryon-photon approx.
        deltagdot = 4.0 / 3.0 * (-thetag - 0.5 * hdot)
        drag = opac * (thetag - thetab)
        thetabdot = 1.0

        if tempb < 2.0e4:
            # ... treat baryons and photons as uncoupled
            thetabdot = -adotoa * thetab + self.ak**2 * cs2 * deltab + pb43 * drag
        else:
            # ... treat baryons and photons as tighly coupled.
            # ... zeroth order approx to baryon velocity
            thetabdot = (
                -adotoa * thetab + self.ak**2 * cs2 * deltab + self.ak**2 * pb43 * (0.25 * deltag - shearg)
            ) / (1.0 + pb43)
            adotdota = 0.5 * (adotoa * adotoa - gpres)

            # ... first-order approximation to baryon-photon slip, thetabdot-thetagdot.
            slip = 2.0 * pb43 / (1.0 + pb43) * adotoa * (thetab - thetag) + 1.0 / opac * (
                -adotdota * thetab
                - adotoa * self.ak**2 * 0.5 * deltag
                + self.ak**2 * (cs2 * deltabdot - 0.25 * deltagdot)
            ) / (1.0 + pb43)
            # ... first oder approximation to baryon velocity
            thetabdot += pb43 / (1.0 + pb43) * slip

        f[:, 6] = thetabdot

        # ... photon total intensity and polarization equations of motion
        f[:, 7] = deltagdot
        thetagdot = (-thetabdot - adotoa * thetab + self.ak**2 * cs2 * deltab) / pb43 + self.ak**2 * (
            0.25 * deltag - shearg
        )
        f[:, 8] = thetagdot

        if tempb < 2.0e5:
            # ... treat baryons and photons as uncoupled
            f[:, 9] = (
                8.0 / 15.0 * thetag
                - 0.6 * self.ak * y[:, 10]
                - opac * y[:, 9]
                + 4.0 / 15.0 * hdot
                + 8.0 / 5.0 * etadot
                + 0.1 * opac * polter
            )

            # ... polarization equations for l=1,2,3...
            f[:, 8 + self.lmax] = -self.ak * y[:, 9 + self.lmax] - opac * y[:, 8 + self.lmax] + 0.5 * opac * polter
            f[:, 9 + self.lmax] = (
                self.ak / 3.0 * (y[:, 8 + self.lmax] - 2.0 * y[:, 10 + self.lmax]) - opac * y[:, 9 + self.lmax]
            )
            f[:, 10 + self.lmax] = (
                self.ak * (0.4 * y[:, 9 + self.lmax] - 0.6 * y[:, 11 + self.lmax])
                - opac * y[:, 10 + self.lmax]
                + 0.1 * opac * polter
            )
            for i in range(2, self.lmax - 1):
                f[:, 8 + i] = (
                    self.ak * self.denl[i] * ((i + 1) * y[:, 7 + i] - (i + 2) * y[:, 9 + i]) - opac * y[:, 8 + i]
                )
                f[:, 9 + self.lmax + i] = (
                    self.ak * self.denl[i] * ((i + 1) * y[:, 8 + self.lmax + i] - (i + 2) * y[:, 10 + self.lmax + i])
                    - opac * y[:, 9 + self.lmax + i]
                )

        else:
            f[:, 9] = 0.0
            f[:, 8 + self.lmax] = 0.0
            f[:, 9 + self.lmax] = 0.0
            f[:, 10 + self.lmax] = 0.0
            for l in range(2, self.lmax - 1):
                f[:, 8 + l] = 0.0
                f[:, 9 + self.lmax + l] = 0.0

        # ... truncate moment expansion
        f[:, 7 + self.lmax] = (
            self.ak * y[:, 6 + self.lmax] - (self.lmax + 1) / tau * y[:, 7 + self.lmax] - opac * y[:, 7 + self.lmax]
        )
        f[:, 8 + 2 * self.lmax] = (
            self.ak * y[:, 7 + 2 * self.lmax]
            - (self.lmax + 1) / tau * y[:, 8 + 2 * self.lmax]
            - opac * y[:, 8 + 2 * self.lmax]
        )

        # ... Massless neutrino equations of motion
        deltardot = 4.0 / 3.0 * (-thetar - 0.5 * hdot)
        f[:, 9 + 2 * self.lmax] = deltardot
        thetardot = self.ak**2 * (0.25 * deltar - shearr)
        f[:, 10 + 2 * self.lmax] = thetardot
        f[:, 11 + 2 * self.lmax] = (
            8.0 / 15.0 * thetar - 0.6 * self.ak * y[:, 12 + 2 * self.lmax] + 4.0 / 15.0 * hdot + 8.0 / 5.0 * etadot
        )
        for l in range(2, self.lmax - 1):
            f[:, 10 + 2 * self.lmax + l] = (
                self.ak
                * self.denl[l]
                * ((l + 1) * y[:, 9 + 2 * self.lmax + l] - (l + 2) * y[:, 11 + 2 * self.lmax + l])
            )
        # ... truncate moment expansion
        f[:, 9 + 3 * self.lmax] = self.ak * y[:, 8 + 3 * self.lmax] - (self.lmax + 1) / tau * y[:, 9 + 3 * self.lmax]

        return f.flatten()

    def adiabatic_ics(self, tau: float) -> np.ndarray:
        y = np.zeros((self.nmodes, self.nvar))
        # a = self.cp.get_a(tau)
        a = tau * self.cp.adotrad
        a2 = a**2

        grho = (
            self.cp.grhom * self.cp.Omegam / a
            + (self.cp.grhog + 3.0 * self.cp.grhor) / a2
            + self.cp.grhom * self.cp.OmegaL * a2
        )
        adotoa = np.sqrt(grho / 3.0)
        gpres = ((self.cp.grhog + 3.0 * self.cp.grhor) / 3.0) / a2 - self.cp.grhom * self.cp.OmegaL * a2
        s = grho + gpres
        fracnu = self.cp.grhor * 4.0 / 3.0 * (3.0) / a2 / s

        # ... use yrad=rho_matter/rho_rad to correct initial conditions for matter+radiation
        yrad = self.cp.grhom * self.cp.Omegam * a / (self.cp.grhog + 3.0 * self.cp.grhor)

        # .. isentropic ("adiabatic") initial conditions
        psi = -1.0
        C = (15.0 + 4.0 * fracnu) / 20.0 * psi
        akt2 = self.ak * tau
        akt2 *= akt2
        h = C * akt2 * (1.0 - 0.2 * yrad)
        eta = 2.0 * C - (5.0 + 4.0 * fracnu) / 6.0 / (15.0 + 4.0 * fracnu) * C * akt2 * (1.0 - yrad / 3.0)
        f1 = (23.0 + 4.0 * fracnu) / (15.0 + 4.0 * fracnu)

        deltac = -0.5 * h
        deltag = -2.0 / 3.0 * h * (1.0 - akt2 / 36.0)
        deltab = 0.75 * deltag
        deltar = -2.0 / 3.0 * h * (1.0 - akt2 / 36.0 * f1)
        thetac = 0.0
        thetag = -C / 18.0 * akt2 * akt2 / tau
        thetab = thetag
        thetar = f1 * thetag
        shearr = 4.0 / 15.0 * self.ak**2 / s * psi * (1.0 + 7.0 / 36.0 * yrad)
        ahdot = 2.0 * C * self.ak**2 * tau * a * (1.0 - 0.3 * yrad)

        # ... metric
        y[:, 0] = a
        y[:, 1] = ahdot
        y[:, 2] = eta

        # .. CDM
        y[:, 3] = deltac
        y[:, 4] = thetac

        # .. baryons
        y[:, 5] = deltab
        y[:, 6] = thetab

        # ... Photons (total intensity and polarization)
        y[:, 7] = deltag
        y[:, 8] = thetag
        y[:, 8 + self.lmax] = 0.0  # shearg
        y[:, 9 + self.lmax] = 0.0  # polarization term

        for l in range(1, self.lmax):
            y[:, 8 + l] = 0.0
            y[:, 9 + self.lmax + l] = 0.0

        # ... massless neutrinos
        y[:, 9 + 2 * self.lmax] = deltar
        y[:, 10 + 2 * self.lmax] = thetar
        y[:, 11 + 2 * self.lmax] = shearr * 2.0

        # for l in range(2, self.lmax):
        y[:, 10 + 2 * (self.lmax + 1) :] = 0.0

        return y.flatten()

    def __init__(self, *, cp: cosmo, th: thermal_history):
        self.cp = cp
        self.th = th

    def compute(
        self, *, kmodes: np.ndarray, aexp_out: np.ndarray, rtol: float = 1e-3, atol: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray]:
        tau_out = self.cp.get_tau(aexp_out)
        tau_start = np.minimum(1e-3 / np.max(kmodes), 0.1)
        tau_max = np.max(tau_out)

        self.nout = aexp_out.shape[0]

        self.ak = np.copy(kmodes)
        self.akmax = np.max(kmodes)

        self.lmax = np.minimum(10, int(1.5 * self.akmax * tau_max + 10.0))
        self.denl = 1 / (2 * np.arange(1, self.lmax + 1) + 1)

        self.nvar = 7 + 3 * (self.lmax + 1)
        self.nmodes = self.ak.shape[0]

        y0 = self.adiabatic_ics(tau_start)
        sol = solve_ivp(self.model, (tau_start, tau_max), y0, t_eval=tau_out, rtol=rtol, atol=atol)

        return np.array(sol["y"]).reshape((self.nmodes, self.nvar, self.nout)), np.array(sol["t"])
        # return y0.reshape((self.nmodes, self.nvar))
