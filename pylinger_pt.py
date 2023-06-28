import numpy as np

import jax.numpy as jnp

from pylinger_cosmo import cosmo, nu_perturb_jax, nu_perturb_numpy, get_tau
from pylinger_thermo import thermal_history

from scipy.integrate import solve_ivp


def model_synchronous(tau, yin, params) -> np.ndarray:
    y = np.copy(yin).reshape((params.nmodes, params.nvar))
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
    polter = y[:, 9] + y[:, 8 + params.lmax] + y[:, 10 + params.lmax]

    # ... massless neutrinos
    deltar = y[:, 9 + 2 * params.lmax]
    thetar = y[:, 10 + 2 * params.lmax]
    shearr = y[:, 11 + 2 * params.lmax] / 2.0

    tempb, cs2, xe = params.th.evaluate_at_tau(tau)

    # ... Thomson opacity coefficient
    akthom = 2.3048e-9 * (1.0 - params.cp.YHe) * params.cp.Omegab * params.cp.H0**2

    # ... Thomson opacity
    opac = xe * akthom / a**2

    # ... Photon mass density over baryon mass density
    photbar = params.cp.grhog / (params.cp.grhom * params.cp.Omegab * a)
    pb43 = 4.0 / 3.0 * photbar

    # ... compute expansion rate
    # grho = (
    #     params.cp.grhom * params.cp.Omegam / a
    #     + (params.cp.grhog + params.cp.grhor) / a**2
    #     + params.cp.grhom * params.cp.OmegaL * a**2
    # )
    # rhonu = params.cp.rhonu_sp(a)
    rhonu = np.interp(a, params.cp.a, params.cp.rhonu)
    # pnu   = params.cp.pnu_sp(a)
    pnu = np.interp(a, params.cp.a, params.cp.pnu)
    grho = (
        params.cp.grhom * params.cp.Omegam / a
        + (params.cp.grhog + params.cp.grhor * (params.cp.Neff + params.cp.Nmnu * rhonu)) / a**2
        + params.cp.grhom * params.cp.OmegaL * a**2
        + params.cp.grhom * params.cp.Omegak
    )
    adotoa = np.sqrt(grho / 3.0)

    f[:, 0] = adotoa * a

    # gpres = (
    #     ((params.cp.grhog + params.cp.grhor) / 3.0) / a**2
    #     - params.cp.grhom * params.cp.OmegaL * a**2
    # )
    gpres = (
        (params.cp.grhog + params.cp.grhor * params.cp.Neff) / 3.0 + params.cp.grhor * params.cp.Nmnu * pnu
    ) / a**2 - params.cp.grhom * params.cp.OmegaL * a**2
    # ... evaluate metric perturbations
    drhonu = np.zeros_like(params.ak)
    dpnu = np.zeros_like(params.ak)
    fnu = np.zeros_like(params.ak)
    shearnu = np.zeros_like(params.ak)

    for i in range(len(params.ak)):
        drhonu[i], dpnu[i], fnu[i], shearnu[i] = nu_perturb_numpy(
            a[0],
            params.cp.amnu,
            y[i, params.iq0 : params.iq1],
            y[i, params.iq1 : params.iq2],
            y[i, params.iq2 : params.iq3],
        )

    dgrho = (
        params.cp.grhom * (params.cp.Omegac * deltac + params.cp.Omegab * deltab) / a
        + (params.cp.grhog * deltag + params.cp.grhor * (params.cp.Neff * deltar + params.cp.Nmnu * drhonu)) / a**2
    )
    dgpres = (
        params.cp.grhog * deltag + params.cp.grhor * params.cp.Neff * deltar
    ) / a**2 / 3.0 + params.cp.grhor * params.cp.Nmnu * dpnu / a**2

    dahdotdtau = -(dgrho + 3.0 * dgpres) * a
    f[:, 1] = dahdotdtau

    # ... force energy conservation
    hdot = (2.0 * params.ak**2 * eta + dgrho) / adotoa

    dgtheta = (
        params.cp.grhom * (params.cp.Omegac * thetac + params.cp.Omegab * thetab) / a
        + 4.0 / 3.0 * (params.cp.grhog * thetag + params.cp.Neff * params.cp.grhor * thetar) / a**2
        + params.cp.Nmnu * params.cp.grhor * params.ak * fnu / a**2
    )
    etadot = 0.5 * dgtheta / params.ak**2
    f[:, 2] = etadot

    dgshear = (
        4.0 / 3.0 * (params.cp.grhog * shearg + params.cp.Neff * params.cp.grhor * shearr) / a**2
        + params.cp.Nmnu * params.cp.grhor * shearnu / a**2
    )

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

    if tempb < 1.0e4:
        # ... treat baryons and photons as uncoupled
        thetabdot = -adotoa * thetab + params.ak**2 * cs2 * deltab + pb43 * drag
    else:
        # ... treat baryons and photons as tighly coupled.
        # ... zeroth order approx to baryon velocity
        thetabdot = (
            -adotoa * thetab + params.ak**2 * cs2 * deltab + params.ak**2 * pb43 * (0.25 * deltag - shearg)
        ) / (1.0 + pb43)
        adotdota = 0.5 * (adotoa * adotoa - gpres)

        # ... first-order approximation to baryon-photon slip, thetabdot-thetagdot.
        slip = 2.0 * pb43 / (1.0 + pb43) * adotoa * (thetab - thetag) + 1.0 / opac * (
            -adotdota * thetab
            - adotoa * params.ak**2 * 0.5 * deltag
            + params.ak**2 * (cs2 * deltabdot - 0.25 * deltagdot)
        ) / (1.0 + pb43)
        # ... first oder approximation to baryon velocity
        thetabdot += pb43 / (1.0 + pb43) * slip

    f[:, 6] = thetabdot

    # ... photon total intensity and polarization equations of motion
    f[:, 7] = deltagdot
    thetagdot = (-thetabdot - adotoa * thetab + params.ak**2 * cs2 * deltab) / pb43 + params.ak**2 * (
        0.25 * deltag - shearg
    )
    f[:, 8] = thetagdot

    if tempb < 1.0e4:
        # ... treat baryons and photons as uncoupled
        f[:, 9] = (
            8.0 / 15.0 * thetag
            - 0.6 * params.ak * y[:, 10]
            - opac * y[:, 9]
            + 4.0 / 15.0 * hdot
            + 8.0 / 5.0 * etadot
            + 0.1 * opac * polter
        )

        # ... polarization equations for l=1,2,3...
        f[:, 8 + params.lmax] = -params.ak * y[:, 9 + params.lmax] - opac * y[:, 8 + params.lmax] + 0.5 * opac * polter
        f[:, 9 + params.lmax] = (
            params.ak / 3.0 * (y[:, 8 + params.lmax] - 2.0 * y[:, 10 + params.lmax]) - opac * y[:, 9 + params.lmax]
        )
        f[:, 10 + params.lmax] = (
            params.ak * (0.4 * y[:, 9 + params.lmax] - 0.6 * y[:, 11 + params.lmax])
            - opac * y[:, 10 + params.lmax]
            + 0.1 * opac * polter
        )
        for i in range(2, params.lmax - 1):
            f[:, 8 + i] = (
                params.ak * (1 / (2 * i + 2)) * ((i + 1) * y[:, 7 + i] - (i + 2) * y[:, 9 + i]) - opac * y[:, 8 + i]
            )
            f[:, 9 + params.lmax + i] = (
                params.ak
                * (1 / (2 * i + 2))
                * ((i + 1) * y[:, 8 + params.lmax + i] - (i + 2) * y[:, 10 + params.lmax + i])
                - opac * y[:, 9 + params.lmax + i]
            )

    else:
        f[:, 9] = 0.0
        f[:, 8 + params.lmax] = 0.0
        f[:, 9 + params.lmax] = 0.0
        f[:, 10 + params.lmax] = 0.0
        for l in range(2, params.lmax - 1):
            f[:, 8 + l] = 0.0
            f[:, 9 + params.lmax + l] = 0.0

    # ... truncate moment expansion
    f[:, 7 + params.lmax] = (
        params.ak * y[:, 6 + params.lmax]
        - (params.lmax + 1) / tau * y[:, 7 + params.lmax]
        - opac * y[:, 7 + params.lmax]
    )
    f[:, 8 + 2 * params.lmax] = (
        params.ak * y[:, 7 + 2 * params.lmax]
        - (params.lmax + 1) / tau * y[:, 8 + 2 * params.lmax]
        - opac * y[:, 8 + 2 * params.lmax]
    )

    # ... Massless neutrino equations of motion
    deltardot = 4.0 / 3.0 * (-thetar - 0.5 * hdot)
    f[:, 9 + 2 * params.lmax] = deltardot
    thetardot = params.ak**2 * (0.25 * deltar - shearr)
    f[:, 10 + 2 * params.lmax] = thetardot
    f[:, 11 + 2 * params.lmax] = (
        8.0 / 15.0 * thetar - 0.6 * params.ak * y[:, 12 + 2 * params.lmax] + 4.0 / 15.0 * hdot + 8.0 / 5.0 * etadot
    )
    for l in range(2, params.lmax - 1):
        f[:, 10 + 2 * params.lmax + l] = (
            params.ak
            / (2 * l + 2)
            * ((l + 1) * y[:, 9 + 2 * params.lmax + l] - (l + 2) * y[:, 11 + 2 * params.lmax + l])
        )
    # ... truncate moment expansion
    f[:, 9 + 3 * params.lmax] = (
        params.ak * y[:, 8 + 3 * params.lmax] - (params.lmax + 1) / tau * y[:, 9 + 3 * params.lmax]
    )

    # ... Massive neutrino equations of motion
    if params.cp.Nmnu > 0:
        # q  = (jnp.arange(0,params.nqmax) - 0.5)  # so dq == 1
        # q.at[0].set(0.0)
        q = np.arange(1, params.nqmax + 1) - 0.5  # so dq == 1
        aq = a[0] * params.cp.amnu / q
        v = 1 / np.sqrt(1 + aq**2)
        dlfdlq = -q / (1.0 + np.exp(-q))  # derivative of the Fermi-Dirac distribution

        f[:, params.iq0 : params.iq1] = (
            -params.ak[:, None] * v[None, :] * y[:, params.iq1 : params.iq2] + hdot[:, None] * dlfdlq / 6.0
        )
        f[:, params.iq1 : params.iq2] = (
            params.ak[:, None]
            * v[None, :]
            * (y[:, params.iq0 : params.iq1] - 2.0 * y[:, params.iq2 : params.iq3])
            / 3.0
        )
        f[:, params.iq2 : params.iq3] = (
            params.ak[:, None]
            * v[None, :]
            * (2 * y[:, params.iq1 : params.iq2] - 3 * y[:, params.iq3 : params.iq4]) / 5.0
            - (hdot[:, None] / 15 + 2 / 5 * etadot[:, None]) * dlfdlq
        )

        for l in range(3, params.lmaxnu - 1):
            f[:, params.iq0 + l * params.nqmax : params.iq0 + (l + 1) * params.nqmax] = (
                params.ak[:, None]
                * v[None, :]
                / (2 * l + 1)
                * (
                    l * y[:, params.iq0 + (l - 1) * params.nqmax : params.iq0 + (l) * params.nqmax]
                    - (l + 1) * y[:, params.iq0 + (l + 1) * params.nqmax : params.iq0 + (l + 2) * params.nqmax]
                )
            )

        # Truncate moment expansion.
        f[:, -params.nqmax :] = (
            params.ak[:, None] * v[None, :] * y[:, -2 * params.nqmax : -params.nqmax]
            - (params.lmaxnu + 1) / tau * y[:, -params.nqmax :]
        )

    return f.flatten()


def adiabatic_ics(tau: float, params) -> np.ndarray:
    """Initial conditions for adiabatic perturbations"""
    y = np.zeros((params.nmodes, params.nvar))
    # a = params.cp.get_a(tau)
    a = tau * params.cp.adotrad
    a2 = a**2

    rhonu = np.interp(a, params.cp.a, params.cp.rhonu)
    pnu = np.interp(a, params.cp.a, params.cp.pnu)
    grho = (
        params.cp.grhom * params.cp.Omegam / a
        + (params.cp.grhog + params.cp.grhor * (params.cp.Neff + params.cp.Nmnu * rhonu)) / a**2
        + params.cp.grhom * params.cp.OmegaL * a**2
        + params.cp.grhom * params.cp.Omegak
    )
    gpres = (
        (params.cp.grhog + params.cp.grhor * params.cp.Neff) / 3.0 + params.cp.grhor * params.cp.Nmnu * pnu
    ) / a**2 - params.cp.grhom * params.cp.OmegaL * a**2
    adotoa = np.sqrt(grho / 3.0)

    s = grho + gpres

    fracnu = params.cp.grhor * (params.cp.Neff + params.cp.Nmnu) * 4.0 / 3.0 / a2 / s

    # ... use yrad=rho_matter/rho_rad to correct initial conditions for matter+radiation
    yrad = (
        params.cp.grhom
        * params.cp.Omegam
        * a
        / (params.cp.grhog + params.cp.grhor * (params.cp.Neff + params.cp.Nmnu * rhonu))
    )

    # .. isentropic ("adiabatic") initial conditions
    psi = -1.0
    C = (15.0 + 4.0 * fracnu) / 20.0 * psi
    akt2 = params.ak * tau
    akt2 *= akt2
    h = C * akt2 * (1.0 - 0.2 * yrad)
    eta = 2.0 * C - (5.0 + 4.0 * fracnu) / 6.0 / (15.0 + 4.0 * fracnu) * C * akt2 * (1.0 - yrad / 3.0)
    f1 = (23.0 + 4.0 * fracnu) / (15.0 + 4.0 * fracnu)

    deltac = -0.5 * h
    deltag = -2.0 / 3.0 * h * (1.0 - akt2 / 36.0)
    deltab = 0.75 * deltag
    deltar = -2.0 / 3.0 * h * (1.0 - akt2 / 36.0 * f1)
    deltan = deltar
    thetac = 0.0
    thetag = -C / 18.0 * akt2 * akt2 / tau
    thetab = thetag
    thetar = f1 * thetag
    thetan = thetar
    shearr = 4.0 / 15.0 * params.ak**2 / s * psi * (1.0 + 7.0 / 36.0 * yrad)
    shearn = shearr
    ahdot = 2.0 * C * params.ak**2 * tau * a * (1.0 - 0.3 * yrad)

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
    y[:, 8 + params.lmax] = 0.0  # shearg
    y[:, 9 + params.lmax] = 0.0  # polarization term

    for l in range(1, params.lmax):
        y[:, 8 + l] = 0.0
        y[:, 9 + params.lmax + l] = 0.0

    # ... massless neutrinos
    y[:, 9 + 2 * params.lmax] = deltar
    y[:, 10 + 2 * params.lmax] = thetar
    y[:, 11 + 2 * params.lmax] = shearr * 2.0

    # for l in range(2, params.lmax):
    y[:, 10 + 2 * (params.lmax + 1) :] = 0.0

    # ... massive neutrinos
    if params.cp.Nmnu > 0:
        q = jnp.arange(1, params.nqmax + 1) - 0.5  # so dq == 1
        aq = a * params.cp.amnu / q
        v = 1 / jnp.sqrt(1 + aq**2)
        akv = jnp.outer(params.ak, v)
        dlfdlq = -q / (1.0 + jnp.exp(-q))
        y[:, params.iq0 : params.iq1] = -0.25 * dlfdlq[None, :] * deltan[:, None]
        y[:, params.iq1 : params.iq2] = -dlfdlq[None, :] * thetan[:, None] / v[None, :] / params.ak[:, None] / 3.0
        y[:, params.iq2 : params.iq3] = -0.5 * dlfdlq[None, :] * shearn[:, None]
        y[:, params.iq3 :] = 0.0
    # for i in range(len(params.ak)):
    #     drhonu, dpnu, fnu, shearnu = nu_perturb( a, params.cp.amnu, y[i,params.iq0], y[i,params.iq1], y[i,params.iq2])

    return y.flatten()


class pt_synchronous:
    def __init__(self, *, cp: cosmo):
        self.cp = cp
        self.th = cp.th

    def compute(
        self, *, kmodes: np.ndarray, aexp_out: np.ndarray, rtol: float = 1e-3, atol: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray]:
        tau_out = np.zeros_like(aexp_out)
        for i in range(len(aexp_out)):
            tau_out[i] = get_tau(aexp_out[i], self.cp)
        tau_start = np.minimum(1e-3 / np.max(kmodes), 0.1)
        tau_max = np.max(tau_out)

        self.nout = aexp_out.shape[0]

        self.ak = np.copy(kmodes)
        self.akmax = np.max(kmodes)

        # number of multipoles for photon and neutrino perturbations
        self.lmax = np.minimum(10, int(1.5 * self.akmax * tau_max + 10.0))  # check in CLASS!
        self.lmaxnu = 32  # 50 in linger_syn, check in CLASS!
        self.nqmax = 15
        self.iq0 = 11 + 3 * self.lmax
        self.iq1 = self.iq0 + self.nqmax
        self.iq2 = self.iq1 + self.nqmax
        self.iq3 = self.iq2 + self.nqmax
        self.iq4 = self.iq3 + self.nqmax

        # set up storage array
        self.nvar = 7 + 3 * (self.lmax + 1) + self.nqmax * (self.lmaxnu + 1)
        self.nmodes = self.ak.shape[0]

        # set initial conditions
        y0 = adiabatic_ics(tau_start, self)

        # solve
        sol = solve_ivp(
            lambda tau, y: model_synchronous(tau, y, self),
            (tau_start, tau_max),
            y0,
            t_eval=tau_out,
            rtol=rtol,
            atol=atol,
        )

        return np.array(sol["y"]).reshape((self.nmodes, self.nvar, self.nout)), np.array(sol["t"])
        # return y0.reshape((self.nmodes, self.nvar))
