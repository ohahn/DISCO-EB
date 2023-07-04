import jax
import jax.numpy as jnp
import jax_cosmo.scipy.interpolate as jaxinterp
from pylinger_cosmo import cosmo, nu_perturb_jax
from functools import partial
import diffrax

@partial(jax.jit, static_argnames=('lmax', 'lmaxnu', 'nqmax'))
def model_synchronous(*, tau, yin, param, kmode, lmax, lmaxnu, nqmax, ):
    """Solve the synchronous gauge perturbation equations for a single mode.

    Parameters
    ----------
    tau : float
        conformal time
    yin : array_like
        input vector of perturbations
    param : array_like
        dictionary of parameters and interpolated background functions
    kmode : float
        wavenumber of mode
    lmax : int
        maximum Boltzmann hierarchy multipole
    lmaxnu : int
        maximum neutrino hierarchy multipole
    nqmax : int
        maximum number of quadrature points for Fermi-Dirac integrals

    Returns
    -------
    f : array_like
        RHS of perturbation equations
    """

    iq0 = 11 + 3 * lmax
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax

    y = jnp.copy(yin)
    f = jnp.zeros_like(y)

    #TODO: add curvature
    # ... curvature
    K = 0
    
    # def cotKgen_zero_curv():
    #     return 1.0/(kmode*tau)
    # def cotKgen_pos_curv():
    #     return jnp.sqrt(K)/kmode/jnp.tan(jnp.sqrt(K)*tau)
    # def cotKgen_neg_curv():
    #     return jnp.sqrt(-K)/kmode/jnp.tanh(jnp.sqrt(-K)*tau)
    
    # cotKgen = jax.lax.switch(int(1+jax.lax.sign(K)), [cotKgen_neg_curv, cotKgen_zero_curv, cotKgen_pos_curv])
    s2_squared = 1.-3.*K/kmode**2
    s_l2 = 1.0
    s_l3 = 1.0

    # ... metric
    a = y[0]
    ahprime = y[1]
    eta = y[2]

    # ... cdm
    deltac = y[3]
    thetac = y[4]

    # ... baryons
    deltab = y[5]
    thetab = y[6]

    # ... photons
    deltag = y[7]
    thetag = y[8]
    shearg = y[9] / 2.0

    # ... polarization term
    polter = y[9] + y[8 + lmax] + y[10 + lmax]

    # ... massless neutrinos
    deltar = y[9 + 2 * lmax]
    thetar = y[10 + 2 * lmax]
    shearr = y[11 + 2 * lmax] / 2.0

    # ... evaluate thermodynamics
    tempb = param['tempb_of_tau_spline']( tau )
    cs2 = param['cs2_of_tau_spline']( tau )
    xe = param['xe_of_tau_spline']( tau )
    xeprime = param['xe_of_tau_spline'].derivative( tau )

    # ... Photon mass density over baryon mass density
    photbar = param['grhog'] / (param['grhom'] * param['Omegab'] * a)
    pb43 = 4.0 / 3.0 * photbar

    # ... compute expansion rate
    rhonu = param['rhonu_of_a_spline']( a )
    pnu = param['pnu_of_a_spline']( a ) 
    grho = (
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaL'] * a**2
        + param['grhom'] * param['Omegak']
    )

    gpres = (
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 - param['grhom'] * param['OmegaL'] * a**2

    aprimeoa = jnp.sqrt(grho / 3.0)
    aprimeprimeoa = 0.5 * (aprimeoa * aprimeoa - gpres)

    # ... Thomson opacity coefficient
    akthom = 2.3048e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2

    # ... Thomson opacity
    # photon_scattering_rate = pvecthermo[pth->index_th_dkappa];
    # tau = 1/ photon_scattering_rate;
    opac    = xe * akthom / a**2
    tauc    = 1. / opac
    taucprime = tau * (2*aprimeoa - xeprime/xe)
    F       = tauc / (1+pb43) #CLASS perturbations.c:10072
    Fprime    = taucprime/(1+pb43) + tauc*pb43*aprimeoa/(1+pb43)**2 #CLASS perturbations.c:10074


    # ... background scale factor evolution
    f = f.at[0].set( aprimeoa * a )
    
    # ... evaluate metric perturbations
    drhonu, dpnu, fnu, shearnu = nu_perturb_jax( a, param['amnu'], y[iq0:iq1], y[iq1:iq2], y[iq2:iq3] )

    dgrho = (
        param['grhom'] * (param['Omegac'] * deltac + param['Omegab'] * deltab) / a
        + (param['grhog'] * deltag + param['grhor'] * (param['Neff'] * deltar + param['Nmnu'] * drhonu)) / a**2
    )
    dgpres = (
        param['grhog'] * deltag + param['grhor'] * param['Neff'] * deltar
    ) / a**2 / 3.0 + param['grhor'] * param['Nmnu'] * dpnu / a**2

    dahprimedtau = -(dgrho + 3.0 * dgpres) * a
    
    f = f.at[1].set( dahprimedtau )

    # ... force energy conservation
    hprime = (2.0 * kmode**2 * eta + dgrho) / aprimeoa

    dgtheta = (
        param['grhom'] * (param['Omegac'] * thetac + param['Omegab'] * thetab) / a
        + 4.0 / 3.0 * (param['grhog'] * thetag + param['Neff'] * param['grhor'] * thetar) / a**2
        + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2
    )
    etaprime = 0.5 * dgtheta / kmode**2
    alpha  = (hprime + 6.*etaprime)/2./kmode**2
    f = f.at[2].set( etaprime )

    dgshear = (
        4.0 / 3.0 * (param['grhog'] * shearg + param['Neff'] * param['grhor'] * shearr) / a**2
        + param['Nmnu'] * param['grhor'] * shearnu / a**2
    )
    
    alphaprime = -3*dgshear/(2*kmode**2) + eta - 2*aprimeoa*alpha
    alphaprime -=  9/2 * a**2/kmode**2 * 4/3*16/45/opac * (thetag+kmode**2*alpha) * param['grhog']

    # ... cdm equations of motion
    deltacprime = -thetac - 0.5 * hprime
    f = f.at[3].set( deltacprime )
    thetacprime = -aprimeoa * thetac
    f = f.at[4].set( thetacprime )


    def calc_baryon_photon_uncoupled( f ):
        # === treat baryons and photons as uncoupled =================================================
        # --- baryon equations of motion -------------------------------------------------------------
        # ... baryon density, BLT11 eq. (2.1a)
        deltabprime = -thetab - 0.5 * hprime
        f = f.at[5].set( deltabprime )
        # ... baryon velocity, BLT11 eq. (2.1b)
        thetabprime = -aprimeoa * thetab + kmode**2 * cs2 * deltab \
                    + pb43 * opac * (thetag - thetab)
        f = f.at[6].set( thetabprime )

        # --- photon equations of motion -------------------------------------------------------------
        # ... photon density, BLT11 eq. (2.4a)
        deltagprime = 4.0 / 3.0 * (-thetag - 0.5 * hprime)
        f = f.at[7].set( deltagprime )
        # ... photon velocity, BLT11 eq. (2.4b)
        thetagprime = kmode**2 * (0.25 * deltag - s2_squared * shearg) \
                    + opac * (thetab - thetag)
        f = f.at[8].set( thetagprime )
        # ... photon shear, BLT11 eq. (2.4c)
        sheargprime = 8./15. * (thetag+kmode**2*alpha) -3/5*kmode*s_l3/s_l2*y[10] \
                    - opac*(y[9]-0.1*s_l2*polter)
        f = f.at[9].set( sheargprime )

        # photon temperature l>=3, BLT11 eq. (2.4d)
        ell  = jnp.arange(2, lmax - 1)
        f = f.at[8+ell].set( kmode  / (2 * ell + 2) * ((ell + 1) * y[7 + ell] - (ell + 2) * y[9 + ell]) - opac * y[8 + ell] )
        # photon temperature hierarchy truncation, BLT11 eq. (2.5)
        f = f.at[7 + lmax].set( kmode * y[6 + lmax] - (lmax + 1) / tau * y[7 + lmax] - opac * y[7 + lmax] )
    
        # polarization equations, BLT11 eq. (2.4e)
        # photon polarization l=0
        f = f.at[8 + lmax].set( -kmode * y[9 + lmax] - opac * y[8 + lmax] + 0.5 * opac * polter )
        # photon polarization l=1
        f = f.at[9 + lmax].set( kmode / 3.0 * (y[8 + lmax] - 2.0 * y[10 + lmax]) - opac * y[9 + lmax] )
        # photon polarization l=2
        f = f.at[10 + lmax].set( kmode * (0.4 * y[9 + lmax] - 0.6 * y[11 + lmax]) - opac * (y[10 + lmax] - 0.1 * s_l2 * polter))
        # photon polarization lmax>l>=3
        ell  = jnp.arange(2, lmax - 1)
        f = f.at[9+lmax+ell].set( kmode  / (2 * ell + 2) * ((ell + 1) * y[8 + lmax + ell] - (ell + 2) * y[10 + lmax + ell]) - opac * y[9 + lmax + ell] )
        # photon polarization hierarchy truncation
        f = f.at[8 + 2 * lmax].set( kmode * y[7 + 2 * lmax] - (lmax + 1) / tau * y[8 + 2 * lmax] - opac * y[8 + 2 * lmax] )
        
        return f
    
    def calc_baryon_photon_tca( f ):
        # === treat baryons and photons as tighly coupled (TCA) =====================================
        # first order slip
        thetaprime  = (-aprimeoa*thetab+kmode**2*(cs2*deltab+pb43/4*deltag))/(1+pb43)

        # second order slip, BLT11 eq. (2.20)
        tca_slip = (taucprime/tauc - 2*aprimeoa/(1+pb43))*(thetab-thetag) \
            + F * (-aprimeprimeoa * thetab +kmode**2*(-aprimeoa*deltag/2 + cs2*(-thetab-0.5*hprime)-4./3.*(-thetag-0.5*hprime)/4.))

        # first order photon shear, BLT11 eq. (2.24)
        shearg    = 16/45*tauc*(thetag + kmode**2 * alpha)
        # first order photon shear time deriv., BLT11 eq. (2.25)
        sheargprime = 16/45*(tauc*(thetaprime + kmode**2 * alphaprime) + taucprime * (thetag + kmode**2 * alpha))

        # dominant second order slip, BLT11 eq. (2.29)
        tca_slip = (1 - 2*aprimeoa * F) * tca_slip + F * kmode**2 * (2*aprimeoa * s2_squared * shearg + s2_squared * sheargprime - (1/3-cs2)*(F*thetaprime + 2*Fprime*thetab))

        # second order photon shear, BLT11 eq. (2.26)
        tca_shearg = (1.-11./6*taucprime) * shearg - 11./6 * tauc * 16/45 * tauc * (thetaprime + kmode**2 * alphaprime)
        
        # --- baryon equations of motion -------------------------------------------------------------
        # ... baryon density
        deltabprime = -thetab - 0.5 * hprime
        f = f.at[5].set( deltabprime )
        # ... baryon velocity
        thetabprime = (-aprimeoa * thetab + kmode**2 * (cs2 * deltab + pb43 * (0.25 * deltag - s2_squared * tca_shearg)) + pb43 * tca_slip) / (1.0 + pb43)
        f = f.at[6].set( thetabprime )

        # --- photon equations of motion -------------------------------------------------------------
        # ... photon density
        deltagprime = 4.0 / 3.0 * (-thetag - 0.5 * hprime)
        f = f.at[7].set( deltagprime )
        # ... photon velocity
        thetagprime = -(thetab + aprimeoa * thetab - kmode**2 * cs2 * deltab) / pb43  + kmode**2 * (0.25 * deltag -s2_squared * tca_shearg)
        f = f.at[8].set( thetagprime )

        return f
    
    # --- check if we are in the tight coupling regime -----------------------------------------------
    tight_coupling_trigger_tau_c_over_tau_h=0.005 # value taken from CLASS
    tight_coupling_trigger_tau_c_over_tau_k=0.008 # value taken from CLASS

    tauh = 1./(aprimeoa)
    tauk = 1./kmode
    tca_condition = jnp.logical_and(jnp.logical_and(tauc/tauh < tight_coupling_trigger_tau_c_over_tau_h, tauc/tauk < tight_coupling_trigger_tau_c_over_tau_k),opac<1e-4)

    f = jax.lax.cond(tca_condition, calc_baryon_photon_tca, calc_baryon_photon_uncoupled, f )

    # --- Massless neutrino equations of motion -------------------------------------------------------
    deltarprime = 4.0 / 3.0 * (-thetar - 0.5 * hprime)
    f = f.at[9 + 2 * lmax].set( deltarprime )
    thetarprime = kmode**2 * (0.25 * deltar - shearr)
    f = f.at[10 + 2 * lmax].set( thetarprime )
    f = f.at[11 + 2 * lmax].set( 8.0 / 15.0 * thetar - 0.6 * kmode * y[12 + 2 * lmax] + 4.0 / 15.0 * hprime + 8.0 / 5.0 * etaprime )
    ell = jnp.arange(2, lmax - 1)
    f = f.at[10 + 2 * lmax + ell].set( kmode / (2 * ell + 2) * ((ell + 1) * y[9 + 2 * lmax + ell] - (ell + 2) * y[11 + 2 * lmax + ell]) )
    
    # ... truncate moment expansion
    f = f.at[9 + 3 * lmax].set( kmode * y[8 + 3 * lmax] - (lmax + 1) / tau * y[9 + 3 * lmax] )

    # --- Massive neutrino equations of motion --------------------------------------------------------
    q = jnp.arange(1, nqmax + 1) - 0.5  # so dq == 1
    aq = a * param['amnu'] / q
    v = 1 / jnp.sqrt(1 + aq**2)
    dlfdlq = -q / (1.0 + jnp.exp(-q))  # derivative of the Fermi-Dirac distribution

    f = f.at[iq0 : iq1].set(
        -kmode * v * y[iq1 : iq2] + hprime* dlfdlq / 6.0
    )
    f = f.at[iq1 : iq2].set(
        kmode * v * (y[iq0 : iq1] - 2.0 * y[iq2 : iq3]) / 3.0
    )
    f = f.at[iq2 : iq3].set(
        kmode * v * (2 * y[iq1 : iq2] - 3 * y[iq3 : iq4]) / 5.0 - (hprime / 15 + 2 / 5 * etaprime) * dlfdlq
    )

    for l in range(3, lmaxnu - 1):
        f = f.at[iq0 + l * nqmax : iq0 + (l + 1) * nqmax].set(
            kmode * v / (2 * l + 1) * (
                l * y[iq0 + (l - 1) * nqmax : iq0 + (l) * nqmax]
                - (l + 1) * y[iq0 + (l + 1) * nqmax : iq0 + (l + 2) * nqmax]
            )
        )

    # Truncate moment expansion.
    f = f.at[-nqmax :].set(
        kmode * v * y[-2 * nqmax : -nqmax] - (lmaxnu + 1) / tau * y[-nqmax :]
    )

    return f.flatten()

# @partial(jax.jit, static_argnames=('num_k', 'nvar', 'lmax', 'nqmax'))
def adiabatic_ics( *, tau: float, param, kmodes, num_k, nvar, lmax, nqmax):
    """Initial conditions for adiabatic perturbations"""
    iq0 = 11 + 3 * lmax
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax

    y = jnp.zeros((num_k, nvar))
    a = tau * param['adotrad']
    a2 = a**2

    rhonu = param['rhonu_of_a_spline'](a)
    pnu = param['pnu_of_a_spline'](a)
    grho = (
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaL'] * a**2
        + param['grhom'] * param['Omegak']
    )
    gpres = (
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 - param['grhom'] * param['OmegaL'] * a**2
    aprimeoa = jnp.sqrt(grho / 3.0)

    s = grho + gpres

    fracnu = param['grhor'] * (param['Neff'] + param['Nmnu']) * 4.0 / 3.0 / a2 / s

    # ... use yrad=rho_matter/rho_rad to correct initial conditions for matter+radiation
    yrad = (
        param['grhom'] * param['Omegam'] * a
        / (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu))
    )

    # .. isentropic ("adiabatic") initial conditions
    psi = -1.0
    C = (15.0 + 4.0 * fracnu) / 20.0 * psi
    akt2 = kmodes * tau
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
    shearr = 4.0 / 15.0 * kmodes**2 / s * psi * (1.0 + 7.0 / 36.0 * yrad)
    shearn = shearr
    ahprime = 2.0 * C * kmodes**2 * tau * a * (1.0 - 0.3 * yrad)

    # ... metric
    y = y.at[:, 0].set( a )
    y = y.at[:, 1].set( ahprime )
    y = y.at[:, 2].set( eta )

    # .. CDM
    y = y.at[:, 3].set( deltac )
    y = y.at[:, 4].set( thetac )

    # .. baryons
    y = y.at[:, 5].set( deltab )
    y = y.at[:, 6].set( thetab )

    # ... Photons (total intensity and polarization)
    y = y.at[:, 7].set( deltag )
    y = y.at[:, 8].set( thetag )
    y = y.at[:, 8 + lmax].set( 0.0 ) # shearg
    y = y.at[:, 9 + lmax].set( 0.0 ) # polarization term

    # for l in range(1, lmax):
    #     y = y.at[:, 8 + l].set( 0.0 )
    #     y = y.at[:, 9 + lmax + l].set( 0.0 )

    # ... massless neutrinos
    y = y.at[:, 9 + 2 * lmax].set( deltar )
    y = y.at[:, 10 + 2 * lmax].set( thetar )
    y = y.at[:, 11 + 2 * lmax].set( shearr * 2.0 )

    # for l in range(2, lmax):
    y = y.at[:, 10 + 2 * (lmax + 1) :].set( 0.0 )

    # ... massive neutrinos
    # if params.cp.Nmnu > 0:
    q = jnp.arange(1, nqmax + 1) - 0.5  # so dq == 1
    aq = a * param['amnu'] / q
    v = 1 / jnp.sqrt(1 + aq**2)
    akv = jnp.outer(kmodes, v)
    dlfdlq = -q / (1.0 + jnp.exp(-q))
    y = y.at[:, iq0:iq1].set( -0.25 * dlfdlq[None, :] * deltan[:, None] )
    y = y.at[:, iq1:iq2].set( -dlfdlq[None, :] * thetan[:, None] / v[None, :] / kmodes[:, None] / 3.0 )
    y = y.at[:, iq2:iq3].set( -0.5 * dlfdlq[None, :] * shearn[:, None] )
    y = y.at[:, iq3:].set( 0.0 )
    
    return y

@partial(jax.jit, static_argnames=("lmax","lmaxnu","nqmax","rtol","atol"))
def evolve_one_mode( *, y0, tau_start, tau_max, tau_out, param, kmode, lmax, lmaxnu, nqmax, rtol, atol ):

    model = diffrax.ODETerm( 
        lambda tau, y , params : 
            model_synchronous( tau=tau, yin=y, param=param, kmode=kmode, lmax=lmax, lmaxnu=lmaxnu, nqmax=nqmax ) 
    )

    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=tau_out)
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol) #, pcoeff=0.4, icoeff=0.3, dcoeff=0)
    sol = diffrax.diffeqsolve(
        terms=model,
        solver=solver,
        t0=tau_start,
        t1=tau_max,
        dt0=tau_start/2,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10010,
    )
    return sol.ys


# @partial(jax.jit, static_argnames=("num_k","lmax","lmaxnu","nqmax","rtol","atol"))
def evolve_perturbations( *, param, aexp_out, kmin : float, kmax : float, num_k : int, lmax : int = 12, lmaxnu : int = 12, nqmax : int = 15, rtol: float = 1e-5, atol: float = 1e-5 ):
    """evolve cosmological perturbations in the synchronous gauge

    Args:
        param (_type_): _description_
        aexp_out (_type_): _description_
        kmin (float): _description_
        kmax (float): _description_
        num_k (int): _description_
        lmax (int, optional): _description_. Defaults to 32.
        lmaxnu (int, optional): _description_. Defaults to 32.
        nqmax (int, optional): _description_. Defaults to 15.
        rtol (float, optional): _description_. Defaults to 1e-3.
        atol (float, optional): _description_. Defaults to 1e-4.

    Returns:
        _type_: _description_
    """
    kmodes = jnp.geomspace(kmin, kmax, num_k)
    nvar   = 7 + 3 * (lmax + 1) + nqmax * (lmaxnu + 1)

    # determine output times from aexp_out
    tau_out = jax.vmap( lambda a: param['tau_of_a_spline'](a) )(aexp_out)
    tau_start = 0.1 #jnp.minimum(1e-3 / jnp.max(kmodes), 0.1)
    tau_max = jnp.max(tau_out)
    nout = aexp_out.shape[0]
    param['nout'] = nout
    
    # set initial conditions
    y0 = adiabatic_ics( tau=tau_start, param=param, kmodes=kmodes, num_k=num_k, nvar=nvar, lmax=lmax, nqmax=nqmax )
    
    # solve ODEs 
    y1 = jax.vmap(
        lambda k_y0 : evolve_one_mode( y0=k_y0[1:], tau_start=tau_start, tau_max=tau_max, tau_out=tau_out, 
                                        param=param, kmode=k_y0[0], lmax=lmax, lmaxnu=lmaxnu, nqmax=nqmax, rtol=rtol, atol=atol ),
                                        in_axes=0
    )(jnp.append(kmodes[:,None],y0,axis=1))
    
    return y1, kmodes
