import jax
import jax.numpy as jnp
from pylinger_background import nu_perturb
from functools import partial
from jaxopt import Bisection
import diffrax as drx
import equinox as eqx

def compute_time_scales( *, k, a, param ):
    rhonu = param['rhonu_of_a_spline'].evaluate(a)

    grho = (
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaL'] * a**2
        + param['grhom'] * param['Omegak']
    )
    aprimeoa = jnp.sqrt(grho / 3.0)

    # ... Thomson opacity coefficient = n_e sigma_T
    akthom = 2.3048e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2

    # dkappa/dtau (dkappa/dtau = a n_e x_e sigma_T = a^{-2} n_e(today) x_e sigma_T in units of 1/Mpc) */
    # pvecthermo[pth->index_th_dkappa] = (1.+z) * (1.+z) * pth->n_e * x0 * sigmaTrescale * _sigma_ * _Mpc_over_m_;

    # ... Thomson opacity
    tau = param['tau_of_a_spline'].evaluate(a)
    xe = param['xe_of_tau_spline'].evaluate(tau)
    opac    = xe * akthom / a**2
    tauc    = 1. / opac
    tauh = 1./aprimeoa
    tauk = 1./k

    return tauh, tauk, tauc

@partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax'))
def model_synchronous(*, tau, yin, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax):     
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
    lmaxg : int
        maximum photon temperature hierarchy multipole
    lmaxgp : int
        maximum photon polarization hierarchy multipole
    lmaxr : int
        maximum massless neutrino hierarchy multipole
    lmaxnu : int
        maximum neutrino hierarchy multipole
    nqmax : int
        maximum number of momentum bins for massive neutrinos

    Returns
    -------
    f : array_like
        RHS of perturbation equations
    """
    Omegac = param['Omegam'] - param['Omegab']
    tempb_of_tau_interp = param['tempb_of_tau_spline']
    cs2_of_tau_interp = param['cs2_of_tau_spline']
    xe_of_tau_interp = param['xe_of_tau_spline']
    rhonu_of_a_interp = param['rhonu_of_a_spline']
    pnu_of_a_interp = param['pnu_of_a_spline']

    iq0 = 10 + lmaxg + lmaxgp + lmaxr
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

    # ... massless neutrinos
    deltar = y[ 9 + lmaxg + lmaxgp]
    thetar = y[10 + lmaxg + lmaxgp]
    shearr = y[11 + lmaxg + lmaxgp] / 2.0

    # ... evaluate thermodynamics
    tempb = tempb_of_tau_interp.evaluate(tau) #param['tempb_of_tau_spline']( tau )
    cs2 = cs2_of_tau_interp.evaluate(tau) #param['cs2_of_tau_spline']( tau )
    xe = xe_of_tau_interp.evaluate(tau) #param['xe_of_tau_spline']( tau )
    xeprime = xe_of_tau_interp.derivative(tau) #param['xe_of_tau_spline'].derivative( tau )

    # ... Photon mass density over baryon mass density
    photbar = param['grhog'] / (param['grhom'] * param['Omegab'] * a)
    pb43 = 4.0 / 3.0 * photbar

    # ... compute expansion rate
    rhonu = rhonu_of_a_interp.evaluate(a) #param['rhonu_of_a_spline']( a )
    pnu = pnu_of_a_interp.evaluate(a) #param['pnu_of_a_spline']( a ) 
    
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
    aprimeprimeoa = 0.5 * (aprimeoa**2 - gpres)

    # ... Thomson opacity coefficient
    akthom = 2.3048e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2

    # ... Thomson opacity
    opac    = xe * akthom / a**2
    tauc    = 1. / opac
    taucprime = tauc * (2*aprimeoa - xeprime/xe)
    F       = tauc / (1+pb43) #CLASS perturbations.c:10072
    Fprime    = taucprime/(1+pb43) + tauc*pb43*aprimeoa/(1+pb43)**2 #CLASS perturbations.c:10074


    # ... background scale factor evolution
    f = f.at[0].set( aprimeoa * a )
    
    # ... evaluate metric perturbations
    drhonu, dpnu, fnu, shearnu = nu_perturb( a, param['amnu'], y[iq0:iq1], y[iq1:iq2], y[iq2:iq3] )

    dgrho = (
        param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
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
        param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a
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

    # ... cdm equations of motion, MB95 eq. (42)
    deltacprime = -thetac - 0.5 * hprime
    f = f.at[3].set( deltacprime )
    thetacprime = -aprimeoa * thetac  # thetac = 0 in synchronous gauge!
    f = f.at[4].set( thetacprime )

    def calc_baryon_photon_uncoupled( f ):
        # === treat baryons and photons as uncoupled ================================================
        idxb = 5
        # --- baryon equations of motion, MB95 eqs. (66) ---------------------------------------------
        # ... baryon density, BLT11 eq. (2.1a)
        deltabprime = -thetab - 0.5 * hprime
        f = f.at[idxb+0].set( deltabprime )
        # ... baryon velocity, BLT11 eq. (2.1b)
        thetabprime = -aprimeoa * thetab + kmode**2 * cs2 * deltab \
                    + pb43 * opac * (thetag - thetab)
        f = f.at[idxb+1].set( thetabprime )

        # --- photon equations of motion, MB95 eqs. (63) ---------------------------------------------
        idxg  = 7
        idxgp = 7 + (lmaxg+1)
        # ... polarization term
        polter = y[idxg+2] + y[idxgp+0] + y[idxgp+2]
        # ... photon density, BLT11 eq. (2.4a)
        deltagprime = 4.0 / 3.0 * (-thetag - 0.5 * hprime)
        f = f.at[idxg+0].set( deltagprime )
        # ... photon velocity, BLT11 eq. (2.4b)
        thetagprime = kmode**2 * (0.25 * deltag - s2_squared * shearg) \
                    - opac * (thetag - thetab)
        f = f.at[idxg+1].set( thetagprime )
        # ... photon shear, BLT11 eq. (2.4c)
        sheargprime = 8./15. * (thetag+kmode**2*alpha) -3/5*kmode*s_l3/s_l2*y[idxg+3] \
                    - opac*(y[idxg+2]-0.1*s_l2*polter)
        f = f.at[idxg+2].set( sheargprime )

        # photon temperature l>=3, BLT11 eq. (2.4d)
        ell  = jnp.arange(3, lmaxg )
        f = f.at[idxg+ell].set( kmode  / (2 * ell + 1) * (ell * y[idxg+ell-1] - (ell + 1) * y[idxg+ell+1]) - opac * y[idxg+ell] )
        # photon temperature hierarchy truncation, BLT11 eq. (2.5)
        f = f.at[idxg+lmaxg].set( kmode * y[idxg+lmaxg-1] - (lmaxg + 1) / tau * y[idxg+lmaxg] - opac * y[idxg+lmaxg] )
    
        # polarization equations, BLT11 eq. (2.4e)
        # photon polarization l=0
        f = f.at[idxgp+0].set( -kmode * y[idxgp+1] - opac * y[idxgp] + 0.5 * opac * polter )
        # photon polarization l=1delta_nu = jnp.sum( y[iq0:iq1] * )
        f = f.at[idxgp+1].set( kmode / 3.0 * (y[idxgp] - 2.0 * y[idxgp+2]) - opac * y[idxgp+1] )
        # photon polarization l=2
        f = f.at[idxgp+2].set( kmode * (0.4 * y[idxgp+1] - 0.6 * y[idxgp+3]) - opac * (y[idxgp+2] - 0.1 * s_l2 * polter))
        # photon polarization lmax>l>=3
        ell  = jnp.arange(3, lmaxgp)
        f = f.at[idxgp+ell].set( kmode  / (2 * ell + 1) * (ell * y[idxgp+ell-1] - (ell + 1) * y[idxgp+ell+1]) - opac * y[idxgp+ell] )
        # photon polarization hierarchy truncation
        f = f.at[idxgp+lmaxgp].set( kmode * y[idxgp+lmaxgp-1] - (lmaxgp + 1) / tau * y[idxgp+lmaxgp] - opac * y[idxgp+lmaxgp] )
        
        return f
    
    def calc_baryon_photon_tca_CLASS( f ):
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
        idxb = 5
        # ... baryon density
        deltabprime = -thetab - 0.5 * hprime
        f = f.at[idxb+0].set( deltabprime )
        # ... baryon velocity, BLT 11, eq. (2.7a)
        thetabprime = (-aprimeoa * thetab + kmode**2 * (cs2 * deltab + pb43 * (0.25 * deltag - s2_squared * tca_shearg)) + pb43 * tca_slip) / (1.0 + pb43)
        f = f.at[idxb+1].set( thetabprime )

        # --- photon equations of motion -------------------------------------------------------------
        idxg  = 7
        # ... photon density
        deltagprime = 4.0 / 3.0 * (-thetag - 0.5 * hprime)
        f = f.at[idxg+0].set( deltagprime )
        # ... photon velocity, BLT11, eq. (2.7b)
        thetagprime = -(thetabprime + aprimeoa * thetab - kmode**2 * cs2 * deltab) / pb43  + kmode**2 * (0.25 * deltag -s2_squared * tca_shearg)
        f = f.at[idxg+1].set( thetagprime )

        return f
    
    def calc_baryon_photon_tca_MB( f ):
        idxb = 5
        idxg = 7

        deltabprime = -thetab - 0.5 * hprime
        f = f.at[idxb+0].set( deltabprime )

        thetabprime = (-aprimeoa * thetab + kmode**2 * cs2 * deltab + kmode**2 * pb43 * (0.25 * deltag - s2_squared * shearg)) / (1.0 + pb43)
        
        deltagprime = 4.0 / 3.0 * (-thetag - 0.5 * hprime)
        f = f.at[idxg+0].set( deltagprime )

        aprimeprimeoa = 0.5*(aprimeoa**2-gpres)
        slip = 2*pb43/(1+pb43)*aprimeoa*(thetab-thetag) + 1/opac *(-aprimeprimeoa*thetab-aprimeoa*kmode**2/2*deltag + kmode**2*(cs2*deltabprime-0.25*deltagprime))/(1+pb43)
        thetabprime += pb43/(1+pb43)*slip
        f = f.at[idxb+1].set( thetabprime )

        thetagprime = (-thetabprime-aprimeoa*thetab+kmode**2*cs2*deltab)/pb43 + kmode**2 *(0.25*deltag-s2_squared*shearg)
        f = f.at[idxg+1].set( thetagprime )
        return f
    
    # --- check if we are in the tight coupling regime -----------------------------------------------
    tight_coupling_trigger_tau_c_over_tau_h=0.015       # value taken from CLASS
    tight_coupling_trigger_tau_c_over_tau_k=0.010       # value taken from CLASS
    radiation_streaming_trigger_tau_c_over_tau = 5.0    # value taken from CLASS, not used (yet)
    radiation_streaming_trigger_tau_over_tau_k = 45.    # value taken from CLASS, not used (yet)

    tauh = 1./aprimeoa
    tauk = 1./kmode
    
    f = jax.lax.cond(
        jnp.logical_or( tauc/tauk > tight_coupling_trigger_tau_c_over_tau_k,
            jnp.logical_and( tauc/tauh > tight_coupling_trigger_tau_c_over_tau_h,
                            tauc/tauk > 0.1*tight_coupling_trigger_tau_c_over_tau_k)),
        calc_baryon_photon_uncoupled, calc_baryon_photon_tca_CLASS, f )
    
    # f = calc_baryon_photon_tca( f )
    
    
    # --- Massless neutrino equations of motion -------------------------------------------------------
    idxr = 9 + lmaxg + lmaxgp
    deltarprime = 4.0 / 3.0 * (-thetar - 0.5 * hprime)
    f = f.at[idxr+0].set( deltarprime )
    thetarprime = kmode**2 * (0.25 * deltar - shearr)
    f = f.at[idxr+1].set( thetarprime )
    shearrprime = 8./15. * (thetar + kmode**2 * alpha) - 0.6 * kmode * y[idxr+3]
    f = f.at[idxr+2].set( shearrprime )
    ell = jnp.arange(3, lmaxr)
    f = f.at[idxr+ell].set( kmode / (2 * ell + 1) * (ell * y[idxr+ell-1] - (ell + 1) * y[idxr+ell+1]) )
    
    # ... truncate moment expansion
    f = f.at[idxr+lmaxr].set( kmode * y[idxr+lmaxr-1] - (lmaxr + 1) / tau * y[idxr+lmaxr] )

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

@partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr'))
def model_synchronous_neutrino_cfa(*, tau, yin, param, kmode, lmaxg, lmaxgp, lmaxr):
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
    lmaxg : int
        maximum photon temperature hierarchy multipole
    lmaxgp : int
        maximum photon polarization hierarchy multipole
    lmaxr : int
        maximum massless neutrino hierarchy multipole
    lmaxnu : int
        maximum neutrino hierarchy multipole
    nqmax : int
        maximum number of momentum bins for massive neutrinos

    Returns
    -------
    f : array_like
        RHS of perturbation equations
    """

    Omegac = param['Omegam'] - param['Omegab']

    iq0 = 10 + lmaxg + lmaxgp + lmaxr

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

    # ... massless neutrinos
    deltar = y[ 9 + lmaxg + lmaxgp]
    thetar = y[10 + lmaxg + lmaxgp]
    shearr = y[11 + lmaxg + lmaxgp] / 2.0
    
    # ... massive neutrinos
    deltanu = y[iq0+0]
    thetanu = y[iq0+1]
    shearnu = y[iq0+2]

    # ... evaluate thermodynamics
    tempb = param['tempb_of_tau_spline'].evaluate( tau )
    cs2 = param['cs2_of_tau_spline'].evaluate( tau )
    xe = param['xe_of_tau_spline'].evaluate( tau )
    xeprime = param['xe_of_tau_spline'].derivative( tau )


    # ... Photon mass density over baryon mass density
    photbar = param['grhog'] / (param['grhom'] * param['Omegab'] * a)
    pb43 = 4.0 / 3.0 * photbar

    # ... massive neutrino thermodynamics
    rhonu = param['rhonu_of_a_spline'].evaluate( a )
    pnu = param['pnu_of_a_spline'].evaluate( a ) 
    rho_plus_p_nu = rhonu+pnu
    ppseudonu = param['ppseudonu_of_a_spline'].evaluate( a ) # pseudo pressure from CLASS IV, LT11
    
    w_nu = pnu / rhonu
    ca2_nu = w_nu/3.0/(1.0+w_nu)*(5.0-ppseudonu/pnu)  # eq. (3.3) in LT11
    ceff2_nu = ca2_nu
    cvis2_nu = 3.*w_nu*ca2_nu # CLASS's fluid approximation eq. (3.15c) in LT11
    cg2_nu = w_nu-w_nu/3.0/(1.0+w_nu)*(3.0*w_nu-2.0+ppseudonu/pnu) # CLASS perturbation.c:7078
    dpnu = cg2_nu * deltanu
    
    # ... compute expansion rate
    grho = (
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaL'] * a**2
        + param['grhom'] * param['Omegak']
    )

    gpres = (
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 - param['grhom'] * param['OmegaL'] * a**2

    aprimeoa = jnp.sqrt(grho / 3.0)                 # Friedmann I
    aprimeprimeoa = 0.5 * (aprimeoa**2 - gpres)     # Friedmann II

    # ... Thomson opacity coefficient
    akthom = 2.3048e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2

    # ... Thomson opacity
    opac    = xe * akthom / a**2
    tauc    = 1. / opac
    taucprime = tauc * (2*aprimeoa - xeprime/xe)
    F       = tauc / (1+pb43) #CLASS perturbations.c:10072
    Fprime  = taucprime/(1+pb43) + tauc*pb43*aprimeoa/(1+pb43)**2 #CLASS perturbations.c:10074


    # ... background scale factor evolution
    f = f.at[0].set( aprimeoa * a )
    
    
    dgrho = (
        param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + (param['grhog'] * deltag + param['grhor'] * (param['Neff'] * deltar + param['Nmnu'] * rhonu * deltanu)) / a**2
    )
    dgpres = (
        param['grhog'] * deltag + param['grhor'] * param['Neff'] * deltar
    ) / a**2 / 3.0 + param['grhor'] * param['Nmnu'] * dpnu / a**2

    dahprimedtau = -(dgrho + 3.0 * dgpres) * a
    
    f = f.at[1].set( dahprimedtau )

    # ... force energy conservation
    hprime = (2.0 * kmode**2 * eta + dgrho) / aprimeoa

    dgtheta = (
        param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a
        + 4.0 / 3.0 * (param['grhog'] * thetag + param['Neff'] * param['grhor'] * thetar) / a**2
        + param['Nmnu'] * param['grhor'] * rho_plus_p_nu * thetanu / a**2
    )
    etaprime = 0.5 * dgtheta / kmode**2
    alpha  = (hprime + 6.*etaprime)/2./kmode**2
    f = f.at[2].set( etaprime )

    dgshear = (
        4.0 / 3.0 * (param['grhog'] * shearg + param['Neff'] * param['grhor'] * shearr) / a**2
        + param['Nmnu'] * param['grhor'] * rho_plus_p_nu * shearnu / a**2
    )
    
    alphaprime = -3*dgshear/(2*kmode**2) + eta - 2*aprimeoa*alpha
    alphaprime -=  9/2 * a**2/kmode**2 * 4/3*16/45/opac * (thetag+kmode**2*alpha) * param['grhog']

    # ... cdm equations of motion, MB95 eq. (42)
    deltacprime = -thetac - 0.5 * hprime
    f = f.at[3].set( deltacprime )
    thetacprime = -aprimeoa * thetac  # thetac = 0 in synchronous gauge!
    f = f.at[4].set( thetacprime )


    def calc_baryon_photon_uncoupled( f ):
        # === treat baryons and photons as uncoupled ================================================
        idxb = 5
        # --- baryon equations of motion, MB95 eqs. (66) ---------------------------------------------
        # ... baryon density, BLT11 eq. (2.1a)
        deltabprime = -thetab - 0.5 * hprime
        f = f.at[idxb+0].set( deltabprime )
        # ... baryon velocity, BLT11 eq. (2.1b)
        thetabprime = -aprimeoa * thetab + kmode**2 * cs2 * deltab \
                    + pb43 * opac * (thetag - thetab)
        f = f.at[idxb+1].set( thetabprime )

        # --- photon equations of motion, MB95 eqs. (63) ---------------------------------------------
        idxg  = 7
        idxgp = 7 + (lmaxg+1)
        # ... polarization term
        polter = y[idxg+2] + y[idxgp+0] + y[idxgp+2]
        # ... photon density, BLT11 eq. (2.4a)
        deltagprime = 4.0 / 3.0 * (-thetag - 0.5 * hprime)
        f = f.at[idxg+0].set( deltagprime )
        # ... photon velocity, BLT11 eq. (2.4b)
        thetagprime = kmode**2 * (0.25 * deltag - s2_squared * shearg) \
                    - opac * (thetag - thetab)
        f = f.at[idxg+1].set( thetagprime )
        # ... photon shear, BLT11 eq. (2.4c)
        sheargprime = 8./15. * (thetag+kmode**2*alpha) -3/5*kmode*s_l3/s_l2*y[idxg+3] \
                    - opac*(y[idxg+2]-0.1*s_l2*polter)
        f = f.at[idxg+2].set( sheargprime )

        # photon temperature l>=3, BLT11 eq. (2.4d)
        ell  = jnp.arange(3, lmaxg )
        f = f.at[idxg+ell].set( kmode  / (2 * ell + 1) * (ell * y[idxg+ell-1] - (ell + 1) * y[idxg+ell+1]) - opac * y[idxg+ell] )
        # photon temperature hierarchy truncation, BLT11 eq. (2.5)
        f = f.at[idxg+lmaxg].set( kmode * y[idxg+lmaxg-1] - (lmaxg + 1) / tau * y[idxg+lmaxg] - opac * y[idxg+lmaxg] )
    
        # polarization equations, BLT11 eq. (2.4e)
        # photon polarization l=0
        f = f.at[idxgp+0].set( -kmode * y[idxgp+1] - opac * y[idxgp] + 0.5 * opac * polter )
        # photon polarization l=1delta_nu = jnp.sum( y[iq0:iq1] * )
        f = f.at[idxgp+1].set( kmode / 3.0 * (y[idxgp] - 2.0 * y[idxgp+2]) - opac * y[idxgp+1] )
        # photon polarization l=2
        f = f.at[idxgp+2].set( kmode * (0.4 * y[idxgp+1] - 0.6 * y[idxgp+3]) - opac * (y[idxgp+2] - 0.1 * s_l2 * polter))
        # photon polarization lmax>l>=3
        ell  = jnp.arange(3, lmaxgp)
        f = f.at[idxgp+ell].set( kmode  / (2 * ell + 1) * (ell * y[idxgp+ell-1] - (ell + 1) * y[idxgp+ell+1]) - opac * y[idxgp+ell] )
        # photon polarization hierarchy truncation
        f = f.at[idxgp+lmaxgp].set( kmode * y[idxgp+lmaxgp-1] - (lmaxgp + 1) / tau * y[idxgp+lmaxgp] - opac * y[idxgp+lmaxgp] )
        
        return f
    
    def calc_baryon_photon_tca_CLASS( f ):
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
        idxb = 5
        # ... baryon density
        deltabprime = -thetab - 0.5 * hprime
        f = f.at[idxb+0].set( deltabprime )
        # ... baryon velocity, BLT 11, eq. (2.7a)
        thetabprime = (-aprimeoa * thetab + kmode**2 * (cs2 * deltab + pb43 * (0.25 * deltag - s2_squared * tca_shearg)) + pb43 * tca_slip) / (1.0 + pb43)
        f = f.at[idxb+1].set( thetabprime )

        # --- photon equations of motion -------------------------------------------------------------
        idxg  = 7
        # ... photon density
        deltagprime = 4.0 / 3.0 * (-thetag - 0.5 * hprime)
        f = f.at[idxg+0].set( deltagprime )
        # ... photon velocity, BLT11, eq. (2.7b)
        thetagprime = -(thetabprime + aprimeoa * thetab - kmode**2 * cs2 * deltab) / pb43  + kmode**2 * (0.25 * deltag -s2_squared * tca_shearg)
        f = f.at[idxg+1].set( thetagprime )

        return f
    
    def calc_baryon_photon_tca_MB( f ):
        idxb = 5
        idxg = 7

        deltabprime = -thetab - 0.5 * hprime
        f = f.at[idxb+0].set( deltabprime )

        thetabprime = (-aprimeoa * thetab + kmode**2 * cs2 * deltab + kmode**2 * pb43 * (0.25 * deltag - s2_squared * shearg)) / (1.0 + pb43)
        
        deltagprime = 4.0 / 3.0 * (-thetag - 0.5 * hprime)
        f = f.at[idxg+0].set( deltagprime )

        aprimeprimeoa = 0.5*(aprimeoa**2-gpres)
        slip = 2*pb43/(1+pb43)*aprimeoa*(thetab-thetag) + 1/opac *(-aprimeprimeoa*thetab-aprimeoa*kmode**2/2*deltag + kmode**2*(cs2*deltabprime-0.25*deltagprime))/(1+pb43)
        thetabprime += pb43/(1+pb43)*slip
        f = f.at[idxb+1].set( thetabprime )

        thetagprime = (-thetabprime-aprimeoa*thetab+kmode**2*cs2*deltab)/pb43 + kmode**2 * (0.25*deltag - s2_squared*shearg)
        f = f.at[idxg+1].set( thetagprime )
        return f
    
    # --- check if we are in the tight coupling regime -----------------------------------------------
    tight_coupling_trigger_tau_c_over_tau_h=0.015       # value taken from CLASS
    tight_coupling_trigger_tau_c_over_tau_k=0.010       # value taken from CLASS
    radiation_streaming_trigger_tau_c_over_tau = 5.0    # value taken from CLASS, not used (yet)
    radiation_streaming_trigger_tau_over_tau_k = 45.    # value taken from CLASS, not used (yet)

    tauh = 1./aprimeoa  # TBC: or 1./(aprimeoa*a)?
    tauk = 1./kmode
    
    f = jax.lax.cond(
        jnp.logical_or( tauc/tauk > tight_coupling_trigger_tau_c_over_tau_k,
            jnp.logical_and( tauc/tauh > tight_coupling_trigger_tau_c_over_tau_h,
                            tauc/tauk > 0.1*tight_coupling_trigger_tau_c_over_tau_k)),
        calc_baryon_photon_uncoupled, calc_baryon_photon_tca_CLASS, f )
    
    
    # --- Massless neutrino equations of motion -------------------------------------------------------
    idxr = 9 + lmaxg + lmaxgp
    deltarprime = 4.0 / 3.0 * (-thetar - 0.5 * hprime)
    f = f.at[idxr+0].set( deltarprime )
    thetarprime = kmode**2 * (0.25 * deltar - shearr)
    f = f.at[idxr+1].set( thetarprime )
    shearrprime = 8./15. * (thetar + kmode**2 * alpha) - 0.6 * kmode * y[idxr+3]
    f = f.at[idxr+2].set( shearrprime )
    ell = jnp.arange(3, lmaxr)
    f = f.at[idxr+ell].set( kmode / (2 * ell + 1) * (ell * y[idxr+ell-1] - (ell + 1) * y[idxr+ell+1]) )
    
    # ... truncate moment expansion
    f = f.at[idxr+lmaxr].set( kmode * y[idxr+lmaxr-1] - (lmaxr + 1) / tau * y[idxr+lmaxr] )

    # --- Massive neutrino equations of motion, fluid approximation -----------------------------------
    # LT11: CLASS IV: ncdm, Lesgourgues & Tram 2011, https://arxiv.org/abs/1104.2935
    # LT11, eq. (3.1a)
    deltanuprime = (1+w_nu)*(-thetanu - 0.5 * hprime) - 3 * aprimeoa * (ceff2_nu-w_nu) * deltanu
    f = f.at[iq0+0].set( deltanuprime )
    # LT11, eq. (3.1b)
    thetanuprime = -aprimeoa * (1-3*ca2_nu) * thetanu + kmode**2 * (ceff2_nu/(1+w_nu) * deltanu - shearnu)
    f = f.at[iq0+1].set( thetanuprime )
    # LT11, eq. (3.15c)
    sigmanuprime = -3*(1/tau + aprimeoa*(2/3 - ca2_nu - ppseudonu/pnu/3)) * shearnu \
        + 8/3 * cvis2_nu/(1+w_nu) * s_l2 * (thetanu + 0.5*hprime)
    f = f.at[iq0+2].set( sigmanuprime )

    return f.flatten()


# @partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax'))
def neutrino_convert_to_fluid(*, tau, yin, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax ):
    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    
    a = yin[0]
    
    nvarnu = 10 + lmaxg + lmaxgp + lmaxr + 3
    y = jnp.zeros((nvarnu))
    
    drhonu, dpnu, fnu, shearnu = nu_perturb( a, param['amnu'], yin[iq0:iq1], yin[iq1:iq2], yin[iq2:iq3] )
    rhonu = param['rhonu_of_a_spline'].evaluate( a )
    pnu = param['pnu_of_a_spline'].evaluate( a ) 
    rho_plus_p = rhonu + pnu
    
    y = y.at[:iq0].set( yin[:iq0] )

    y = y.at[iq0+0].set( drhonu / rhonu)
    y = y.at[iq0+1].set( kmode * fnu / rho_plus_p)
    y = y.at[iq0+2].set( shearnu / rho_plus_p )
    
    return y

# @partial(jax.jit, static_argnames=('num_k', 'nvar', 'lmax', 'nqmax'))
def adiabatic_ics_one_mode( *, tau: float, param, kmode, nvar, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax):
    """Initial conditions for adiabatic perturbations"""
    Omegac = param['Omegam'] - param['Omegab']

    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax

    y = jnp.zeros((nvar))
    a = param['a_of_tau_spline'].evaluate(tau)

    rhonu = param['rhonu_of_a_spline'].evaluate(a)
    pnu = param['pnu_of_a_spline'].evaluate(a)
    
    grho = (
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaL'] * a**2
        + param['grhom'] * param['Omegak']
    )
    gpres = (
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 - param['grhom'] * param['OmegaL'] * a**2

    s = grho + gpres

    fracnu = param['grhor'] * (param['Neff'] + param['Nmnu']) * 4.0 / 3.0 / a**2 / s

    # ... use yrad=rho_matter/rho_rad to correct initial conditions for matter+radiation
    yrad = (
        param['grhom'] * param['Omegam'] * a
        / (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu))
    )

    # .. isentropic ("adiabatic") initial conditions
    psi = -1.0
    C = (15.0 + 4.0 * fracnu) / 20.0 * psi
    akt2 = (kmode * tau)**2
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
    shearr = 4.0 / 15.0 * kmode**2 / s * psi * (1.0 + 7.0 / 36.0 * yrad)
    shearn = shearr
    ahprime = 2.0 * C * kmode**2 * tau * a * (1.0 - 0.3 * yrad)

    # ... metric
    y = y.at[0].set( a )
    y = y.at[1].set( ahprime )
    y = y.at[2].set( eta )

    # .. CDM
    y = y.at[3].set( deltac )
    y = y.at[4].set( thetac )

    # .. baryons
    y = y.at[5].set( deltab )
    y = y.at[6].set( thetab )

    # ... Photons (total intensity and polarization)
    y = y.at[7].set( deltag )
    y = y.at[8].set( thetag )
    # shear and polarization are zero at the initial time
    
    # ... massless neutrinos
    y = y.at[ 9 + lmaxg + lmaxgp].set( deltar )
    y = y.at[10 + lmaxg + lmaxgp].set( thetar )
    y = y.at[11 + lmaxg + lmaxgp].set( shearr * 2.0 )
    # higher moments are zero at the initial time

    # ... massive neutrinos
    # if params.cp.Nmnu > 0:
    q = jnp.arange(1, nqmax + 1) - 0.5  # so dq == 1
    aq = a * param['amnu'] / q
    v = 1 / jnp.sqrt(1 + aq**2)
    akv = jnp.outer(kmode, v)
    dlfdlq = -q / (1.0 + jnp.exp(-q))
    y = y.at[iq0:iq1].set( -0.25 * dlfdlq * deltan )
    y = y.at[iq1:iq2].set( -dlfdlq * thetan / v / kmode / 3.0 )
    y = y.at[iq2:iq3].set( -0.5 * dlfdlq * shearn )
    # higher moments are zero at the initial time
    
    return y

# @jax.jit
def determine_starting_time( *, param, k ):
    # ADOPTED from CLASS:
    # largest wavelengths start being sampled when universe is sufficiently opaque. This is quantified in terms of the ratio of thermo to hubble time scales, 
    # \f$ \tau_c/\tau_H \f$. Start when start_largek_at_tau_c_over_tau_h equals this ratio. Decrease this value to start integrating the wavenumbers earlier 
    # in time.
    start_small_k_at_tau_c_over_tau_h = 0.0015 

    # ADOPTED from CLASS:
    #  largest wavelengths start being sampled when mode is sufficiently outside Hubble scale. This is quantified in terms of the ratio of hubble time scale 
    #  to wavenumber time scale, \f$ \tau_h/\tau_k \f$ which is roughly equal to (k*tau). Start when this ratio equals start_large_k_at_tau_k_over_tau_h. 
    #  Decrease this value to start integrating the wavenumbers earlier in time. 
    start_large_k_at_tau_h_over_tau_k = 0.07

    tau0 = param['taumin']
    tau1 = param['tau_of_a_spline'].evaluate( 1e-2 ) # don't start after a=0.01
    akthom = 2.3048e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2

    tau_k = 1.0/k

    def get_tauc_tauH( tau, param ):
        xe = param['xe_of_tau_spline'].evaluate( tau )
        a = param['a_of_tau_spline'].evaluate( tau )
        opac = xe * akthom / a**2

        rhonu = param['rhonu_of_a_spline'].evaluate( a )
        aprimeoa = jnp.sqrt( (param['grhom'] * param['Omegam'] / a
                        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
                        + param['grhom'] * param['OmegaL'] * a**2
                        + param['grhom'] * param['Omegak'])/3.0 )
        return 1.0/opac, 1.0/aprimeoa

    def get_tauH( tau, param ):
        a = param['a_of_tau_spline'].evaluate( tau )

        rhonu = param['rhonu_of_a_spline'].evaluate( a )
        aprimeoa = jnp.sqrt( (param['grhom'] * param['Omegam'] / a
                        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
                        + param['grhom'] * param['OmegaL'] * a**2
                        + param['grhom'] * param['Omegak'])/3.0 )
        return 1.0/aprimeoa

    # condition for small k: tau_c(a) / tau_H(a) < start_small_k_at_tau_c_over_tau_h
    def cond_small_k( logtau, param ):
        tau_c, tau_H = get_tauc_tauH( jnp.exp(logtau), param )
        return tau_c/tau_H/start_small_k_at_tau_c_over_tau_h - 1.0

    # condition for large k: tau_H(a) / tau_k < start_large_k_at_tau_k_over_tau_h
    def cond_large_k( logtau, param ):
        tau_H = get_tauH( jnp.exp(logtau), param )
        return tau_H/tau_k/start_large_k_at_tau_h_over_tau_k - 1.0

    
    logtau_small_k =  Bisection(optimality_fun=lambda lt : cond_small_k(lt,param), lower=jnp.log(tau0), upper=jnp.log(tau1), maxiter=8, check_bracket=False).run().params
    logtau_large_k =  Bisection(optimality_fun=lambda lt : cond_large_k(lt,param), lower=jnp.log(tau0), upper=jnp.log(tau1), maxiter=8, check_bracket=False).run().params

    return jnp.exp(jnp.minimum(logtau_small_k, logtau_large_k))


class VectorField(eqx.Module):
    model: eqx.Module

    def __call__(self, t, y, args):
        return self.model(t, y, args)
    

@partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax'))
def evolve_one_mode( *, tau_max, tau_out, param, kmode, 
                        lmaxg : int = 12, lmaxgp : int = 12, lmaxr : int = 17, lmaxnu : int = 17, \
                        nqmax : int = 15, rtol: float = 1e-3, atol: float = 1e-4 ):

    # ... wrapper for model1: synchronous gauge without any optimizations, effectively this is Ma & Bertschinger 1995 
    # ... plus CLASS tight coupling approximation (Blas, Lesgourgues & Tram 2011, CLASS II)
    model1_ = VectorField( 
        lambda tau, y , params : model_synchronous( tau=tau, yin=y, param=params[0], kmode=params[1],   
                                                   lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax ) )
    model1 = drx.ODETerm( model1_ )
    
    # ... wrapper for model2: synchronous gauge with neutrino fluid approximation, following Lesgourgues & Tram 2011 (CLASS IV)
    model2_ = VectorField( 
        lambda tau, y , params :  model_synchronous_neutrino_cfa( tau=tau, yin=y, param=params[0], kmode=params[1], 
                                                                  lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr ) )
    model2 = drx.ODETerm( model2_ )
    
    # ... determine the number of active variables (i.e. the number of equations), absent any optimizations
    nvar   = 7 + (lmaxg + 1) + (lmaxgp + 1) + (lmaxr + 1) + nqmax * (lmaxnu + 1)

    # ... determine starting time
    tau_start = determine_starting_time( param=param, k=kmode )
    tau_start = jnp.minimum( jnp.min(tau_out), tau_start )

    # ... set adiabatic ICs
    y0 = adiabatic_ics_one_mode( tau=tau_start, param=param, kmode=kmode, nvar=nvar, 
                       lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax )


    # ... switch between model1 and model2 depending on tau
    nu_fluid_trigger_tau_over_tau_k = 31 # value taken from default CLASS settings

    tauk = 1./kmode
    tau_neutrino_cfa = jnp.minimum(tauk * nu_fluid_trigger_tau_over_tau_k, 0.999*tau_max) # don't go to tau_max so that all modes are converted to massive neutrino approx

    # create list of saveat times for first and second part of evolution
    saveat1 = drx.SaveAt(ts= jnp.where(tau_out<tau_neutrino_cfa,tau_out,tau_neutrino_cfa) )
    saveat2 = drx.SaveAt(ts= jnp.where(tau_out>=tau_neutrino_cfa,tau_out,tau_neutrino_cfa) )
    
    # create solver, we use the Kvaerno5 solver, which is a 5th order implicit solver
    solver = drx.Kvaerno5()
    
    # create stepsize controller, we use a PID controller
    stepsize_controller = drx.PIDController(rtol=rtol, atol=atol)#, pcoeff=0.4, icoeff=0.3, dcoeff=0)
    
    # solve before neutrinos become fluid
    sol1 = drx.diffeqsolve(
        terms=model1,
        solver=solver,
        t0=tau_start,
        t1=tau_neutrino_cfa,
        dt0=jnp.minimum(tau_start/2, 0.5*(tau_neutrino_cfa-tau_start)),
        y0=y0,
        saveat=saveat1,
        stepsize_controller=stepsize_controller,
        max_steps=4096,
        args=(param, kmode, ),
        # adjoint=drx.RecursiveCheckpointAdjoint(),
        adjoint=drx.DirectAdjoint(),
    )
    
    # convert neutrinos to fluid by integrating over the momentum bins
    y0_neutrino_cfa = neutrino_convert_to_fluid( tau=sol1.ts[-1], yin=sol1.ys[-1,:], param=param, kmode=kmode, 
                                            lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax )
    
    # solve after neutrinos become fluid
    sol2 = drx.diffeqsolve(
        terms=model2,
        solver=solver,
        t0=sol1.ts[-1],
        t1=tau_max,
        dt0=jnp.minimum(tau_neutrino_cfa*0.5, 0.5*(tau_max - sol1.ts[-1])),
        y0=y0_neutrino_cfa,
        saveat=saveat2,
        stepsize_controller=stepsize_controller,
        max_steps=4096,
        args=( param, kmode, ),
        # adjoint=drx.RecursiveCheckpointAdjoint(),
        adjoint=drx.DirectAdjoint(),
    )
    y1_converted = jax.vmap( 
        lambda tau, yin : neutrino_convert_to_fluid(tau=tau, yin=yin, param=param, kmode=kmode, lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax ), 
        in_axes=0, out_axes=0 )( sol1.ts, yin=sol1.ys )
    
    # solve after neutrinos become fluid
    return jnp.where( tau_out[:,None]<tau_neutrino_cfa, y1_converted, sol2.ys )


@partial(jax.jit, static_argnames=("num_k","lmaxg","lmaxgp", "lmaxr", "lmaxnu","nqmax","rtol","atol"))
def evolve_perturbations( *, param, aexp_out, kmin : float, kmax : float, num_k : int, \
                         lmaxg : int = 12, lmaxgp : int = 12, lmaxr : int = 17, lmaxnu : int = 17, \
                         nqmax : int = 15, rtol: float = 1e-4, atol: float = 1e-7 ):
    """evolve cosmological perturbations in the synchronous gauge

    Parameters
    ----------
    param : dict
        dictionary of parameters and interpolated functions
    aexp_out : array
        array of scale factors at which to output
    kmin : float
        minimum wavenumber
    kmax : float
        maximum wavenumber
    num_k : int
        number of wavenumbers
    lmaxg : int
        maximum multipole for photon temperature
    lmaxgp : int
        maximum multipole for photon polarization
    lmaxr : int
        maximum multipole for massless neutrinos
    lmaxnu : int
        maximum multipole for massive neutrinos
    nqmax : int
        number of momentum bins for massive neutrinos
    rtol : float
        relative tolerance for ODE solver
    atol : float
        absolute tolerance for ODE solver

    Returns
    -------
    y : array
        array of shape (num_k, nout, nvar) containing the perturbations
    k : array
        array of shape (num_k) containing the wavenumbers
    """
    kmodes = jnp.geomspace(kmin, kmax, num_k)
    

    # determine output times from aexp_out
    tau_out = jax.vmap( lambda a: param['tau_of_a_spline'].evaluate(a) )(aexp_out)
    tau_max = jnp.max(tau_out)
    nout = aexp_out.shape[0]
    param['nout'] = nout
    
    # set up ICs and solve ODEs for all the modes
    y1 = jax.vmap(
        lambda k : evolve_one_mode( tau_max=tau_max, tau_out=tau_out, 
                                    param=param, kmode=k, lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, 
                                    lmaxnu=lmaxnu, nqmax=nqmax, rtol=rtol, atol=atol ),
                                    in_axes=0
    )(kmodes)
    
    return y1, kmodes
