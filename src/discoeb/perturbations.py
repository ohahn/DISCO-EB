import jax
import jax.numpy as jnp

from .util import lngamma_complex_e, root_find_bisect, savgol_filter
from .cosmo import get_neutrino_momentum_bins, get_aprimeoa

import diffrax as drx
from jaxtyping import Array, PyTree, Scalar
import equinox as eqx

from functools import partial
import jax.flatten_util as fu


# @partial( jax.jit, static_argnames=('nqmax0',) )
def nu_perturb( a : float, amnu: float, psi0: jax.Array, psi1 : jax.Array, psi2 : jax.Array, nqmax : int ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """ Compute the perturbations of density, energy flux, pressure, and
        shear stress of one flavor of massive neutrinos, in units of the mean
        density of one flavor of massless neutrinos, by integrating over 
        momentum.

    Args:
        a (float): scale factor
        amnu (float): neutrino mass in units of neutrino temperature (m_nu*c**2/(k_B*T_nu0).
        psi0 (jax.Array): l=0 neutrino perturbations for all momentum bins
        psi1 (jax.Array): l=1 neutrino perturbations for all momentum bins
        psi2 (jax.Array): l=2 neutrino perturbations for all momentum bins
        nq (int, optional): _description_. Defaults to 1000.
        qmax (float, optional): _description_. Defaults to 30..

    Returns:
        _type_: drhonu, dpnu, fnu, shearnu
    """
    
    q, w = get_neutrino_momentum_bins( nqmax )
    aq = a * amnu / q
    v = 1 / jnp.sqrt(1 + aq**2)

    drhonu = jnp.sum(w * psi0 / v)
    dpnu = jnp.sum(w * psi0 * v) / 3
    fnu = jnp.sum(w * psi1) 
    shearnu = jnp.sum(w * psi2 * v) * 2 / 3

    return drhonu, dpnu, fnu, shearnu


@partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax'))
def model_synchronous(*, tau, y, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax ):     
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
        wavenumber of modef
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
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax

    # y = jnp.copy(yin)
    f = jnp.zeros_like( y )

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

    #ahprime = y[1]
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

    # ... quintessence field
    deltaq = y[-2]
    thetaq = y[-1]

    # ... evaluate thermodynamics
    # tempb   = param['tempba_of_tau_spline'].evaluate( tau ) / a
    # xeprime = param['xe_of_tau_spline'].derivative( tau )
    # cs2     = param['cs2a_of_tau_spline'].evaluate( tau ) / a
    # xe      = param['xe_of_tau_spline'].evaluate( tau )

    cs2     = param['cs2a_of_tau_spline'].evaluate( param['tau_of_a_spline'].evaluate( a ) ) / a
    xe      = param['xe_of_tau_spline'].evaluate( param['tau_of_a_spline'].evaluate( a ) )
    
    # ... Photon mass density over baryon mass density
    photbar = param['grhog'] / (param['grhom'] * param['Omegab'] * a)
    pb43 = 4.0 / 3.0 * photbar

    # massive neutrinos
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a)))
    # pnu = jnp.exp(param['logpnu_of_loga_spline'].evaluate( jnp.log(a) ) )

    # ... quintessence
    cs2_Q              = param['cs2_DE'] 
    w_Q                = param['w_DE_0'] + param['w_DE_a'] * (1.0 - a) 
    rho_Q              = a**(-3*(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(a-1)*param['w_DE_a'])
    rho_plus_p_theta_Q = (1+w_Q) * rho_Q * param['grhom'] * param['OmegaDE'] * thetaq * a**2
    
    # ... homogeneous background
    grho = (
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * rho_Q * a**2
        + param['grhom'] * param['Omegak']
    )

    # gpres = (
    #     (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    # ) / a**2 + w_Q * param['grhom'] * param['OmegaDE'] * rho_Q * a**2

    # ... compute expansion rate

    aprimeoa = jnp.sqrt(grho / 3.0)                # Friedmann I
    # aprimeprimeoa = 0.5 * (aprimeoa**2 - gpres)    # Friedmann II

    # quintessence EOS time derivatives
    w_Q_prime = -param['w_DE_a'] * aprimeoa * a
    ca2_Q     = w_Q - w_Q_prime / 3 / ((1+w_Q)+1e-6) / aprimeoa

    # ... Thomson opacity coefficient
    akthom = 2.3038921003709498e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2

    # ... Thomson opacity
    opac    = xe * akthom / a**2
    #tauc    = 1. / opac
    #taucprime = tauc * (2*aprimeoa - xeprime/xe)
    #F       = tauc / (1+pb43) #CLASS perturbations.c:10072
    #Fprime  = taucprime/(1+pb43) + tauc*pb43*aprimeoa/(1+pb43)**2 #CLASS perturbations.c:10074

    
    # ... background scale factor evolution
    f = f.at[0].set( aprimeoa * a )
    
    # ... evaluate metric perturbations
    drhonu, dpnu, fnu, shearnu = nu_perturb( a, param['amnu'], y[iq0:iq1], y[iq1:iq2], y[iq2:iq3], nqmax=nqmax )

    dgrho = (
        param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + (param['grhog'] * deltag + param['grhor'] * (param['Neff'] * deltar + param['Nmnu'] * drhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2
    )
    dgpres = (
        (param['grhog'] * deltag + param['grhor'] * param['Neff'] * deltar) / a**2 / 3.0 
        + param['grhor'] * param['Nmnu'] * dpnu / a**2 
        + (cs2_Q * param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2 + (cs2_Q-ca2_Q)*(3*aprimeoa * rho_plus_p_theta_Q / kmode**2))
    )
    dgtheta = (
        param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a
        + 4.0 / 3.0 * (param['grhog'] * thetag + param['Neff'] * param['grhor'] * thetar) / a**2
        + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2
        + rho_plus_p_theta_Q
    )
    dgshear = (
        4.0 / 3.0 * (param['grhog'] * shearg + param['Neff'] * param['grhor'] * shearr) / a**2
        + param['Nmnu'] * param['grhor'] * shearnu / a**2
    )

    dahprimedtau = -(dgrho + 3.0 * dgpres) * a
    
    f = f.at[1].set( dahprimedtau )

    # ... force energy conservation
    hprime = (2.0 * kmode**2 * eta + dgrho) / aprimeoa

    etaprime = 0.5 * dgtheta / kmode**2
    alpha  = (hprime + 6.*etaprime)/2./kmode**2
    f = f.at[2].set( etaprime )
    
    # alphaprime = -3*dgshear/(2*kmode**2) + eta - 2*aprimeoa*alpha
    # alphaprime -=  9/2 * a**2/kmode**2 * 4/3 * 16/45/opac * (thetag+kmode**2*alpha) * param['grhog']

    # ... cdm equations of motion, MB95 eq. (42)
    deltacprime = -thetac - 0.5 * hprime
    f = f.at[3].set( deltacprime )
    thetacprime = -aprimeoa * thetac  # thetac = 0 in synchronous gauge!
    f = f.at[4].set( thetacprime )

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

    #... photon temperature l>=3, BLT11 eq. (2.4d)
    ell  = jnp.arange(3, lmaxg )
    f = f.at[idxg+ell].set( kmode  / (2 * ell + 1) * (ell * y[idxg+ell-1] - (ell + 1) * y[idxg+ell+1]) - opac * y[idxg+ell] )
    # photon temperature hierarchy truncation, BLT11 eq. (2.5)
    f = f.at[idxg+lmaxg].set( kmode * y[idxg+lmaxg-1] - (lmaxg + 1) / tau * y[idxg+lmaxg] - opac * y[idxg+lmaxg] )

    #... polarization equations, BLT11 eq. (2.4e)
    ell  = jnp.arange(0, lmaxgp) # l=0...lmaxgp-1
    f = f.at[idxgp+ell].set( kmode  / (2 * ell + 1) * (ell * y[idxgp+ell-1] - (ell + 1) * y[idxgp+ell+1]) - opac * y[idxgp+ell] )
    f = f.at[idxgp+0].add( opac * polter / 2 )  # photon polarization l=0
    f = f.at[idxgp+2].add( opac * polter / 10 ) # photon polarization l=2
    
    # photon polarization hierarchy truncation
    f = f.at[idxgp+lmaxgp].set( kmode * y[idxgp+lmaxgp-1] - (lmaxgp + 1) / tau * y[idxgp+lmaxgp] - opac * y[idxgp+lmaxgp] )
    
    
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
    # q = jnp.arange(1, nqmax + 1) - 0.5  # so dq == 1 # if not using CAMB approx
    q, _ = get_neutrino_momentum_bins( nqmax )
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

    ell = jnp.arange(3, lmaxnu)
    vv = jnp.tile(v, lmaxnu - 3)
    denl = jnp.repeat( 2*ell+1, nqmax )

    f = f.at[iq0 + 3 * nqmax : iq0 + lmaxnu * nqmax].set(
        kmode * vv / denl * (
            jnp.repeat( ell, nqmax ) * y[iq0 + 2*nqmax : iq0 + (lmaxnu-1)*nqmax] 
            - jnp.repeat( ell+1, nqmax ) * y[iq0 + 4*nqmax : iq0 + (lmaxnu+1)*nqmax]
        ) 
    )

    # Truncate moment expansion.
    f = f.at[-nqmax-2 :-2].set(
        kmode * v * y[-2 * nqmax-2 : -nqmax-2] - (lmaxnu + 1) / tau * y[-nqmax-2 :-2]
    )

    # ---- Quintessence equations of motion -----------------------------------------------------------
    # ... Ballesteros & Lesgourgues (2010, BL10), arXiv:1004.5509
    f = f.at[-2].set( # BL10, eq. (3.5)
        -(1+w_Q) *(thetaq + 0.5 * hprime) - 3*(cs2_Q - w_Q) * aprimeoa * deltaq - 9*(1+w_Q)*(cs2_Q-ca2_Q)*aprimeoa**2/kmode**2 * thetaq
    )
    f = f.at[-1].set( # BL10, eq. (3.6)
        -(1-3*cs2_Q)*aprimeoa*thetaq + cs2_Q/(1+w_Q) * kmode**2 * deltaq
    )

    return f.flatten()


def convert_to_output_variables(*, y, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax ):
    """Convert the synchronous gauge perturbations to the output fields.

    Parameters
    ----------
    y : array_like
        input vector of perturbations
    param : dict
        dictionary of parameters and interpolated background functions
    kmode : float
        wavenumber of mode [1/Mpc]
    lmaxg : int
        maximum photon Boltmann hierarchy moment
    lmaxgp : int
        maximum photon polarization hierarchy moment
    lmaxr : int
        maximum massless neutrino hierarchy moment
    lmaxnu : int
        maximum massive neutrino hierarchy moment
    nqmax : int
        number of momentum bins for massive neutrinos

    Returns
    -------
    yout : array_like
        output vector of perturbations:
            eta, etaprime, hprime, alpha,       # 0-3
            deltam,  thetam / (aH),             # 4-5
            deltabc, thetabc / (aH),            # 6-7
            deltac,  thetac / (aH),             # 8-9
            deltab,  thetab / (aH),             # 10-11
            deltag,  thetag / (aH),             # 12-13
            deltar,  thetar / (aH),             # 14-15
            deltanu, thetanu / (aH),            # 16-17
            deltaq,  thetaq / (aH),             # 18-19
    where aH = \mathcal{H} = a' / a, which is the conformal Hubble rate.
    """

    Omegac = param['Omegam'] - param['Omegab']

    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax

    a = y[0]
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

    # ... massless neutrinos
    deltar = y[ 9 + lmaxg + lmaxgp]
    thetar = y[10 + lmaxg + lmaxgp]

    #... massive neutrinos
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a)))
    pnu = jnp.exp(param['logpnu_of_loga_spline'].evaluate( jnp.log(a) ) )
    rho_plus_p = rhonu + pnu

    drhonu, _, fnu, _ = nu_perturb( a, param['amnu'], y[iq0:iq1], y[iq1:iq2], y[iq2:iq3], nqmax=nqmax )
    deltanu = drhonu / rhonu
    thetanu = kmode * fnu / rho_plus_p

    # ... quintessence field
    deltaq    = y[-2]
    thetaq    = y[-1]
    w_Q       = param['w_DE_0'] + param['w_DE_a'] * (1.0 - a)
    rho_Q     = a**(-3*(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(a-1)*param['w_DE_a'])
    rho_plus_p_theta_Q = (1+w_Q) * rho_Q * param['grhom'] * param['OmegaDE'] * thetaq * a**2


    # ... background
    grho = (
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * rho_Q * a**2
        + param['grhom'] * param['Omegak']
    )

    gpres = (
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 + w_Q * param['grhom'] * param['OmegaDE'] * rho_Q * a**2
    
    # ... compute expansion rate
    aprimeoa = jnp.sqrt(grho / 3.0)                # Friedmann I
    
    # ... metric perturbations
    dgrho = (
        param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + (param['grhog'] * deltag + param['grhor'] * (param['Neff'] * deltar + param['Nmnu'] * drhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * deltaq * rho_Q * a**2
    )
    dgtheta = (
        param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a
        + 4.0 / 3.0 * (param['grhog'] * thetag + param['Neff'] * param['grhor'] * thetar) / a**2
        + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2
        + rho_plus_p_theta_Q
    )
    
    hprime = (2.0 * kmode**2 * eta + dgrho) / aprimeoa
    etaprime = 0.5 * dgtheta / kmode**2
    alpha  = (hprime + 6.*etaprime)/2./kmode**2


    # total matter perturbations
    deltam = (
        ( param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab) / a
        + (param['grhor'] * param['Nmnu'] * drhonu) / a**2) / (param['grhom'] * param['Omegam'] / a
        + (param['grhor'] * param['Nmnu'] * rhonu)/ a**2 )
    )
    thetam = (
        (param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a + param['Nmnu'] * param['grhor'] * kmode * fnu / a**2) 
        / (3.0 * (param['grhom'] * param['Omegam'] / a + param['grhor'] * param['Nmnu'] * rhonu / a**2 ))
    )

    deltabc = (param['grhom'] * (Omegac * deltac + param['Omegab'] * deltab)/ a) \
        / (param['grhom'] * param['Omegam'] / a)
    thetabc = (param['grhom'] * (Omegac * thetac + param['Omegab'] * thetab) / a) \
        / (3.0 * (param['grhom'] * param['Omegam'] / a) / a**2)
    
    #... gauge trafo from comoving (MB95 eq. 27b)
    thetam   += alpha * kmode**2
    thetabc  += alpha * kmode**2

    ##################################################################################################################

    # store fields of interest
    yout = jnp.array([
        eta, etaprime, hprime, alpha,       # 0-3
        deltam,  thetam  / aprimeoa,        # 4-5
        deltabc, thetabc / aprimeoa,        # 6-7
        deltac,  thetac  / aprimeoa,        # 8-9
        deltab,  thetab  / aprimeoa,        # 10-11
        deltag,  thetag  / aprimeoa,        # 12-13
        deltar,  thetar  / aprimeoa,        # 14-15
        deltanu, thetanu / aprimeoa,        # 16-17
        deltaq,  thetaq  / aprimeoa,        # 18-19
    ])
            
    return yout


def adiabatic_ics_one_mode( *, tau: float, param, kmode, nvar, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax ):
    """Initial conditions for adiabatic perturbations"""
    Omegac = param['Omegam'] - param['Omegab']

    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax

    y = jnp.zeros((nvar))
    a = param['a_of_tau_spline'].evaluate(tau)

    # .. isentropic ("adiabatic") initial conditions
    rhom  = param['grhom'] * param['Omegam'] / a**3
    rhor  = (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu']*jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a))))) / a**4
    rhonu = param['grhor'] * (param['Neff'] + param['Nmnu']*jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a)))) / a**4

    fracb  = param['Omegab'] / param['Omegam']
    fracg  = param['grhog'] / rhor
    fracnu = rhonu / rhor

    om    = a * rhom / jnp.sqrt(rhor)
    
    curvature_ini = -1.0
    s2_squared = 1.0

    #... photons
    deltag = -(kmode*tau)**2 / 3 * (1 - om * tau / 5) * curvature_ini * s2_squared
    thetag = -(kmode*tau)**3/tau /36 * (1-3*(1+5*fracb-fracnu)/20/(1-fracnu)*om*tau) * curvature_ini * s2_squared

    #... baryons
    deltab = 0.75 * deltag
    thetab = thetag

    #... CDM
    deltac = 0.75 * deltag
    thetac = 0.0

    #... massless neutrinos
    deltar = deltag
    thetar = -(kmode*tau)**4/tau/36/(4*fracnu+15) * (4*fracnu+11+12 - 3*(8*fracnu*fracnu+50*fracnu+275)/20/(2*fracnu+15)*tau*om) * curvature_ini
    shearr = (kmode*tau)**2/(45+12*fracnu) * (3*s2_squared-1) * (1+(4*fracnu-5)/4/(2*fracnu+15)*tau*om) * curvature_ini

    #... massive neutrinos
    deltan = deltar
    thetan = thetar
    shearn = shearr

    # ... quintessence, Ballesteros & Lesgourgues (2010, BL20), arXiv:1004.5509
    cs2_Q  = param['cs2_DE']
    w_Q    = param['w_DE_0'] + param['w_DE_a'] * (1.0 - a)
    deltaq = (kmode*tau)**2 / 4 * (1+w_Q)*(4-3*cs2_Q)/(4-6*w_Q+3*cs2_Q) * curvature_ini * s2_squared # BL10 eq. 3.7
    thetaq = (kmode*tau)**4 / tau / 4 * cs2_Q/(4-6*w_Q+3*cs2_Q) * curvature_ini * s2_squared      # BL10 eq. 3.8

    # metric
    eta = curvature_ini * (1-(kmode*tau)**2/12/(15+4*fracnu)*(5+4*s2_squared*fracnu - (16*fracnu*fracnu+280*fracnu+325)/10/(2*fracnu+15)*tau*om))

    ahprime = 0.0 # will not be evolved, only constraint

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
    # q = jnp.arange(1, nqmax + 1) - 0.5  # so dq == 1 # if not using CAMB approx
    q, _ = get_neutrino_momentum_bins( nqmax )
    aq = a * param['amnu'] / q
    v = 1 / jnp.sqrt(1 + aq**2)
    # akv = jnp.outer(kmode, v)
    dlfdlq = -q / (1.0 + jnp.exp(-q))
    y = y.at[iq0:iq1].set( -0.25 * dlfdlq * deltan)
    y = y.at[iq1:iq2].set( -dlfdlq * thetan / v / kmode / 3.0)
    y = y.at[iq2:iq3].set( -0.5 * dlfdlq * shearn)
    # higher moments are zero at the initial time

    # ... quintessence, Ballesteros & Lesgourgues (2010, BL20), arXiv:1004.5509
    y = y.at[-2].set( deltaq )
    y = y.at[-1].set( thetaq )
    
    return y


def determine_starting_time( *, param, k ):
    # ADOPTED from CLASS:
    # largest wavelengths start being sampled when universe is sufficiently opaque. This is quantified in terms of the ratio of thermo to hubble time scales, 
    # \f$ \tau_c/\tau_H \f$. Start when start_largek_at_tau_c_over_tau_h equals this ratio. Decrease this value to start integrating the wavenumbers earlier 
    # in time.
    start_small_k_at_tau_c_over_tau_h =  0.0004 

    # ADOPTED from CLASS:
    #  largest wavelengths start being sampled when mode is sufficiently outside Hubble scale. This is quantified in terms of the ratio of hubble time scale 
    #  to wavenumber time scale, \f$ \tau_h/\tau_k \f$ which is roughly equal to (k*tau). Start when this ratio equals start_large_k_at_tau_k_over_tau_h. 
    #  Decrease this value to start integrating the wavenumbers earlier in time. 
    start_large_k_at_tau_h_over_tau_k = 0.07 #0.05

    tau0 = param['taumin']
    tau1 = param['tau_of_a_spline'].evaluate( 0.1 ) # don't start after a=0.1
    tau_k = 1.0/k

    def compute_aprimeoa( a, param ):
        # assume neutrinos fully relativistic and no DE
        grho = (
            param['grhom'] * param['Omegam'] / a
            + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'])) / a**2
        )
        return jnp.sqrt( grho / 3.0 )

    def get_tauc_tauH( tau, param ):
        akthom = 2.3048e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2
        xe = param['xe_of_tau_spline'].evaluate( tau )
        a = param['a_of_tau_spline'].evaluate(tau)
        opac = xe * akthom / a**2

        # grho, _ = compute_rho_p( a, param )

        aprimeoa = compute_aprimeoa( a, param ) #jnp.sqrt( grho / 3.0 )
        return 1.0/opac, 1.0/aprimeoa
    
    
    def get_tauH( tau, param ):
        a = param['a_of_tau_spline'].evaluate(tau)

        # grho, _ = compute_rho_p( a, param )

        aprimeoa = compute_aprimeoa( a, param ) #jnp.sqrt( grho / 3.0 )
        return 1.0/aprimeoa

    # condition for small k: tau_c(a) / tau_H(a) < start_small_k_at_tau_c_over_tau_h
    def cond_small_k( logtau, param ):
        tau_c, tau_H = get_tauc_tauH( jnp.exp(logtau), param[0] )
        # adotoa/opac > start_small_k_at_tau_c_over_tau_h
        start_small_k_at_tau_c_over_tau_h = param[1]
        return tau_c/tau_H/start_small_k_at_tau_c_over_tau_h - 1.0

    # condition for large k: tau_H(a) / tau_k < start_large_k_at_tau_k_over_tau_h
    def cond_large_k( logtau, param ):
        tau_H = get_tauH( jnp.exp(logtau), param[0] )
        tau_k = param[2]
        start_large_k_at_tau_h_over_tau_k = param[1]
        return tau_H/tau_k/start_large_k_at_tau_h_over_tau_k - 1.0

    logtau_large_k = root_find_bisect(func=cond_large_k, xleft=jnp.log(tau0), xright=jnp.log(tau1), numit=7, param=(param,start_large_k_at_tau_h_over_tau_k,tau_k) )
    logtau_small_k = root_find_bisect(func=cond_small_k, xleft=jnp.log(tau0), xright=jnp.log(tau1), numit=7, param=(param,start_small_k_at_tau_c_over_tau_h) )

    return jnp.exp(jnp.minimum(logtau_small_k, logtau_large_k))


class VectorField(eqx.Module):
    model: eqx.Module

    def __call__(self, t, y, args):
        return self.model(t, y, args)
    

def rms_norm_filtered(x: PyTree, filter_indices: jnp.array, weights: jnp.array) -> Scalar:
    x, _ = fu.ravel_pytree(x)
    if x.size == 0:
        return 0
    return _rms_norm(x[filter_indices] * weights)


@jax.custom_jvp
def _rms_norm(x):
    x_sq = jnp.real(x * jnp.conj(x))
    return jnp.sqrt(jnp.mean(x_sq))


@_rms_norm.defjvp
def _rms_norm_jvp(x, tx):
    (x,) = x
    (tx,) = tx
    out = _rms_norm(x)
    # Get zero gradient, rather than NaN gradient, in these cases
    pred = (out == 0) | jnp.isinf(out)
    numerator = jnp.where(pred, 0, x)
    denominator = jnp.where(pred, 1, out * x.size)
    t_out = jnp.dot(numerator / denominator, tx)
    return out, t_out
    

@partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax','max_steps'))
def evolve_one_mode( *, tau_max, tau_out, param, kmode, 
                        lmaxg : int, lmaxgp : int, lmaxr : int, lmaxnu : int, \
                        nqmax : int, rtol: float, atol: float,
                        pcoeff : float, icoeff : float, dcoeff : float, factormax : float, factormin : float, max_steps : int  ):

    modelX_ = VectorField( 
        lambda tau, y , params : model_synchronous( tau=tau, y=y, param=params[0], kmode=params[1],  
                                                   lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax) )
    modelX = drx.ODETerm( modelX_ )
    
    # ... determine the number of active variables (i.e. the number of equations), absent any optimizations
    nvar   = 7 + (lmaxg + 1) + (lmaxgp + 1) + (lmaxr + 1) + nqmax * (lmaxnu + 1) + 2

    # ... determine starting time
    tau_start = determine_starting_time( param=param, k=kmode )
    tau_start = 0.99 * jnp.minimum( jnp.min(tau_out), tau_start )

    # ... set adiabatic ICs
    y0 = adiabatic_ics_one_mode( tau=tau_start, param=param, kmode=kmode, nvar=nvar, 
                       lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax )

    # create solver wrapper, we use the Kvaerno5 solver, which is a 5th order implicit solver
    def DEsolve_implicit( *, model, t0, t1, y0, saveat ):
        return drx.diffeqsolve(
            terms=model,
            solver=drx.Kvaerno5(),
            t0=t0,
            t1=t1,
            dt0=jnp.minimum(t0/4, 0.5*(t1-t0)),
            y0=y0,
            saveat=saveat,  
            stepsize_controller = drx.PIDController(rtol=rtol, atol=atol, norm=lambda t:rms_norm_filtered(t,jnp.array([0,2,3,5,6,7]), jnp.array([1,kmode**2,1,1,1/kmode**2,1])), 
                                                    pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff, factormax=factormax, factormin=factormin),
            # default controller has icoeff=1, pcoeff=0, dcoeff=0
            max_steps=max_steps,
            args=(param, kmode, ),
            # adjoint=drx.RecursiveCheckpointAdjoint(), # for backward differentiation
            adjoint=drx.DirectAdjoint(),  #for forward differentiation
            # adjoint=drx.BacksolveAdjoint(), # for backward differentiation
        )

    # solve before neutrinos become fluid
    saveat = drx.SaveAt(ts=tau_out)
    sol = DEsolve_implicit( model=modelX, t0=tau_start, t1=tau_max, y0=y0, saveat=saveat )

    # convert outputs
    yout = jax.vmap( lambda y : convert_to_output_variables( y=y, param=param, kmode=kmode, 
                                                   lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax) )( sol.ys )
    
    return yout



def evolve_perturbations( *, param, aexp_out, kmin : float, kmax : float, num_k : int, \
                         lmaxg : int = 11, lmaxgp : int = 11, lmaxr : int = 11, lmaxnu : int = 8, \
                         nqmax : int = 3, rtol: float = 1e-4, atol: float = 1e-4,
                         pcoeff : float = 0.25, icoeff : float = 0.80, dcoeff : float = 0.0, \
                         factormax : float = 20.0, factormin : float = 0.3, max_steps : int = 2048 ):
    """evolve cosmological perturbations in the synchronous gauge

    Parameters
    ----------
    param : dict
        dictionary of parameters and interpolated functions
    aexp_out : array
        array of scale factors at which to output
    kmin : float
        minimum wavenumber [in units 1/Mpc]
    kmax : float
        maximum wavenumber [in units 1/Mpc]
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
        array of shape (num_k) containing the wavenumbers [in units 1/Mpc]
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
                                    lmaxnu=lmaxnu, nqmax=nqmax, rtol=rtol, atol=atol,
                                    pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff, 
                                    factormax=factormax, factormin=factormin, max_steps=max_steps ),
                                    in_axes=0
    )(kmodes)
    
    return y1, kmodes


@partial(jax.jit, static_argnames=('N'))
def get_xi_from_P( *, k : jnp.array, Pk : jnp.array, N : int, ell : int = 0 ):
    """ get the correlation function from the power spectrum  using FFTlog, cf.
        J. D. Talman (1978). JCP, 29:35-48		
        A. J. S. Hamilton (2000).  MNRAS, 312:257-284

    Args:
        k (array_like)   : the wavenumbers [units 1/Mpc]
        Pk (array_like)  : the power spectrum
        N (int)          : length of the input vector
        ell (int)        : the multipole to compute (0,2,4,...)

    Returns:
        xi (array_like)  : the correlation function
        r (array_like)   : the radii [units in Mpc]
    """
    N = len(k)
    kmin = k[0]
    kmax = k[N-1]

    L = jnp.log(kmax/kmin)

    # FFTlog algorithm:
    fPk = jnp.fft.rfft( Pk * k**1.5 )

    ki = jnp.pi * jnp.arange( N//2+1 ) / L
    zp = (1.5+ell)/2 + 1j* ki

    theta = jax.vmap( lambda z: jnp.imag( lngamma_complex_e( z ) ) )( zp )

    fPk = fPk * jnp.exp( 2j * (theta - jnp.log(jnp.pi) * ki) )

    r  = 2*jnp.pi/k
    xi = jnp.real( 1j**ell * jnp.fft.irfft( fPk ) / (2*jnp.pi*r)**1.5 )
    return xi[::-1], r[::-1] # reverse order since 1/k is decreasing for increasing k


def get_power( *, k : jax.Array, y : jax.Array, idx : int , param : dict) -> jax.Array:
    """ compute the power spectrum from the perturbations
    
    Args:
        k (array_like)   : the wavenumbers [in units 1/Mpc]
        y (array_like)   : the perturbations
        idx (int)        : index of the perturbation to compute the power spectrum for
        param (dict)     : dictionary of all parameters
        
    Returns:
        Pk (array_like)  : the power spectrum
    """
    return 2 * jnp.pi**2 * param['A_s'] *(k/param['k_p'])**(param['n_s'] - 1) * k**(-3) * y[...,idx]**2


def get_power_smoothed( *, k : jax.Array, y : jax.Array, dlogk : float, idx : int , param : dict) -> jax.Array:
    """ compute Savitzky-Golay smoothed version of the power spectrum
    
    Args:
        k (array_like)   : the wavenumbers [in units 1/Mpc]
        y (array_like)   : the perturbations
        dlogk (float)    : the log bin width (dlogk = 1.0)
        idx (int)        : index of the perturbation to compute the power spectrum for
        param (dict)     : dictionary of all parameters
        
    Returns:
        Pk (array_like)  : the power spectrum
    """
    window_length = round(dlogk/(jnp.log(k[1])-jnp.log(k[0])))
    window_length += (window_length+1)%2

    Pm  = get_power( y=y, k=k, idx=idx, param=param )

    Pms = jnp.exp(savgol_filter(y=jnp.log(Pm), window_length=window_length, polyorder=3))

    # replace boundary affected regions with original signal
    Pms = Pms.at[:window_length//2].set( Pm[:window_length//2] )
    Pms = Pms.at[-window_length//2:].set( Pm[-window_length//2:] )

    return Pms

def power_Kaiser( *, y : jax.Array, kmodes : jax.Array, bias : float, mu_sampling : bool = True, smooth_dlogk : float = None, nmu : int, param : dict) -> tuple[jax.Array]:
    """ compute the anisotropic power spectrum using the Kaiser formula
    
    Args:
        y (array_like)       : input solution from the EB solver
        kmodes (array_like)  : the list of wave numbers in units of [1/Mpc]
        bias (float)         : linear tracer bias
        mu_sampling (bool)   : if True, sample the mu bins, else sample the theta bins
        smooth_dlogk (float) : if not None, use Savitzky-Golay smoothing at this log scale
        sigma_z0 (float)     : redshift error sigma_z = sigma_z0 * (1+z)
        nmu (int)            : number of mu bins
        param (dict)         : dictionary of all data

    Returns:
        P(k,mu) (array_like) : anisotropic spectrum
        mu (array_like)      : mu bins
        theta (array_like)   : theta bins, mu = cos(theta)
    """
    
    if mu_sampling:
        mu = jnp.linspace(-1,1,nmu)
    else:
        theta = jnp.linspace(0,jnp.pi,nmu)
        mu = jnp.cos(theta)
    
    if smooth_dlogk is None:
        fac = 2 * jnp.pi**2 * param['A_s']
        deltam = jnp.sqrt(fac *(kmodes/param['k_p'])**(param['n_s'] - 1) * kmodes**(-3)) * y[:,4]
        thetam = jnp.sqrt(fac *(kmodes/param['k_p'])**(param['n_s'] - 1) * kmodes**(-3)) * y[:,5]
    else:
        Pdelta = get_power_smoothed( y=y, k=kmodes, dlogk=smooth_dlogk, idx=4, param=param )
        Ptheta = get_power_smoothed( y=y, k=kmodes, dlogk=smooth_dlogk, idx=5, param=param )
        deltam = jnp.sqrt(Pdelta)
        thetam = -jnp.sqrt(Ptheta)

    # thetam already contains 1/ mathcal{H} factor   -f delta = theta
    Pkmu = (bias*deltam[:,None] - mu[None,:]**2 * thetam[:,None])**2
    return Pkmu, mu



def power_multipoles( *, y : jnp.array, kmodes : jnp.array, b : float, param ) -> tuple[jax.Array]:
    """ compute the power spectrum multipoles (l=0,2,4)

    Args:
        y (array_like)       : input solution from the EB solver
        kmodes (array_like)  : the list of wave numbers [in units 1/Mpc]
        b (float)            : linear bias

    Returns:
        P0 (array_like)      : monopole
        P2 (array_like)      : quadrupole
        P4 (array_like)      : hexadecapole
    """
    fac = 2 * jnp.pi**2 * param['A_s']
    deltam = jnp.sqrt(fac *(kmodes/param['k_p'])**(param['n_s'] - 1) * kmodes**(-3)) * y[:,4]
    thetam = jnp.sqrt(fac *(kmodes/param['k_p'])**(param['n_s'] - 1) * kmodes**(-3)) * y[:,5]

    # powerspectrum multipoles
    P0 = b**2 * deltam**2 - 2*b/3 * deltam*thetam + 1/5*thetam**2
    P2 = -4*b/3 * deltam * thetam + 4/7 * thetam**2
    P4 = 8/35 * thetam**2

    return P0, P2, P4

