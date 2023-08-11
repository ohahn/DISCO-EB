import jax
import jax.numpy as jnp
from pylinger_background import nu_perturb
from functools import partial
import diffrax as drx
import equinox as eqx

import jax.flatten_util as fu

from diffrax.custom_types import Array, PyTree, Scalar

# @partial(jax.jit, inline=True)
def compute_rho_p( a, param ):
    rhonu = param['rhonu_of_a_spline'].evaluate(a)
    pnu = param['pnu_of_a_spline'].evaluate( a ) 
    
    rhoDE = a**(-3*(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(a-1)*param['w_DE_a'])
    wDE = param['w_DE_0'] + param['w_DE_a'] * (1-a)

    grho = (
        param['grhom'] * param['Omegam'] / a
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / a**2
        + param['grhom'] * param['OmegaDE'] * rhoDE * a**2
        + param['grhom'] * param['Omegak']
    )

    gpres = (
        (param['grhog'] + param['grhor'] * param['Neff']) / 3.0 + param['grhor'] * param['Nmnu'] * pnu
    ) / a**2 + wDE * param['grhom'] * param['OmegaDE'] * rhoDE * a**2

    return grho, gpres

def compute_time_scales( *, k, a, param ):
    
    grho,_ = compute_rho_p( a, param )
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

# @partial(jax.jit, inline=True)
def compute_fields_RSA( *, k, aprimeoa, hprime, eta, deltab, thetab, cs2_b, tau_c, tau_c_prime ):
    """  Compute relativistic species in radiation streaming approximation (RSA) of Blas, Lesgroupe, Tram, 2011 (BLT11)

    Args:
        k (float): _description_
        aprimeoa (float): _description_
        hprime (float): _description_
        eta (float): _description_
        deltab (float): _description_
        thetab (float): _description_
        cs2_b (float): _description_
        tau_c (float): _description_
        tau_c_prime (float): _description_

    Returns:
        deltag, thetag, sigmag, deltar, thetar, sigmar : massless ur PT fields in RSA
    """
    # ... photons 
    # BLT11 eq. (4.12a)
    deltag = 4 / k**2 * ( aprimeoa *hprime - k**2 * eta ) + 4 / (k**2 * tau_c) * (thetab + hprime/2)
    # BLT11 eq. (4.12b)
    thetag = - hprime / 2 + 3/(k**2 * tau_c) * ( - tau_c_prime/tau_c * (thetab + hprime/2) + (-aprimeoa * thetab + cs2_b * k**2 * deltab + k**2 * eta) )
    # BLT11 eq. (4.12c)
    shearg = 0.0
    
    # ... massless neutrinos
    # BLT11 eq. (4.12d)
    deltar = 4 / k**2 * ( aprimeoa *hprime - k**2 * eta )
    # BLT11 eq. (4.12e)
    thetar = - hprime / 2
    # BLT11 eq. (4.12f)
    shearr = 0.0

    return deltag, thetag, shearg, deltar, thetar, shearr 


@partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax', 'do_neutrino_cfa', 'do_relativistic_sa'))
def model_synchronous(*, tau, yin, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax, 
                      do_neutrino_cfa : bool, do_relativistic_sa: bool):     
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
    y = jnp.copy(yin)
    f = jnp.zeros_like(y)
    
    # some indexing convenience:
    iq0 = 10 + lmaxg + lmaxgp + lmaxr

    if do_relativistic_sa:
        iq0 = 7 

    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    iq4 = iq3 + nqmax


    Omegab  = param['Omegab']
    Omegac  = param['Omegam'] - Omegab
    OmegaDE = param['OmegaDE']
    amnu    = param['amnu']
    grhom   = param['grhom']
    grhog   = param['grhog']
    grhor   = param['grhor']


    # --- evaluate background thermodynamics
    tempb   = param['tempb_of_tau_spline'].evaluate( tau )
    cs2     = param['cs2_of_tau_spline'].evaluate( tau )
    xe      = param['xe_of_tau_spline'].evaluate( tau )
    xeprime = param['xe_of_tau_spline'].derivative( tau )

    # ... Photon mass density over baryon mass density
    a       = y[0]
    photbar = grhog / (grhom * Omegab * a)
    pb43    = 4.0 / 3.0 * photbar


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
    a       = y[0]
    ahprime = y[1]
    eta     = y[2]

    # ... cdm
    deltac = y[3]
    thetac = y[4]

    # ... baryons
    deltab = y[5]
    thetab = y[6]

    # --- ultrarelativistic species
    if not do_relativistic_sa:
        # ... photons
        deltag = y[7]
        thetag = y[8]
        shearg = y[9] / 2.0

        # ... massless neutrinos
        deltar = y[ 9 + lmaxg + lmaxgp]
        thetar = y[10 + lmaxg + lmaxgp]
        shearr = y[11 + lmaxg + lmaxgp] / 2.0
    else:
        deltag = 0.0  # we set them to zero hereo and obtain them in the RSA lateronce hprime is known
        deltar = 0.0  # 
        

    # ... quintessence field
    deltaq = y[-2]
    thetaq = y[-1]

    # ... massive neutrino thermodynamics
    if not do_neutrino_cfa:
        drhonu, dpnu, fnu, dshearnu = nu_perturb( a, amnu, y[iq0:iq1], y[iq1:iq2], y[iq2:iq3] )
        dthetanu = kmode * fnu

    else:

        deltanu = y[iq0+0]
        thetanu = y[iq0+1]
        shearnu = y[iq0+2]

        rhonu = param['rhonu_of_a_spline'].evaluate( a )
        pnu = param['pnu_of_a_spline'].evaluate( a ) 
        rho_plus_p_nu = rhonu+pnu
        ppseudonu = param['ppseudonu_of_a_spline'].evaluate( a ) # pseudo pressure from CLASS IV, LT11
        
        w_nu = pnu / rhonu
        ca2_nu = w_nu/3.0/(1.0+w_nu)*(5.0-ppseudonu/pnu)  # eq. (3.3) in LT11
        ceff2_nu = ca2_nu
        cvis2_nu = 3.*w_nu*ca2_nu # CLASS's fluid approximation eq. (3.15c) in LT11
        cg2_nu   = w_nu-w_nu/3.0/(1.0+w_nu)*(3.0*w_nu-2.0+ppseudonu/pnu) # CLASS perturbation.c:7078

        drhonu   = rhonu * deltanu
        dpnu     = cg2_nu * deltanu
        dthetanu = rho_plus_p_nu * thetanu
        dshearnu = rho_plus_p_nu * shearnu
    

    # ... compute expansion rate
    grho, gpres = compute_rho_p( a, param )

    aprimeoa = jnp.sqrt(grho / 3.0)                # Friedmann I
    aprimeprimeoa = 0.5 * (aprimeoa**2 - gpres)    # Friedmann II

    # ... Thomson opacity and its rate of change
    akthom  = 2.3048e-9 * (1.0 - param['YHe']) * Omegab * param['H0']**2 # TODO: store in param
    opac    = xe * akthom / a**2
    tauc    = 1. / opac
    taucprime = tauc * (2*aprimeoa - xeprime/xe)
    F       = tauc / (1+pb43) #CLASS perturbations.c:10072
    Fprime  = taucprime/(1+pb43) + tauc*pb43*aprimeoa/(1+pb43)**2 #CLASS perturbations.c:10074

    # ... quintessence thermodynamics
    cs2_Q     = param['cs2_DE']
    w_Q       = param['w_DE_0'] + param['w_DE_a'] * (1.0 - a)
    w_Q_prime = - param['w_DE_a'] * aprimeoa * a
    ca2_Q     = w_Q - w_Q_prime / 3 / ((1+w_Q)+1e-6) / aprimeoa
    rhoDE     = a**(-3*(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(a-1)*param['w_DE_a'])
    rho_plus_p_theta_Q = (1+w_Q) * rhoDE * grhom * OmegaDE * thetaq * a**2

    # ... background scale factor evolution
    f = f.at[0].set( aprimeoa * a )
    
    # ... evaluate metric perturbations
    dgrho = (
        grhom * (Omegac * deltac + Omegab * deltab) / a
        + (grhog * deltag + grhor * (param['Neff'] * deltar + param['Nmnu'] * drhonu)) / a**2
        + grhom * OmegaDE * deltaq * rhoDE * a**2
    )

    if do_relativistic_sa:
        hprime = (2.0 * kmode**2 * eta + dgrho) / aprimeoa

        deltag, thetag, shearg, deltar, thetar, shearr = compute_fields_RSA( k=kmode, aprimeoa=aprimeoa, hprime=hprime, 
                                                                             eta=eta, deltab=deltab, thetab=thetab, cs2_b=cs2,
                                                                             tau_c=tauc, tau_c_prime=taucprime )

    dgpres = (
        (grhog * deltag + grhor * param['Neff'] * deltar) / a**2 / 3.0 
        + grhor * param['Nmnu'] * dpnu / a**2 
        + (cs2_Q * grhom * OmegaDE * deltaq * rhoDE * a**2 + (cs2_Q-ca2_Q)*(3*aprimeoa * rho_plus_p_theta_Q / kmode**2)) 
    )
    dgtheta = (
        grhom * (Omegac * thetac + Omegab * thetab) / a
        + 4.0 / 3.0 * (grhog * thetag + param['Neff'] * grhor * thetar) / a**2
        + param['Nmnu'] * grhor * dthetanu / a**2
        + rho_plus_p_theta_Q
    )
    dgshear = (
        4.0 / 3.0 * (grhog * shearg + param['Neff'] * grhor * shearr) / a**2
        + param['Nmnu'] * grhor * dshearnu / a**2
    )

    dahprimedtau = -(dgrho + 3.0 * dgpres) * a
    
    f = f.at[1].set( dahprimedtau )

    # ... hprime is not evolved but the energy constraint
    hprime = (2.0 * kmode**2 * eta + dgrho) / aprimeoa
    etaprime = 0.5 * dgtheta / kmode**2
    alpha  = (hprime + 6.*etaprime)/2./kmode**2
    f = f.at[2].set( etaprime )
    
    alphaprime = -3*dgshear/(2*kmode**2) + eta - 2*aprimeoa*alpha
    alphaprime -=  9/2 * a**2/kmode**2 * 4/3*16/45/opac * (thetag+kmode**2*alpha) * grhog

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
    
    if not do_relativistic_sa:
        # --- check if we are in the tight coupling regime -----------------------------------------------
        tight_coupling_trigger_tau_c_over_tau_h=0.015       # value taken from CLASS
        tight_coupling_trigger_tau_c_over_tau_k=0.010       # value taken from CLASS
        
        tauh = 1./aprimeoa
        tauk = 1./kmode
        
        f = jax.lax.cond(
            jnp.logical_or( tauc/tauk > tight_coupling_trigger_tau_c_over_tau_k,
                jnp.logical_and( tauc/tauh > tight_coupling_trigger_tau_c_over_tau_h,
                                tauc/tauk > 0.1*tight_coupling_trigger_tau_c_over_tau_k)),
            calc_baryon_photon_uncoupled, calc_baryon_photon_tca_CLASS, f )
        
        # f = calc_baryon_photon_tca( f )
        # f = calc_baryon_photon_uncoupled( f )
        
        
        # --- Massless neutrino equations of motion -------------------------------------------------------
        #.. MB95, eqs. (49)
        idxr = 9 + lmaxg + lmaxgp
        deltarprime = 4.0 / 3.0 * (-thetar - 0.5 * hprime)
        f = f.at[idxr+0].set( deltarprime )
        thetarprime = kmode**2 * (0.25 * deltar - shearr)
        f = f.at[idxr+1].set( thetarprime )
        shearrprime = 8./15. * (thetar + kmode**2 * alpha) - 0.6 * kmode * y[idxr+3]
        f = f.at[idxr+2].set( shearrprime )
        ell = jnp.arange(3, lmaxr)
        f = f.at[idxr+ell].set( kmode / (2 * ell + 1) * (ell * y[idxr+ell-1] - (ell + 1) * y[idxr+ell+1]) )
        
        # ... truncate moment expansion, MB95, eq. (51) to find
        f = f.at[idxr+lmaxr].set( kmode * y[idxr+lmaxr-1] - (lmaxr + 1) / tau * y[idxr+lmaxr] )

    else:
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

    # --- Massive neutrino equations of motion --------------------------------------------------------
    if not do_neutrino_cfa:
        # --- Massive neutrino equations of motion, full hierarchy 
        q = jnp.arange(1, nqmax + 1) - 0.5  # so dq == 1
        aq = a * amnu / q
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
        f = f.at[-nqmax-2 :-2].set(
            kmode * v * y[-2 * nqmax-2: -nqmax-2] - (lmaxnu + 1) / tau * y[-nqmax-2:-2]
        )
        
    else:
        # --- Massive neutrino equations of motion, fluid approximation 
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


    # ---- Quintessence equations of motion -----------------------------------------------------------
    # ... Ballesteros & Lesgourgues (2010, BL10), arXiv:1004.5509
    f = f.at[-2].set( # BL10, eq. (3.5)
        -(1+w_Q) *(thetaq + 0.5 * hprime) - 3*(cs2_Q - w_Q) * aprimeoa * deltaq 
        - 9*(1+w_Q)*(cs2_Q-ca2_Q)*aprimeoa**2/kmode**2 * thetaq
    )
    f = f.at[-1].set( # BL10, eq. (3.6)
        -(1-3*cs2_Q)*aprimeoa*thetaq + cs2_Q/(1+w_Q) * kmode**2 * deltaq
    )

    return f


def convert_to_neutrino_fluid(*, tau, yin, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax ):
    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq1 = iq0 + nqmax
    iq2 = iq1 + nqmax
    iq3 = iq2 + nqmax
    
    a = yin[0]
    
    nvarnu = 10 + lmaxg + lmaxgp + lmaxr + 3 + 2
    y = jnp.zeros((nvarnu))
    
    drhonu, _, fnu, shearnu = nu_perturb( a, param['amnu'], yin[iq0:iq1], yin[iq1:iq2], yin[iq2:iq3] )
    rhonu = param['rhonu_of_a_spline'].evaluate( a )
    pnu = param['pnu_of_a_spline'].evaluate( a ) 
    rho_plus_p = rhonu + pnu
    
    # copy over all the other variables
    y = y.at[:iq0].set( yin[:iq0] )

    # convert massive neutrinos
    y = y.at[iq0+0].set( drhonu / rhonu)
    y = y.at[iq0+1].set( kmode * fnu / rho_plus_p)
    y = y.at[iq0+2].set( shearnu / rho_plus_p )

    # copy quintessence
    y = y.at[-2:].set( yin[-2:] )
    
    return y

def convert_to_rsa(*, tau, yin, param, kmode, lmaxg, lmaxgp, lmaxr, lmaxnu, nqmax ):
    iq0 = 10 + lmaxg + lmaxgp + lmaxr
    iq0_new = 7
    a = yin[0]
    
    nvarnu = 7 + 3 + 2
    y = jnp.zeros((nvarnu))
    
    # copy over all the other variables
    y = y.at[:iq0_new].set( yin[:iq0_new] )

    # convert massive neutrinos
    y = y.at[iq0_new:iq0_new+3].set( yin[iq0:iq0+3] )

    # copy quintessence
    y = y.at[-2:].set( yin[-2:] )
    
    return y


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

    # .. isentropic ("adiabatic") initial conditions
    rhom  = param['grhom'] * param['Omegam'] / a**3
    rhor  = (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu']*param['rhonu_of_a_spline'].evaluate(a))) / a**4
    rhonu = param['grhor'] * (param['Neff'] + param['Nmnu']*param['rhonu_of_a_spline'].evaluate(a)) / a**4

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
    q = jnp.arange(1, nqmax + 1) - 0.5  # so dq == 1
    aq = a * param['amnu'] / q
    v = 1 / jnp.sqrt(1 + aq**2)
    # akv = jnp.outer(kmode, v)
    dlfdlq = -q / (1.0 + jnp.exp(-q))
    y = y.at[iq0:iq1].set( -0.25 * dlfdlq * deltan )
    y = y.at[iq1:iq2].set( -dlfdlq * thetan / v / kmode / 3.0 )
    y = y.at[iq2:iq3].set( -0.5 * dlfdlq * shearn )
    # higher moments are zero at the initial time

    # ... quintessence, Ballesteros & Lesgourgues (2010, BL20), arXiv:1004.5509
    y = y.at[-2].set( deltaq )
    y = y.at[-1].set( thetaq )
    
    return y

def root_find_bisect( *, func, xleft, xright, numit, param ):
    """
    Simple bisection routine for root finding.
    
    Parameters
    ----------
    func : function
        Function to be evaluated.
    xleft : float
        Left boundary of the interval.
    xright : float
        Right boundary of the interval.
    numit : int
        Number of iterations.

    Returns
    -------
    x0 : float
        Approximation to the root, given by the midpoint of the final interval.

    """
    for i in range(numit):
        xmid = 0.5 * (xleft + xright)
        xleft, xright = jax.lax.cond(func(xmid, param) * func(xleft, param) > 0, lambda x : (xmid, xright), lambda x : (xleft, xmid), None )
    return 0.5 * (xleft + xright)


def determine_starting_time( *, param, k ):
    # ADOPTED from CLASS:
    # largest wavelengths start being sampled when universe is sufficiently opaque. This is quantified in terms of the ratio of thermo to hubble time scales, 
    # \f$ \tau_c/\tau_H \f$. Start when start_largek_at_tau_c_over_tau_h equals this ratio. Decrease this value to start integrating the wavenumbers earlier 
    # in time.
    start_small_k_at_tau_c_over_tau_h =  0.0015 #0.0004 #0.0015 #/ 200. 

    # ADOPTED from CLASS:
    #  largest wavelengths start being sampled when mode is sufficiently outside Hubble scale. This is quantified in terms of the ratio of hubble time scale 
    #  to wavenumber time scale, \f$ \tau_h/\tau_k \f$ which is roughly equal to (k*tau). Start when this ratio equals start_large_k_at_tau_k_over_tau_h. 
    #  Decrease this value to start integrating the wavenumbers earlier in time. 
    start_large_k_at_tau_h_over_tau_k = 0.07 #0.05

    tau0 = param['taumin']
    tau1 = param['tau_of_a_spline'].evaluate( 0.1 ) # don't start after a=0.1
    tau_k = 1.0/k

    def get_tauc_tauH( tau, param ):
        akthom = 2.3048e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2
        xe = param['xe_of_tau_spline'].evaluate( tau )
        a = param['a_of_tau_spline'].evaluate( tau )
        opac = xe * akthom / a**2

        grho, _ = compute_rho_p( a, param )

        aprimeoa = jnp.sqrt( grho / 3.0 )
        return 1.0/opac, 1.0/aprimeoa
    
    
    def get_tauH( tau, param ):
        a = param['a_of_tau_spline'].evaluate( tau )

        grho, _ = compute_rho_p( a, param )

        aprimeoa = jnp.sqrt( grho / 3.0 )
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

def determine_free_streaming_time( *, param, k, radiation_streaming_trigger_tau_c_over_tau ):

    tau0 = param['taumin']
    tau1 = param['tau_of_a_spline'].evaluate( 1.0 ) 
    # tau_k = 1.0/k

    def get_tauc_tauH( tau, param ):
        akthom = 2.3048e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2
        xe = param['xe_of_tau_spline'].evaluate( tau )
        a = param['a_of_tau_spline'].evaluate( tau )
        opac = xe * akthom / a**2

        grho, _ = compute_rho_p( a, param )

        aprimeoa = jnp.sqrt( grho / 3.0 )
        return 1.0/opac, 1.0/aprimeoa

    def cond_free_stream( logtau, param ):
        tau_c, _ = get_tauc_tauH( jnp.exp(logtau), param[0] )
        radiation_streaming_trigger_tau_c_over_tau = param[1]
        return tau_c / jnp.exp(logtau) / radiation_streaming_trigger_tau_c_over_tau - 1.0
    
    logtau_free_stream = root_find_bisect(func=cond_free_stream, xleft=jnp.log(tau0), xright=jnp.log(tau1), numit=7, param=(param,radiation_streaming_trigger_tau_c_over_tau))

    return jnp.exp(logtau_free_stream)



class VectorField(eqx.Module):
    model: eqx.Module

    def __call__(self, t, y, args):
        return self.model(t, y, args)
    

def rms_norm_filtered(x: PyTree, filter_indices: jnp.array) -> Scalar:
    x, _ = fu.ravel_pytree(x)
    if x.size == 0:
        return 0
    return _rms_norm(x[filter_indices])


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
    

@partial(jax.jit, static_argnames=('lmaxg', 'lmaxgp', 'lmaxr', 'lmaxnu', 'nqmax'))
def evolve_one_mode( *, tau_max, tau_out, param, kmode, 
                        lmaxg : int, lmaxgp : int, lmaxr : int, lmaxnu : int, \
                        nqmax : int, rtol: float, atol: float,
                        pcoeff : float, icoeff : float, dcoeff : float, factormax : float, factormin : float  ):
    
    nu_fluid_trigger_tau_over_tau_k            = 31.0   # value taken from default CLASS settings
    radiation_streaming_trigger_tau_c_over_tau = 5.0    # value taken from CLASS
    radiation_streaming_trigger_tau_over_tau_k = 45. #45.    # value taken from CLASS

    # ... wrapper for model1: synchronous gauge without any optimizations, effectively this is Ma & Bertschinger 1995 
    # ... plus CLASS tight coupling approximation (Blas, Lesgourgues & Tram 2011, CLASS II)
    model1_ = VectorField( 
        lambda tau, y , params : model_synchronous( tau=tau, yin=y, param=params[0], kmode=params[1],  
                                                   lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax, 
                                                   do_neutrino_cfa=False, do_relativistic_sa=False ) )
    model1 = drx.ODETerm( model1_ )
    
    # ... wrapper for model2: synchronous gauge with neutrino fluid approximation, following Lesgourgues & Tram 2011 (CLASS IV)
    model2_ = VectorField( 
        lambda tau, y , params :  model_synchronous( tau=tau, yin=y, param=params[0], kmode=params[1], 
                                                    lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax, 
                                                    do_neutrino_cfa=True, do_relativistic_sa=False ) )
    model2 = drx.ODETerm( model2_ )

    # ... wrapper for model3: synchronous gauge with neutrino fluid and free streaming approx, following Lesgourgues & Tram 2011 (CLASS IV)
    model3_ = VectorField( 
        lambda tau, y , params :  model_synchronous( tau=tau, yin=y, param=params[0], kmode=params[1], 
                                                    lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax, 
                                                    do_neutrino_cfa=True, do_relativistic_sa=True ) )
    model3 = drx.ODETerm( model3_ )
    
    # ... determine the number of active variables (i.e. the number of equations), absent any optimizations
    nvar   = 7 + (lmaxg + 1) + (lmaxgp + 1) + (lmaxr + 1) + nqmax * (lmaxnu + 1) + 2

    # ... determine starting time
    # tau_start = 0.01
    tau_start = determine_starting_time( param=param, k=kmode )
    # # tau_start = jnp.minimum( param['tau_of_a_spline'].evaluate(0.01), tau_start )
    tau_start = jnp.minimum( jnp.min(tau_out), tau_start )

    # ... set adiabatic ICs
    y0 = adiabatic_ics_one_mode( tau=tau_start, param=param, kmode=kmode, nvar=nvar, 
                       lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax )


    # ... switch between mfodel1 and model2 depending on tau
    tauk = 1./kmode
    tau_neutrino_cfa = jnp.minimum(tauk * nu_fluid_trigger_tau_over_tau_k, 0.999*tau_max) # don't go to tau_max so that all modes are converted to massive neutrino approx

    # determine time to apply free streaming approximation. if tau_free_stream < tau_neutrino_cfa, then don't use free streaming approximation
    tau_free_stream = determine_free_streaming_time( param=param, k=kmode,  radiation_streaming_trigger_tau_c_over_tau=radiation_streaming_trigger_tau_c_over_tau)
    tau_free_stream = jnp.minimum( jnp.maximum( tau_free_stream, radiation_streaming_trigger_tau_over_tau_k/kmode ), tau_max )
    # tau_free_stream = jax.lax.cond( tau_free_stream > tau_neutrino_cfa, lambda x: tau_free_stream, lambda x: x, tau_max)
    
    # create solver wrapper, we use the Kvaerno5 solver, which is a 5th order implicit solver
    def DEsolve( *, model, t0, t1, y0, saveat ):
        return drx.diffeqsolve(
            terms=model,
            solver=drx.Kvaerno5(),
            t0=t0,
            t1=t1,
            dt0=jnp.minimum(t0/4, 0.5*(t1-t0)),
            y0=y0,
            saveat=saveat,  
            # stepsize_controller = drx.PIDController(rtol=rtol, atol=atol, norm=lambda t:rms_norm_filtered(t,jnp.array([0,2,3,5,6]))), #pcoeff=0.0, icoeff=1.0, dcoeff=0.0, factormax=10., factormin=0.1),
            stepsize_controller = drx.PIDController(rtol=rtol, atol=atol, norm=lambda t:rms_norm_filtered(t,jnp.array([0,1,2,3,5,6,7])), 
                                                    pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff, factormax=factormax, factormin=factormin),
            # default controller has icoeff=1, pcoeff=0, dcoeff=0
            max_steps=4096*4,
            args=(param, kmode, ),
            # adjoint=drx.RecursiveCheckpointAdjoint(),
            adjoint=drx.DirectAdjoint(),
        )


    if False:

        # solve before neutrinos become fluid
        saveat1 = drx.SaveAt(ts= jnp.where(tau_out<tau_neutrino_cfa,tau_out,tau_neutrino_cfa) )
        saveat2 = drx.SaveAt(ts= jnp.where(tau_out>=tau_neutrino_cfa,tau_out,tau_neutrino_cfa) )

        sol1 = DEsolve( model=model1, t0=tau_start, t1=tau_neutrino_cfa, y0=y0, saveat=saveat1 )
        # convert neutrinos to fluid by integrating over the momentum bins
        y0_neutrino_cfa = convert_to_neutrino_fluid( tau=sol1.ts[-1], yin=sol1.ys[-1,:], param=param, kmode=kmode, 
                                                lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax )
        # solve after neutrinos become fluid
        sol2 = DEsolve( model=model2, t0=sol1.ts[-1], t1=tau_max, y0=y0_neutrino_cfa, saveat=saveat2 )
        
        y1_converted = jax.vmap( 
            lambda tau, yin : convert_to_neutrino_fluid(tau=tau, yin=yin, param=param, kmode=kmode, lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax ), 
            in_axes=0, out_axes=0 )( sol1.ts, yin=sol1.ys )
        
        return jnp.where( tau_out[:,None]<tau_neutrino_cfa, y1_converted, sol2.ys )

    elif False:
        # solve before neutrinos become fluid
        saveat = drx.SaveAt(ts=tau_out)
        sol = DEsolve( model=model1, t0=tau_start, t1=tau_max, y0=y0, saveat=saveat )
        return sol.ys


    else:

        # create list of saveat times for first and second part of evolution
        saveat1 = drx.SaveAt(ts= jnp.where(tau_out<tau_neutrino_cfa,tau_out,tau_neutrino_cfa) )
        saveat2 = drx.SaveAt(ts= jnp.select([tau_out<tau_neutrino_cfa,tau_out>=tau_free_stream],[tau_neutrino_cfa,tau_free_stream],tau_out))
        saveat3 = drx.SaveAt(ts= jnp.where(tau_out>=tau_free_stream,tau_out,tau_free_stream) )

        # solve before neutrinos become fluid
        sol1 = DEsolve( model=model1, t0=tau_start, t1=tau_neutrino_cfa, y0=y0, saveat=saveat1 )
        
        # convert neutrinos to fluid by integrating over the momentum bins
        y0_neutrino_cfa = convert_to_neutrino_fluid( tau=sol1.ts[-1], yin=sol1.ys[-1,:], param=param, kmode=kmode, 
                                                lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax )
        
        # solve after neutrinos become fluid
        sol2 = DEsolve( model=model2, t0=sol1.ts[-1], t1=tau_free_stream, y0=y0_neutrino_cfa, saveat=saveat2 )
        
        y1_converted = jax.vmap( 
            lambda tau, yin : convert_to_neutrino_fluid(tau=tau, yin=yin, param=param, kmode=kmode, lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax ), 
            in_axes=0, out_axes=0 )( sol1.ts, yin=sol1.ys )
        
        # convert neutrinos to fluid by integrating over the momentum bins
        y1_rsa = convert_to_rsa( tau=sol2.ts[-1], yin=sol2.ys[-1,:], param=param, kmode=kmode, 
                                                lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax )
        
        # solve using radiation streaming approximation
        sol3 = DEsolve( model=model3, t0=sol2.ts[-1], t1=tau_max, y0=y1_rsa, saveat=saveat3 )
        
        y1_converted = jax.vmap(
            lambda tau, yin : convert_to_rsa(tau=tau, yin=yin, param=param, kmode=kmode, lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax ),
            in_axes=0, out_axes=0 )( sol2.ts, yin=y1_converted )
        
        y2_converted = jax.vmap(
            lambda tau, yin : convert_to_rsa(tau=tau, yin=yin, param=param, kmode=kmode, lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax ),
            in_axes=0, out_axes=0 )( sol2.ts, yin=sol2.ys )
        
        return jnp.select( [tau_out[:,None]<tau_neutrino_cfa, tau_out[:,None]>tau_free_stream], [y1_converted, sol3.ys], y2_converted )


def evolve_perturbations( *, param, aexp_out, kmin : float, kmax : float, num_k : int, \
                         lmaxg : int = 11, lmaxgp : int = 11, lmaxr : int = 11, lmaxnu : int = 17, \
                         nqmax : int = 15, rtol: float = 1e-4, atol: float = 1e-4,
                         pcoeff : float = 0.25, icoeff : float = 0.60, dcoeff : float = 0.0, factormax : float = 20.0, factormin : float = 0.3 ):
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
                                    lmaxnu=lmaxnu, nqmax=nqmax, rtol=rtol, atol=atol,
                                    pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff, factormax=factormax, factormin=factormin ),
                                    in_axes=0
    )(kmodes)
    
    return y1, kmodes

