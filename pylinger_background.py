import pylinger_thermodynamics as pthermo

import jax
import jax.numpy as jnp
import diffrax as drx
import jax_cosmo.scipy.interpolate as jaxinterp
from jax_cosmo.scipy.integrate import romb
from functools import partial


def nu_background( a : float, amnu: float, nq : int = 1000, qmax : float = 30.) -> tuple[float, float]:
    """ computes the neutrino density and pressure of one flavour of massive neutrinos
        in units of the mean density of one flavour of massless neutrinos

    Args:
        a (float): scale factor
        amnu (float): neutrino mass in units of neutrino temperature (m_nu*c**2/(k_B*T_nu0).
        nq (int, optional): number of integration points. Defaults to 1000.
        qmax (float, optional): maximum momentum. Defaults to 30..

    Returns:
        tuple[float, float]: rho_nu/rho_nu0, p_nu/p_nu0
    """

    # const = 7 * np.pi**4 / 120
    const = 5.682196976983475
    
    # q is the comoving momentum in units of k_B*T_nu0/c.
    # Integrate up to qmax and then use asymptotic expansion for remainder.
    dq   = qmax / nq
    q    = dq * jnp.arange(1,nq+1)
    aq   = a * amnu / q
    v    = 1 / jnp.sqrt(1 + aq**2)   # = (1/aq) / sqrt(1+1/aq**2)
    qdn  = dq * q**3 / (jnp.exp(q) + 1)
    dum1 = qdn / v
    dum2 = qdn * v
    dum3 = qdn * v**3
    
    rho_spline = jaxinterp.InterpolatedUnivariateSpline(q, dum1)
    rhonu = rho_spline.integral(0, qmax)[0]
    p_spline = jaxinterp.InterpolatedUnivariateSpline(q, dum2)
    pnu = p_spline.integral(0, qmax)[0]
    pp_spline = jaxinterp.InterpolatedUnivariateSpline(q, dum3)
    ppnu = pp_spline.integral(0, qmax)[0]

    # Apply asymptotic corrrection for q>qmax and normalize by relativistic
    # energy density.
    rhonu = (rhonu / dq + dum1[-1] / dq) / const
    pnu = (pnu / dq + dum2[-1] / dq) / const / 3
    ppnu = (ppnu / dq + dum3[-1] / dq) / const / 3
    
    return rhonu, pnu, ppnu


def nu_perturb( a : float, amnu: float, psi0, psi1, psi2, nq : int = 1000, qmax : float = 30.):
    """ Compute the perturbations of density, energy flux, pressure, and
        shear stress of one flavor of massive neutrinos, in units of the mean
        density of one flavor of massless neutrinos, by integrating over 
        momentum.

    Args:
        a (float): scale factor
        amnu (float): neutrino mass in units of neutrino temperature (m_nu*c**2/(k_B*T_nu0).
        psi0 (_type_): 
        psi1 (_type_): _description_
        psi2 (_type_): _description_
        nq (int, optional): _description_. Defaults to 1000.
        qmax (float, optional): _description_. Defaults to 30..

    Returns:
        _type_: drhonu, dpnu, fnu, shearnu
    """
    nqmax0 = len(psi0)
    qmax0  = nqmax0 - 0.5
    # const = 7 * np.pi**4 / 120
    const = 5.682196976983475

    g1 = jnp.zeros((nqmax0+1))
    g2 = jnp.zeros((nqmax0+1))
    g3 = jnp.zeros((nqmax0+1))
    g4 = jnp.zeros((nqmax0+1))
    q  = (jnp.arange(1,nqmax0+1) - 0.5)  # so dq == 1
    qq = jnp.arange(0,nqmax0+1)  # so dq == 1
    # q.at[0].set(0.0)

    aq = a * amnu / q
    v = 1 / jnp.sqrt(1 + aq**2)
    qdn = q**3 / (jnp.exp(q) + 1)
    g1 = g1.at[1:].set( qdn * psi0 / v )
    g2 = g2.at[1:].set( qdn * psi0 * v )
    g3 = g3.at[1:].set( qdn * psi1 )
    g4 = g4.at[1:].set( qdn * psi2 * v )

    g01 = jnp.trapz(g1, qq)
    g02 = jnp.trapz(g2, qq)
    g03 = jnp.trapz(g3, qq)
    g04 = jnp.trapz(g4, qq)

    # Apply asymptotic corrrection for q>qmax0
    drhonu = (g01 + g1[-1] * 2 / qmax) / const
    dpnu = (g02 + g2[-1] * 2 / qmax) / const / 3
    fnu = (g03 + g3[-1] * 2 / qmax) / const
    shearnu = (g04 + g4[-1] * 2 / qmax) / const * 2 / 3

    return drhonu, dpnu, fnu, shearnu


# @partial(jax.jit, static_argnames=("params",))
def dtauda_(a, grhom, grhog, grhor, Omegam, OmegaDE, w_DE_0, w_DE_a, Omegak, Neff, Nmnu, rhonu_spline):
    """Derivative of conformal time with respect to scale factor"""
    rho_DE = a**(-3*(1+w_DE_0+w_DE_a)) * jnp.exp(3*(a-1)*w_DE_a)
    grho2 = grhom * Omegam * a \
        + (grhog + grhor*(Neff+Nmnu*rhonu_spline.evaluate(a))) \
        + grhom * OmegaDE * rho_DE * a**4 \
        + grhom * Omegak * a**2
    return jnp.sqrt(3.0 / grho2)


@partial(jax.jit, static_argnames=('class_thermo',))
def evolve_background( *, param, class_thermo = None, rtol: float = 1e-5, atol: float = 1e-7, order: int = 5):
    c2ok = 1.62581581e4 # K / eV
    amin = 1e-9
    amax = 1.01
    num_thermo = 2048 # length of thermal history arrays

    # mean densities
    Omegak = 0.0 #1.0 - Omegam - OmegaL

    param['grhom'] = 3.3379e-11 * param['H0']**2    # critical density at z=0 in h^2/Mpc^3
    param['grhog'] = 1.4952e-13 * param['Tcmb']**4  # photon density in h^2/Mpc^3
    param['grhor'] = 3.3957e-14 * param['Tcmb']**4  # neutrino density per flavour in h^2/Mpc^3
    param['adotrad'] = 2.8948e-7 * param['Tcmb']**2 # Hubble during radiation domination

    param['amnu'] = param['mnu'] * c2ok / param['Tcmb'] # conversion factor for Neutrinos masses (m_nu*c**2/(k_B*T_nu0)

    if class_thermo is not None:
        amin = jnp.min( class_thermo['scale factor a'] )
        amax = jnp.max( class_thermo['scale factor a'] )

    
    # Compute the scale factor linearly spaced in log(a)
    a = jnp.geomspace(amin, amax, num_thermo)
    param['a'] = a

    # Compute the neutrino density and pressure
    rhonu_, pnu_, ppnu_ = jax.vmap( lambda a_ : nu_background( a_, param['amnu'] ), in_axes=0 )( param['a'] )

    rhonu_coeff = drx.backward_hermite_coefficients(ts=a, ys=rhonu_)
    pnu_coeff = drx.backward_hermite_coefficients(ts=a, ys=pnu_)
    ppnu_coeff = drx.backward_hermite_coefficients(ts=a, ys=ppnu_)
    rhonu_spline = drx.CubicInterpolation( ts=a, coeffs=rhonu_coeff )
    
    param['rhonu_of_a_spline']     = rhonu_spline
    param['pnu_of_a_spline']       = drx.CubicInterpolation( ts=a, coeffs=pnu_coeff )
    param['ppseudonu_of_a_spline'] = drx.CubicInterpolation( ts=a, coeffs=ppnu_coeff )
    param['taumin'] = amin / param['adotrad']
    param['taumax'] = param['taumin'] + romb( lambda a_: dtauda_(a_,param['grhom'], param['grhog'], param['grhor'], 
                                                                 param['Omegam'], param['OmegaDE'], param['w_DE_0'], param['w_DE_a'],
                                                                 param['Omegak'], param['Neff'], param['Nmnu'], 
                                                                 param['rhonu_of_a_spline']), amin, amax )


    if class_thermo is not None:
        # use input CLASS thermodynamics
        # interpolating splines for the thermal history
        param['cs2_coeff'] = drx.backward_hermite_coefficients(ts=class_thermo['conf. time [Mpc]'][::-1], ys=class_thermo['c_b^2'][::-1])
        param['tb_coeff']  = drx.backward_hermite_coefficients(ts=class_thermo['conf. time [Mpc]'][::-1], ys=class_thermo['Tb [K]'][::-1])
        param['xe_coeff']  = drx.backward_hermite_coefficients(ts=class_thermo['conf. time [Mpc]'][::-1], ys=class_thermo['x_e'][::-1])
        param['a_coeff']   = drx.backward_hermite_coefficients(ts=class_thermo['conf. time [Mpc]'][::-1], ys=class_thermo['scale factor a'][::-1])
        param['tau_coeff'] = drx.backward_hermite_coefficients(ts=class_thermo['scale factor a'][::-1],   ys=class_thermo['conf. time [Mpc]'][::-1])
        
        param['cs2_of_tau_spline']   = drx.CubicInterpolation( ts=class_thermo['conf. time [Mpc]'][::-1], coeffs=param['cs2_coeff'] )
        param['tempb_of_tau_spline'] = drx.CubicInterpolation( ts=class_thermo['conf. time [Mpc]'][::-1], coeffs=param['tb_coeff'] )
        param['xe_of_tau_spline']    = drx.CubicInterpolation( ts=class_thermo['conf. time [Mpc]'][::-1], coeffs=param['xe_coeff'] )
        param['a_of_tau_spline']     = drx.CubicInterpolation( ts=class_thermo['conf. time [Mpc]'][::-1], coeffs=param['a_coeff'] )
        param['tau_of_a_spline']     = drx.CubicInterpolation( ts=class_thermo['scale factor a'][::-1],   coeffs=param['tau_coeff'] )
    
    else:    
        # Compute the thermal history
        th, param = pthermo.compute( param=param, nthermo=2048 )
        
        param['th'] = th # for debugging, otherwise no need to store
        param['a_in_spline']         = th['a']
        param['tau_in_spline']       = th['tau']

        # interpolating splines for the thermal history
        param['cs2_coeff'] = drx.backward_hermite_coefficients(ts=th['tau'], ys=th['cs2'])
        param['tb_coeff']  = drx.backward_hermite_coefficients(ts=th['tau'], ys=th['tb'])
        param['xe_coeff']  = drx.backward_hermite_coefficients(ts=th['tau'], ys=th['xe'])
        param['a_coeff']   = drx.backward_hermite_coefficients(ts=th['tau'], ys=th['a'])
        param['tau_coeff'] = drx.backward_hermite_coefficients(ts=th['a'],   ys=th['tau'])
        
        param['cs2_of_tau_spline']   = drx.CubicInterpolation( ts=th['tau'], coeffs=param['cs2_coeff'] )
        param['tempb_of_tau_spline'] = drx.CubicInterpolation( ts=th['tau'], coeffs=param['tb_coeff'] )
        param['xe_of_tau_spline']    = drx.CubicInterpolation( ts=th['tau'], coeffs=param['xe_coeff'] )
        param['a_of_tau_spline']     = drx.CubicInterpolation( ts=th['tau'], coeffs=param['a_coeff'] )
        param['tau_of_a_spline']     = drx.CubicInterpolation( ts=th['a'],   coeffs=param['tau_coeff'] )
    
    
    return param 
        

