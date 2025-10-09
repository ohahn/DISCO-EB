import jax
import jax.numpy as jnp
import diffrax as drx
from jax_cosmo.scipy.integrate import romb

from .thermodynamics_recfast import evaluate_thermo as evaluate_thermo_recfast
from .thermodynamics_mb95 import compute_thermo as compute_thermo_mb95

from .spline_interpolation import spline_interpolation
from .util import generalized_gauss_laguerre_weights, integrate_trapz


def get_neutrino_momentum_bins(  nqmax : int ) -> tuple[jax.Array, jax.Array]:
    """Get the momentum bins and integral kernel weights for neutrinos

    Args:
        nqmax (int): Number of momentum bins.

    Returns:
        jax.Array: q, w
    """
    # fermi_dirac_const = 7 * np.pi**4 / 120
    fermi_dirac_const = 5.682196976983475

    # nqmax = 3,4,5 are from high accuracy formulas from CAMB, higher values resort to modified Gauss-Laguerre,
    # which is not pre-computed however
    if nqmax == 3:
        q = jnp.array([0.913201, 3.37517, 7.79184])
        dlfdlq = -q/(1+jnp.exp(-q))
        w = jnp.array([0.0687359, 3.31435, 2.29911]) / (-0.25*dlfdlq)
    elif nqmax == 4:
        q = jnp.array([0.7, 2.62814, 5.90428, 12.0])
        dlfdlq = -q/(1+jnp.exp(-q))
        w = jnp.array([0.0200251, 1.84539, 3.52736, 0.289427]) / (-0.25*dlfdlq)
    elif nqmax == 5:
        q = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
        dlfdlq = -q/(1+jnp.exp(-q))
        w = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681]) / (-0.25*dlfdlq)
    else:
        alpha = 1
        q, w = generalized_gauss_laguerre_weights( nqmax, alpha )
        w *= q**3 / (1 + jnp.exp(-q)) * q**-alpha

    return q, w / fermi_dirac_const


def nu_background( a : float, amnu: float, nq : int = 8 ) -> tuple[float, float, float]:
    """ computes the neutrino density and pressure of one flavour of massive neutrinos
        in units of the mean density of one flavour of massless neutrinos

    Args:
        a (float): scale factor
        amnu (float): neutrino mass in units of neutrino temperature (m_nu*c**2/(k_B*T_nu0).
        nq (int, optional): number of integration points. Defaults to 8.

    Returns:
        tuple[float, float, float]: rho_nu/rho_nu0, p_nu/p_nu0, pp_nu/pp_nu0
    """

    # q is the comoving momentum in units of k_B*T_nu0/c.
    v    = lambda q: 1 / jnp.sqrt(1 + (a * amnu / q)**2)   # = (1/aq) / sqrt(1+1/aq**2)

    q, w = get_neutrino_momentum_bins( nq )
    rhonu = jnp.dot( w, 1. / v(q) )
    pnu = jnp.dot( w, v(q) / 3 )
    ppnu = jnp.dot( w, v(q)**3 / 3 )

    return rhonu, pnu, ppnu


def dtauda_(a, grhom, grhog, grhor, Omegam, OmegaDE, w_DE_0, w_DE_a, Omegak, Neff, Nmnu, logrhonu_spline):
    """Derivative of conformal time with respect to scale factor"""
    # rhonu = jax.vmap( lambda aa: nu_background(aa,amnu)[0] )( jnp.atleast_1d(a) )
    rhonu = jnp.exp(logrhonu_spline.evaluate(jnp.log(a)))
    rho_DE = a**(-3*(1+w_DE_0+w_DE_a)) * jnp.exp(3*(a-1)*w_DE_a)
    grho2 = grhom * Omegam * a \
        + (grhog + grhor*(Neff+Nmnu*rhonu)) \
        + grhom * OmegaDE * rho_DE * a**4 \
        + grhom * Omegak * a**2
    return jnp.sqrt(3.0 / grho2).reshape( jnp.asarray(a).shape )


def dadtau(a, param ):
    """Derivative of scale factor with respect to conformal time"""
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a)))
    # rhonu = jax.vmap( lambda aa: nu_background(aa,param['amnu'])[0] )( jnp.atleast_1d(a) )
    rho_DE = a**(-3*(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(a-1)*param['w_DE_a'])
    grho2 = param['grhom'] * param['Omegam'] * a \
        + (param['grhog'] + param['grhor']*(param['Neff']+param['Nmnu']*rhonu)) \
        + param['grhom'] * param['OmegaDE'] * rho_DE * a**4 \
        + param['grhom'] * param['Omegak'] * a**2
    return jnp.sqrt(grho2 / 3.0).reshape( jnp.asarray(a).shape )


def dtauda(a, param ):
    """Derivative of conformal time with respect to scale factor"""
    return 1/dadtau(a, param)


def get_aprimeoa( *, param, aexp ):
    """Compute the conformal Hubble function

    Args:
        param (dict): dictionary of cosmological parameters
        aexp (float, jax.Array): scale factor

    Returns:
        float: conformal H(a)
    """
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(aexp)))
    rho_Q = aexp**(-3*(1+param['w_DE_0']+param['w_DE_a'])) * jnp.exp(3*(aexp-1)*param['w_DE_a'])

    # ... background energy density
    grho = (
        param['grhom'] * param['Omegam'] / aexp
        + (param['grhog'] + param['grhor'] * (param['Neff'] + param['Nmnu'] * rhonu)) / aexp**2
        + param['grhom'] * param['OmegaDE'] * rho_Q * aexp**2
        + param['grhom'] * param['Omegak']
    )

    aprimeoa = jnp.sqrt(grho / 3.0)
    return aprimeoa


def compute_angular_diameter_distance( *, aexp, param ):
    """Compute the angular diameter distance

    Args:
        aexp (float): scale factor
        param (dict): dictionary of cosmological parameters

    Returns:
        float: angular diameter distance
    """
    aexpv = jnp.linspace( aexp, 1.0, 1000 )
    aH = get_aprimeoa( param=param, aexp=aexpv ) * aexpv
    Da = aexp * integrate_trapz( 1/aH, aexpv)
    return Da

def setup_background_evolution( *, amin, amax, param ):
    c2ok = 1.62581581e4 # K / eV
    num_neutrino = 512  # number of neutrino history arrays
    
    param['amin'] = amin
    param['amax'] = amax   

    # mean densities
    Omegak = 0.0 #1.0 - Omegam - OmegaL
    param['grhom'] = 3.33795017e-11 * param['H0']**2    # 8πG rho_c / c^2 * in 1/Mpc^2
    param['grhog'] = 1.49594245e-13 * param['Tcmb']**4  # photon density in 1/Mpc^2
    param['grhor'] = 3.39739477e-14 * param['Tcmb']**4  # neutrino density per flavour in 1/Mpc^2
    param['adotrad'] = jnp.sqrt((param['grhog']+param['grhor']*(param['Neff']+param['Nmnu'])) / 3.0)
    # param['adotrad'] = 2.8948e-7 * param['Tcmb']**2 # Hubble during radiation domination

    param['amnu'] = param['mnu'] * c2ok / param['Tcmb'] # conversion factor for Neutrinos masses (m_nu*c**2/(k_B*T_nu0)

    # Compute the scale factor linearly spaced in log(a)
    a = jnp.geomspace(amin*0.9, amax*1.1, num_neutrino)
    loga = jnp.log(a)
    param['a'] = a

    # Compute the neutrino density and pressure
    rhonu_, pnu_, ppnu_ = jax.vmap( lambda a_ : nu_background( a_, param['amnu'] ), in_axes=0 )( a )

    param['logrhonu_of_loga_spline']     = spline_interpolation( loga, jnp.log(rhonu_) )
    param['logpnu_of_loga_spline']       = spline_interpolation( loga, jnp.log(pnu_) )
    param['logppseudonu_of_loga_spline'] = spline_interpolation( loga, jnp.log(ppnu_) )

    # compute the energy density today due to massive neutrinos
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(0.0))
    Omegamnu = (param['grhor'] * rhonu) / param['grhom']
    param['Omegamnu'] = Omegamnu

    # ensure curvature is correct
    Omegar = (param['Neff']+param['Nmnu']*jnp.exp(param['logrhonu_of_loga_spline'].evaluate(0.0))) * param['grhor'] / param['grhom']
    Omegag = param['grhog'] / param['grhom']
    param['OmegaDE'] = 1.0 - param['Omegak'] - Omegar - Omegag - param['Omegam']

    # Compute the conformal time interval
    param['taumin'] = amin / param['adotrad']
    param['taumax'] = (param['taumin'] + 
        romb( lambda a_: dtauda_(a_,param['grhom'], param['grhog'], param['grhor'], 
                        param['Omegam'], param['OmegaDE'], param['w_DE_0'], param['w_DE_a'],
                        param['Omegak'], param['Neff'], param['Nmnu'], 
                        param['logrhonu_of_loga_spline']), amin, amax )
    )

    return param


def evolve_background( *, param, thermo_module = 'RECFAST', num_thermo: int = 512, rtol: float = 1e-5, atol: float = 1e-7, order: int = 5, class_thermo = None ):
    """Evolve the cosmological background and thermal history

    Parameters
    ----------
    param : dict
        Dictionary of cosmological parameters
    thermo_module : str, optional
        Thermal history module to use: 'RECFAST' (default, high accuracy),
        'MB95' (faster, approximate), or 'CLASS' (use external CLASS data)
    num_thermo : int, optional
        Number of sampling points for thermal history arrays. Default is 512.
        Higher values increase accuracy but slow computation. Validated accuracy:
        - 512: <0.03% error on P(k), 1.7x faster thermal history (recommended)
        - 1024: reference accuracy, slower
        - 256: <0.35% error on P(k), 3.3x faster thermal history
    rtol : float, optional
        Relative tolerance for ODE solvers. Default is 1e-5.
    atol : float, optional
        Absolute tolerance for ODE solvers. Default is 1e-7.
    order : int, optional
        Order of spline interpolation. Default is 5.
    class_thermo : dict, optional
        CLASS thermodynamics data (only used when thermo_module='CLASS')

    Returns
    -------
    param : dict
        Updated parameter dictionary with background evolution results and
        spline interpolations for thermal quantities
    """
    c2ok = 1.62581581e4 # K / eV

    amin = 1e-9
    amax = 1.01

    if thermo_module == 'CLASS':
        amin = jnp.min( class_thermo['scale factor a'] )
        amax = jnp.max( class_thermo['scale factor a'] )
    
    param = setup_background_evolution( amin=amin, amax=amax, param=param )

    if thermo_module == 'RECFAST':
        # Compute the thermal history
        param, tau, aexp, cs2, Tm, mu, xe, xeHI, xeHeI, xeHeII, xeprime_recfast = evaluate_thermo_recfast( param=param, num_thermo=num_thermo )

        param['aexp'] = aexp
        param['tau'] = tau
        param['xe'] = xe
        param['xeHI'] = xeHI
        param['xeHeI'] = xeHeI
        param['xeHeII'] = xeHeII
        param['cs2'] = cs2
        param['Tm'] = Tm

        param['tau_of_a_spline']      = spline_interpolation( aexp, tau )
        param['a_of_tau_spline']      = spline_interpolation( tau, aexp )
        param['xe_of_tau_spline']     = spline_interpolation( tau, xe )
        param['cs2a_of_tau_spline']   = spline_interpolation( tau, aexp*cs2 )
        param['tempba_of_tau_spline'] = spline_interpolation( tau, aexp*Tm )

        # Pre-composed splines for direct a-to-quantity lookups (performance optimization)
        param['xe_of_loga_spline']    = spline_interpolation( jnp.log(aexp), xe )
        param['cs2a_of_loga_spline']  = spline_interpolation( jnp.log(aexp), aexp*cs2 )

    elif thermo_module == 'MB95':

        # Compute the thermal history
        th, param = compute_thermo_mb95( param=param, nthermo=num_thermo )

        xe = th['xe']
        xeHI = th['xHII']
        xeHeI = th['xHeII']
        xeHeII = th['xHeIII']
        aexp = th['a']
        tau = th['tau']
        cs2 = th['cs2']
        Tm = th['tb']

        param['xe'] = xe
        param['xeHI'] = xeHI
        param['xeHeI'] = xeHeI
        param['xeHeII'] = xeHeII
        param['aexp'] = aexp
        param['tau'] = tau

        param['xe_of_tau_spline']     = spline_interpolation( tau, xe )
        param['cs2a_of_tau_spline']   = spline_interpolation( tau, aexp*cs2 )
        param['tempba_of_tau_spline'] = spline_interpolation( tau, aexp*Tm )
        param['tau_of_a_spline'] = spline_interpolation( aexp, tau )
        param['a_of_tau_spline'] = spline_interpolation( tau, aexp )

        # Pre-composed splines for direct a-to-quantity lookups (performance optimization)
        param['xe_of_loga_spline']    = spline_interpolation( jnp.log(aexp), xe )
        param['cs2a_of_loga_spline']  = spline_interpolation( jnp.log(aexp), aexp*cs2 )

    elif thermo_module == 'CLASS':
        # use input CLASS thermodynamics
        # interpolating splines for the thermal history        
        param['cs2_of_tau_spline']   = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['c_b^2'][::-1] )
        param['tempb_of_tau_spline'] = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['Tb [K]'][::-1] )
        param['xe_of_tau_spline']    = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['x_e'][::-1] )
        param['a_of_tau_spline']     = spline_interpolation( class_thermo['conf. time [Mpc]'][::-1], class_thermo['scale factor a'][::-1] )
        param['tau_of_a_spline']     = spline_interpolation( class_thermo['scale factor a'][::-1],   class_thermo['conf. time [Mpc]'][::-1] )

        param['aexp'] = class_thermo['scale factor a'][::-1]
        param['tau'] = class_thermo['conf. time [Mpc]'][::-1]

        # Pre-composed splines for direct a-to-quantity lookups (performance optimization)
        aexp_class = class_thermo['scale factor a'][::-1]
        param['xe_of_loga_spline']   = spline_interpolation( jnp.log(aexp_class), class_thermo['x_e'][::-1] )
        param['cs2a_of_loga_spline'] = spline_interpolation( jnp.log(aexp_class), aexp_class * class_thermo['c_b^2'][::-1] )
    


    # compute optical depth and visibility functions
    akthom = 2.3038921003709498e-9 * (1.0 - param['YHe']) * param['Omegab'] * param['H0']**2
    # grho_v = jax.vmap( lambda a: grho( param, a ) )( a )(aexp)
    # aprimeoa = jnp.sqrt( grho_v / 3.0 )
    aprimeoa = get_aprimeoa( param=param, aexp=aexp )

    tau_pre_recomb = param['tau_of_a_spline'].evaluate( 1e-4 )
    xe_full   = 1 + param['YHe'] / (1 - param['YHe'])
    xe        = jnp.where( tau <= tau_pre_recomb, xe_full, param['xe_of_tau_spline'].evaluate( tau ) )
    xeprime   = jnp.where( tau <= tau_pre_recomb, 0.0, param['xe_of_tau_spline'].derivative( tau ) )

    # xe = param['xe_of_tau_spline'].evaluate( tau )
    # xeprime = param['xe_of_tau_spline'].derivative( tau )
    # xepprime  = param['xe_of_tau_spline'].derivative2( tau )
    opac      = xe * akthom / aexp**2

    opacspline = spline_interpolation( tau, opac, integrate_from_start=False)
    opacprime  = opacspline.derivative( tau )
    opacpprime = opacspline.derivative2( tau )
    # opacprime = xeprime * akthom / aexp**2 - 2 * xe * akthom / aexp**2 * aprimeoa

    optical_depth = opacspline.integral( tau )
    optical_depth_today = opacspline.integral( param['tau_of_a_spline'].evaluate( 1.0 ) )
    optical_depth -= optical_depth_today

    # optical_depth = jnp.array([optical_depth[0], *optical_depth])
    expmmu    = jnp.exp(-optical_depth)
    vis       = opac * expmmu
    dvis      = (opacprime + opac**2) * expmmu
    ddvis     = (opacpprime + 3 * opac * opacprime + opac**3) * expmmu

    param['optical_depth'] = optical_depth
    param['opac'] = opac
    param['gvis'] = vis
    param['gvisprime'] = dvis
    param['gvispprime'] = ddvis

    if thermo_module == 'RECFAST':
        param['xeprime_recf'] = xeprime_recfast

    param['xeprime'] = xeprime
    
    return param 
        

