import jax
import jax.numpy as jnp
from .util import gauss_laguerre_weights, generalized_gauss_laguerre_weights
from .spline_interpolation import spline_interpolation


# the trapz integration has been moved from jnp to jax.scipy
# in newer versions of jax, and might disappear altogether
if hasattr(jnp,'trapz'):
  integrate_trapz = jnp.trapz
else:
  integrate_trapz = jax.scipy.integrate.trapezoid


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


# @partial(jax.jit, static_argnames=("params",))
# @jax.jit
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
    """Derivative of conformal time with respect to scale factor"""
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
