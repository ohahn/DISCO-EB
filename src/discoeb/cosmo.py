import jax
import jax.numpy as jnp
import jax_cosmo.scipy.interpolate as jaxinterp


# the trapz integration has been moved from jnp to jax.scipy
# in newer versions of jax, and might disappear altogether
if hasattr(jnp,'trapz'):
  integrate_trapz = jnp.trapz
else:
  integrate_trapz = jax.scipy.integrate.trapezoid


# @partial( jax.jit, static_argnames=('nqmax',) )
def get_neutrino_momentum_bins(  nqmax : int ) -> tuple[jax.Array, jax.Array]:
    """Get the momentum bins and integral kernel weights for neutrinos

    Args:
        nqmax (int): Number of momentum bins.

    Returns:
        jax.Array: q, w
    """
    # fermi_dirac_const = 7 * np.pi**4 / 120
    fermi_dirac_const = 5.682196976983475

    if nqmax == 3:
        q = jnp.array([0.913201, 3.37517, 7.79184])
        w = jnp.array([0.0687359, 3.31435, 2.29911])
    elif nqmax == 4:
        q = jnp.array([0.7, 2.62814, 5.90428, 12.0])
        w = jnp.array([0.0200251, 1.84539, 3.52736, 0.289427])
    elif nqmax == 5:
        q = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
        w = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681])
    else:
        dq = (12 + nqmax/5)/nqmax
        q = (jnp.arange(1, nqmax + 1) - 0.5) * dq
        dlfdlq = -q/(1+jnp.exp(-q))
        w = dq * q**3 / (jnp.exp(q) + 1) * (-0.25*dlfdlq)
    dlfdlq = -q/(1+jnp.exp(-q))  #TODO: recompute the coefficients without the dlfdlq factor from CAMB
    w /= (-0.25*dlfdlq)
    return q, w / fermi_dirac_const 

# @jax.jit
def nu_background( a : float, amnu: float, nq : int = 1000 ) -> tuple[float, float, float]:
    """ computes the neutrino density and pressure of one flavour of massive neutrinos
        in units of the mean density of one flavour of massless neutrinos

    Args:
        a (float): scale factor
        amnu (float): neutrino mass in units of neutrino temperature (m_nu*c**2/(k_B*T_nu0).
        nq (int, optional): number of integration points. Defaults to 1000.
        qmax (float, optional): maximum momentum. Defaults to 30.

    Returns:
        tuple[float, float, float]: rho_nu/rho_nu0, p_nu/p_nu0
    """
    qmax = (12 + nq/10)

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
    # rhonu = (rhonu / dq + dum1[-1] / dq) / const
    # pnu = (pnu / dq + dum2[-1] / dq) / const / 3
    # ppnu = (ppnu / dq + dum3[-1] / dq) / const / 3
    rhonu = rhonu / dq / const
    pnu = pnu / dq / const / 3
    ppnu = ppnu / dq / const / 3
    
    return rhonu, pnu, ppnu


# @partial(jax.jit, static_argnames=("params",))
# @jax.jit
def dtauda_(a, grhom, grhog, grhor, Omegam, OmegaDE, w_DE_0, w_DE_a, Omegak, Neff, Nmnu, amnu):
    """Derivative of conformal time with respect to scale factor"""
    # jax.debug.print( 'a={} amnu={}',a, amnu)
    rhonu = jax.vmap( lambda aa: nu_background(aa,amnu)[0] )( jnp.atleast_1d(a) )
    rho_DE = a**(-3*(1+w_DE_0+w_DE_a)) * jnp.exp(3*(a-1)*w_DE_a)
    grho2 = grhom * Omegam * a \
        + (grhog + grhor*(Neff+Nmnu*rhonu)) \
        + grhom * OmegaDE * rho_DE * a**4 \
        + grhom * Omegak * a**2
    return jnp.sqrt(3.0 / grho2).reshape( jnp.asarray(a).shape )

def dadtau(a, param ):
    """Derivative of conformal time with respect to scale factor"""
    # rhonu = jax.vmap( lambda aa: nu_background(aa,param['amnu'])[0] )( jnp.atleast_1d(a) )
    rhonu = jnp.exp(param['logrhonu_of_loga_spline'].evaluate(jnp.log(a)))

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
