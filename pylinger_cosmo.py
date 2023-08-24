import jax
import jax.numpy as jnp
import jax_cosmo.scipy.interpolate as jaxinterp


@jax.jit
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


# @partial(jax.jit, static_argnames=("params",))
@jax.jit
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
    return 1/dadtau(a, param)