import numpy as np
from scipy.integrate import solve_ivp

import jax
import jax.numpy as jnp
import jax_cosmo.scipy.interpolate as jaxinterp
from functools import partial

@jax.jit
def ninu1( a : float, amnu: float, nq : int = 1000, qmax : float = 30.) -> tuple[float, float]:
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
    q    = dq * jnp.arange(0,nq+1)
    aq   = a * amnu / (q+1e-8) # prevent division by zero
    v    = 1 / jnp.sqrt(1 + aq**2)
    qdn  = dq * q**3 / (jnp.exp(q) + 1)
    dum1 = qdn / v
    dum2 = qdn * v
    
    rho_spline = jaxinterp.InterpolatedUnivariateSpline(q, dum1)
    rhonu = rho_spline.integral(0, qmax)
    p_spline = jaxinterp.InterpolatedUnivariateSpline(q, dum2)
    pnu = p_spline.integral(0, qmax)

    # Apply asymptotic corrrection for q>qmax and normalize by relativistic
    # energy density.
    rhonu = (rhonu / dq + dum1[-1] / dq) / const
    pnu = (pnu / dq + dum2[-1] / dq) / const / 3
    
    return rhonu, pnu

@jax.jit
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
    nqmax0 = 15
    qmax0  = nqmax0 - 0.5
    # const = 7 * np.pi**4 / 120
    const = 5.682196976983475

    g1 = jnp.zeros((nqmax0+1))
    g2 = jnp.zeros((nqmax0+1))
    g3 = jnp.zeros((nqmax0+1))
    g4 = jnp.zeros((nqmax0+1))
    q  = (jnp.arange(0,nqmax0+1) - 0.5)  # so dq == 1
    q.at[0].set(0.0)
    aq = a * amnu / q
    v = 1 / jnp.sqrt(1 + aq**2)
    qdn = q**3 / (jnp.exp(q) + 1)
    g1 = qdn * psi0 / v
    g2 = qdn * psi0 * v
    g3 = qdn * psi1 
    g4 = qdn * psi1 * v

    g1_sp = jaxinterp.InterpolatedUnivariateSpline(q, g1)
    g01 = g1_sp.integral(0, qmax0)
    g2_sp = jaxinterp.InterpolatedUnivariateSpline(q, g2)
    g02 = g2_sp.integral(0, qmax0)
    g3_sp = jaxinterp.InterpolatedUnivariateSpline(q, g3)
    g03 = g3_sp.integral(0, qmax0)
    g4_sp = jaxinterp.InterpolatedUnivariateSpline(q, g4)
    g04 = g4_sp.integral(0, qmax0)

    # Apply asymptotic corrrection for q>qmax0
    drhonu = (g01 + g1[-1] * 2 / qmax) / const
    dpnu = (g02 + g2[-1] * 2 / qmax) / const / 3
    fnu = (g03 + g3[-1] * 2 / qmax) / const
    shearnu = (g04 + g4[-1] * 2 / qmax) / const * 2 / 3

    return drhonu, dpnu, fnu, shearnu

@partial(jax.jit, static_argnames=("cosmo_params",))
def dtauda(a, tau, cosmo_params):
    rhonu = cosmo_params.rhonu_sp(a)
    """Derivative of conformal time with respect to scale factor"""
    grho2 = cosmo_params.grhom * cosmo_params.Omegam * a 
    + (cosmo_params.grhog + cosmo_params.grhor*(cosmo_params.Neff+cosmo_params.Nmnu*rhonu)) 
    + cosmo_params.grhom * cosmo_params.OmegaL * a**4 
    + cosmo_params.grhom * cosmo_params.Omegak * a**2
    return jnp.sqrt(3.0 / grho2)

@partial(jax.jit, static_argnames=("cosmo_params",))
def d2tauda2(a, tau, cosmo_params):
    """Second derivative of conformal time with respect to scale factor"""
    jac = jnp.zeros((1, 1))
    rhonu = cosmo_params.rhonu_sp(a)
    grho2 = cosmo_params.grhom * cosmo_params.Omegam * a 
    + (cosmo_params.grhog + cosmo_params.grhor*(cosmo_params.Neff+cosmo_params.Nmnu*rhonu)) 
    + cosmo_params.grhom * cosmo_params.OmegaL * a**4 
    + cosmo_params.grhom * cosmo_params.Omegak * a**2
    jac.at[0, 0].set( -0.5 * (cosmo_params.grhom * (4 * a**3 * cosmo_params.OmegaL + cosmo_params.Omegam 
                                                    + 2 * a * cosmo_params.Omegak) 
                                                    + cosmo_params.grhor * cosmo_params.Nmnu * a * cosmo_params.dr1(a)) \
                                                        * jnp.sqrt(3.0) / grho2**1.5 )
    return jac


class cosmo:

    def __init__(self, *, Omegam: float, Omegab: float, OmegaL: float, H0: float, Tcmb: float, YHe: float, Neff: float, Nmnu: int = 0, mnu: float = 0.0):
        self.Omegam = Omegam
        self.Omegab = Omegab
        self.OmegaL = OmegaL
        self.Nmnu = Nmnu
        self.mnu = mnu
        self.H0 = H0
        self.Tcmb = Tcmb
        self.YHe = YHe
        self.Neff = Neff
        self.grhom = 3.3379e-11 * H0**2
        self.grhog = 1.4952e-13 * Tcmb**4
        self.grhor = 3.3957e-14 * Tcmb**4
        self.adotrad = 2.8948e-7 * Tcmb**2
        self.Omegac = Omegam - Omegab
        self.Omegak = 1.0 - Omegam - OmegaL
        # self.OmegaL = 1.0 - Omegam - self.Omegar

        # conversion factor for Neutrinos masses (m_nu*c**2/(k_B*T_nu0)
        c2ok = 1.62581581e4 # K / eV
        self.mfac = c2ok / Tcmb
        self.amnu = self.mnu * self.mfac

        # Compute the scale factor
        amin = 1e-9
        amax = 1.1
        self.a = np.geomspace(amin, amax, 1000)

        # Compute the neutrino density and pressure
        self.rhonu = np.zeros_like(self.a)
        self.pnu = np.zeros_like(self.a)
        for i in range(len(self.a)):
            self.rhonu[i], self.pnu[i] = ninu1(self.a[i], self.amnu )

        self.rhonu_sp = jaxinterp.InterpolatedUnivariateSpline(self.a, self.rhonu)
        self.pnu_sp = jaxinterp.InterpolatedUnivariateSpline(self.a, self.pnu)
        
        self.r1 = jaxinterp.InterpolatedUnivariateSpline(self.a, jnp.log(self.rhonu))
        self.p1 = jaxinterp.InterpolatedUnivariateSpline(self.a, jnp.log(self.pnu))
        
        self.dr1 = lambda a : self.r1.derivative(a)
        self.ddr1 = lambda a : self.r1.derivative(a,n=2)
        self.dp1 = lambda a : self.p1.derivative(a)

        # Compute cosmic time
        tauini = 0.0
        sol = solve_ivp(lambda a,tau: dtauda(a,tau,self), \
                        (amin, amax), [tauini], \
                        jac= lambda a,tau: d2tauda2(a,tau,self), \
                        method="BDF", t_eval=self.a)
        # self.a = np.array(sol["t"])
        self.tau = np.array(sol["y"][0])

    def get_tau(self, a: float):
        return np.interp(a, self.a, self.tau)

    def get_a(self, tau: float):
        return np.interp(tau, self.tau, self.a)
