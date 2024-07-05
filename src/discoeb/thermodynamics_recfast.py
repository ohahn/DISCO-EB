import diffrax as drx
import equinox as eqx
import jax 
import jax.numpy as jnp
from functools import partial 
from typing import Tuple

from .cosmo import dadtau
from .util import softclip

from .ode_integrators_stiff import GRKT4
from diffrax import Kvaerno5


const_c2ok = 1.62581581e4 # K / eV
const_c_Mpc_s = 9.71561189e-15 # Mpc/s

const_G       = 6.67430e-11          # Gravitational Constant [N m^2/kg^2], PDG 2023
const_mH      = 1.67353284e-27      # H atom mass [kg], PDG 2023
const_me      = 9.1093837015e-31    # Electron mass [kg], PDG 2023
const_mHe_mH  = 3.97146570884       # Helium / Hydrogen mass ratio, https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
const_c       = 2.99792458e+08      # Speed of light [m/s]     
const_h       = 6.62607015e-34      # Planck's constant [Js], PDG 2023
const_kB      = 1.380649e-23        # Boltzman's constant [J/K], PDG 2023

const_aRad    = 7.565914E-16        # Radiation constant W/m^4 K^4
const_sigmaT  = 6.6524616E-29       # Thomson cross section [m^2], PDG 2023

const_Lam2s1sH  = 8.2245809/const_c_Mpc_s         # H 2s-1s two photon rate in [s^-1] 
const_Lam2s1sHe = 51.3 /const_c_Mpc_s             # HeI 2s-1s two photon rate in [s^-1]
const_LyalphaH  = 1.215670e-07      # H Lyman alpha wavelength in [m]              
const_LyalphaHe = 5.843344e-08      # HeI 2^1p - 1^1s wavelength in [m]                -> HeI 5.04259042e-8 from Drake (1993)
const_EionH2s   = 3.944934227013e4  # H 2s ionization energy in [K] 
const_EionHe2s  = 4.608856067179e4  # HeI 2s ionization energy in [K]  
const_EionHe12s = 6.314878282674e5  # HeII 1s ionization energy in [K] 
const_E2s1sH    = 1.183515881558e5  # H 2s energy from the ground state [K] 
const_E2s1sHe   = 2.392347612515e5  # HeI 2s energy from the ground state in [K]
const_E2p2sHe   = 6.988756904317e3  # HeI 2p - 2s energy difference in  [K]


const_Hfac = 1/(1.0e+06*3.0856775807e+13) # 1 km/s/Mpc in 1/s

MAX_EXP = 60

@jax.custom_jvp
def NHnow( a, YHe, H0, Omegab ):
    mu_H = 1/(1-YHe) 
    rho_c = 3 * (H0*const_Hfac)**2 / (8 * jnp.pi * const_G) 
    return rho_c * Omegab / (const_mH * mu_H) / a**3

@NHnow.defjvp
def NHnow_jvp( primals, tangents ):
    a, YHe, H0, Omegab = primals
    da, dYHe, dH0, dOmegab = tangents
    N = NHnow( a, YHe, H0, Omegab )
    dN = N * ( -3 * da/a + dOmegab/Omegab + 2*dH0/H0 - dYHe/(1-YHe)  )
    return N, dN

@jax.custom_jvp
def SahaBoltzmann_( gi, gc, E_ion, T ):
    rescale = 1.0e-16 #1e-9
    # c1 = (const_h/(2*jnp.pi*const_me)) * (const_h/const_kB) #/ 1e-13**(2/3)
    c1 = 2.578838606475204e-06 * (1e-13/rescale)**(2/3)
    betaE = jnp.minimum(jnp.maximum(E_ion/T,-MAX_EXP),MAX_EXP)
    return gi/(2*gc) * (c1/T)**1.5 * jnp.exp(betaE)

@SahaBoltzmann_.defjvp
def SahaBoltzmann_jvp( primals, tangents ):
    gi, gc, E_ion, T = primals
    dgi, dgc, dE_ion, dT = tangents
    S = SahaBoltzmann_( gi, gc, E_ion, T )
    dS = S * (1/gi * dgi - 1/gc * dgc + 1/T * dE_ion - (1.5 + E_ion/T)/T * dT)
    return S, dS

def SahaBoltzmann( *, gi, gc, E_ion, T ):
    return SahaBoltzmann_( gi, gc, E_ion, T )

@jax.custom_jvp
def fBoltzmann_( gj, gi, E, T ):
    betaE = jnp.minimum(jnp.maximum(E/T,-MAX_EXP),MAX_EXP)
    return gj/gi * jnp.exp(-betaE)

@fBoltzmann_.defjvp
def fBoltzmann_jvp( primals, tangents ):
    """
    Compute (forward-mode) Jacobian-vector product of fBoltzmann
    """
    gj, gi, E, T = primals
    dgj, dgi, dE, dT = tangents
    fB = fBoltzmann_( gj, gi, E, T )
    dfB = fB * ( dgj/gj - dgi/gi - dE / T + E / T**2 * dT )
    return fB, dfB

def fBoltzmann( *, gj, gi, E, T ):
    """ 
    Boltzmann factor for transition from level j to level i
    """
    return fBoltzmann_( gj, gi, E, T )

def Saha_HeII( a, param ):
    """
    Saha equation for HeII recombination
    """
    T = param['Tcmb'] / a
    betaE = const_EionHe12s / T
    fHe = param['YHe']/(const_mHe_mH*(1.0-param['YHe']))
    A = 1 + fHe
    B = 1 + 2*fHe
    R = (2*jnp.pi* const_me * const_kB / const_h**2 * T )**1.5 / NHnow( a, param['YHe'], param['H0'], param['Omegab'] ) * jnp.exp( - betaE )

    xe = jax.lax.cond( R>1e5, 
                      lambda x: fHe * (1 - B/R + (1 + 5*fHe + 6*fHe**2)/R**2), # asymptotic expansion to prevent truncation errors
                      lambda x:  -(R-A)/2 + jnp.sqrt( (R-A)**2/4 + R*B ) - A, None )
    
    return xe



def model_recfast( *, logtau : float, yin : jnp.array, param : dict, noiseless_dT : bool = False ) -> jnp.array:
  """
    Recombination model from RECFAST (Seager et al. 1999), minor changes from original implementation
  """
  rescale = 1.0e-16
  
  tau  = jnp.exp(logtau)
  a    = jnp.exp(yin[0])
  xHep = yin[1]
  xp   = yin[2]
  xe   = xHep + xp
  # TM   = jnp.clip(yin[3], 1e-8, 1.1*param['Tcmb']) / a    # y[3]=a*T since it is O(1) and does not evolve by orders of magnitude
  TM   = yin[3] / a
  TR   = param['Tcmb'] / a

  fHe = param['YHe'] / (const_mHe_mH*(1-param['YHe']))
  
  Hz = dadtau( a=a, param=param ) / a**2  # is in units of 1/Mpc

  nHtot = NHnow( a, param['YHe'], param['H0'], param['Omegab'] ) * rescale

  ε = 1e-4
  nH1  = jax.lax.cond( xp > 1-ε, lambda x: 0.0, lambda x: (1-xp) * nHtot, None )
  nHe1 = jax.lax.cond( xHep > (1-ε)*fHe, lambda x: 0.0, lambda x: (fHe - xHep) * nHtot, None )

  #  Calculate the Saha and Boltzmann equilibrium relations
  SHe = SahaBoltzmann( gi=1, gc=2, E_ion=const_EionHe2s, T=TM )
  fBHe = fBoltzmann( gj=1, gi=1, E=const_E2s1sHe, T=TM )
  
  SH = SahaBoltzmann( gi=2, gc=1, E_ion=const_EionH2s, T=TM)
  fBH = fBoltzmann( gj=1, gi=1, E=const_E2s1sH, T=TM)
  
  # For HeI, the energy levels of 2s and 2p are quite different.
  # Therefore the ratio b should be a boltzmann factor, and this
  # will also have to be incorporated into the derivatives wrt TM

  fBHe2p2s = fBoltzmann( gj=3, gi=1, E=const_E2p2sHe, T=TM)
  
  # H recombination coefficient alphaBH and
  # photoionization coefficient betaH. 
  a1 =4.309 / const_c_Mpc_s * (1e-13/rescale); a2 = -0.6166; a3 = 0.6703; a4 = 0.5300
  T4 = TM/1.0e+04
  alphaH = a1 * T4**a2 / (1.0 + a3 * T4**a4) * 1.0e-06 

  # Multiply alpha by the fudge factor input.F. This step is what makes
  # this simple ode method approximate the full multi-level calculation
  # for H. Note that because beta is derived from alpha, it is also affected
  # by F.   
  # fudge_F = 1.14
  fudge_F = 1.125 # use fudged updated fudge factor from newer recfast 
  
  alphaH *= fudge_F
  betaH = ((alphaH*rescale)/SH)/rescale  # rescale is used to prevent underflow in single precision

  # He case B recombination and photoionization coefficient 
  # No need to multiply a fudge factor.     
  # a1 = 1.691e-12; a2 = 1.519; T0=3.0; T1=3.2026e+04
  a1 = 16.91 / const_c_Mpc_s * (1e-13/rescale); a2 = 1.519; T0=3.0; T1=3.2026e+04
 
  alphaHe = jnp.sqrt(TM/T0)*(1+jnp.sqrt(TM/T0))**(1-a2) \
    * (1.0+jnp.sqrt(TM/T1))**(1+a2)
  alphaHe = a1 / alphaHe / 1.0e+06 
  betaHe = ((alphaHe*rescale) / SHe)/rescale # rescale is used to prevent underflow in single precision
  
  # Cosmological redshifting 
  rescale13 = rescale**(1/3)
  KHe = (const_LyalphaHe/rescale13)**3 / (Hz * 8 * jnp.pi)  # use Peebles coeff. for He # TODO: change units of const_LyalphaHe
  KH =  (const_LyalphaH/rescale13)**3 / (Hz * 8 * jnp.pi)   # use Peebles coeff. for H # TODO: change units of const_LyalphaH

  # Inhibition factors
  fCHe = (1 + KHe*const_Lam2s1sHe * nHe1 * fBHe2p2s) \
    / (1 + KHe * (const_Lam2s1sHe + betaHe) * nHe1 * fBHe2p2s)
  fCH = (1 + KH*const_Lam2s1sH * nH1) / (1 + KH * (const_Lam2s1sH + betaH) * nH1)

  
  # Finally calculate the rate equations.
  #   0 = a, 1 = He, 2 = H,  3 = TM
  dlogadtau = a * Hz
  arescale = rescale

  ε = 1e-4
  dxHepdtau = jax.lax.cond( jnp.fabs(xHep) > ε, 
              lambda x: a  * (-alphaHe*xe*nHtot + betaHe*(fHe/xHep-1.0)*fBHe)*xHep*fCHe, 
              lambda x: a  * (-(alphaHe*arescale)*xe*xHep*(nHtot/arescale) + (betaHe*arescale)*(fHe-xHep)*(fBHe/arescale))*fCHe, None )
              
  dxpdtau   = a * fCH  * jax.lax.cond( jnp.fabs(xp) > ε, 
                                lambda x: (-alphaH*xe*nHtot + betaH*fBH*(1.0/xp-1.0))*xp,
                                lambda x: (-(alphaH*arescale)*(nHtot/arescale)*xe + betaH*fBH*(xp-1.0))*xp, None )
  
  # limit compton term to avoid numerical problems
  # Comp = 8/3 * const_sigmaT * const_aRad / const_me / const_c * TR**4 
  Comp = (4.707988984123603e-06 * TR)**4 / const_c_Mpc_s

  # fHe = param['YHe'] / (const_mHe_mH*(1-param['YHe']))
  compton_term = Comp * xe/(1 + xe + fHe)
  daTdtau      = a**2 * (compton_term * (TR - TM) - Hz * TM )

  # if called in noiseless mode, set the derivative to zero if the temperature is not evolving
  # otherwise the right hand side of the ODE can be noisy on the solution as the system is very stiff
  ε = 1e-5
  daTdtau = jax.lax.cond( noiseless_dT & (jnp.abs(a*TR-a*TM)<ε), lambda x: 0.0, lambda x: daTdtau, None )

  dy = jnp.array([ dlogadtau, dxHepdtau, dxpdtau, daTdtau ]) * tau
  return dy

class VectorField(eqx.Module):
    model: eqx.Module

    def __call__(self, logtau, y, args):
        return self.model(logtau, y, args)   

# @partial(jax.jit, backend='cpu')
# @jax.jit
def compute_thermo( *, param : dict ) -> dict:

    model = drx.ODETerm(VectorField(
        lambda logtau, y , params : model_recfast( logtau=logtau, yin=y, param=params[0])
    ))

    t0 = jnp.log(param['taumin'])
    t1 = jnp.log(param['taumax'])

    y0 = jnp.array( [ jnp.log(param['amin']), param['YHe']/(const_mHe_mH*(1.0-param['YHe'])), 1.0, param['Tcmb'] ] )

    # jax.debug.print('t0 = {}, t1 = {}, y0 = {}, f0 = {}', t0, t1, y0, model_recfast(tau=t0,yin=y0,param=param))

    
    param['fHe'] = param['YHe']/(const_mHe_mH*(1.0-param['YHe']))

    # saveat = drx.SaveAt( dense=True )
    # saveat = drx.SaveAt( t0=True,t1=True, dense=True )
    # saveat = drx.SaveAt( t0=True,t1=True,ts=jnp.geomspace(t0*1.001,t1*0.999,1024))#, steps=True, dense=True )
    # saveat = drx.SaveAt( t0=False, t1=True, ts=jnp.linspace(t0,t1,1024,endpoint=False), steps=True, dense=True )
    saveat = drx.SaveAt( t0=True, t1=True, steps=True )#, dense=True )

    sol =drx.diffeqsolve(
        terms=model,
        solver=GRKT4( ), #drx.NewtonNonlinearSolver(rtol=1e-3,atol=1e-3) ),
        t0=t0,
        t1=t1,
        dt0=(t1-t0)*1e-4,
        y0=y0,
        saveat=saveat,  
        # stepsize_controller = drx.PIDController(rtol=1e-8,atol=1e-10,pcoeff=0.2, icoeff=1 ),
        stepsize_controller = drx.PIDController(rtol=1e-6,atol=1e-8,pcoeff=0.2, icoeff=1, dtmin=1e-6, force_dtmin=True ),
        max_steps=2048,
        args=(param, ),
        # adjoint=drx.RecursiveCheckpointAdjoint(),
        adjoint=drx.DirectAdjoint(),
        # adjoint=drx.ImplicitAdjoint(),
    )

    # jax.debug.print('sol = {}', sol)
    # param['thermo_solution'] = sol

    return sol, param


@partial(jax.jit, static_argnames=("num_thermo",))
def evaluate_thermo( *, param : dict, num_thermo = 2048 ) -> jax.Array:

    sol    = param['sol']

    # convert unused output array entries to NaNs (which are ignored by the interpolation)
    invalid = sol.ts == jnp.inf
    logtau = jnp.where( invalid, jnp.nan, sol.ts )
    y = jnp.zeros_like(sol.ys)
    for i in range(4):
      y = y.at[:,i].set(jnp.where( invalid, jnp.nan, sol.ys[:,i] ))

    # collect output 
    tau       = jnp.exp(logtau)
    dydtau    = jax.vmap( lambda logtau_, y_: model_recfast( logtau=logtau_, yin=y_, param=param, noiseless_dT=True ) )( logtau, y )
    a         = jnp.exp(y[:,0])

    xeHeII    = jax.vmap( lambda a_: Saha_HeII( a_, param) )( a )
    xeHeI     = y[:,1]
    xeHI      = y[:,2]
    xe        = xeHI + xeHeI + xeHeII
    Tm        = y[:,3]/a
    daTmdtau  = dydtau[:,3] / tau
    daTmda    = daTmdtau / dadtau(a=a, param=param) #Hubble( a=a, param=param) 
    mu        = 1/(1 + (1/const_mHe_mH-1) * param['YHe'] + (1-param['YHe']) * xe)

    # mu        = 1/(1 - 0.75 * param['YHe'] + (1 - param['YHe']) * xe)
    cs2       = const_kB/ const_mH / const_c**2 / mu * Tm * (4 - daTmda / (Tm)) /3
    cs2       = cs2.at[0].set( const_kB/ const_mH / const_c**2 / mu[0] * Tm[0] * 4/3 )

    return tau, a, cs2, Tm, mu, xe, xeHI, xeHeI, xeHeII, 
