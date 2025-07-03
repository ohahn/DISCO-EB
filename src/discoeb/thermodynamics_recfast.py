
####################################################################################################################
# This file is not licensed under the GNU GPL license!
# Large portions of this code, which is part of the DISCO-EB module have been adapted from the recfast code base.
# The original recfast code can be found at: https://www.astro.ubc.ca/people/scott/recfast.html
# The recfast code base is distributed under the following license:
####################################################################################################################
 # Integrator for Cosmic Recombination of Hydrogen and Helium,
 # developed by Douglas Scott (dscott@astro.ubc.ca)
 # based on calculations in the papers Seager, Sasselov & Scott
 # (ApJ, 523, L1, 1999; ApJS, 128, 407, 2000)
 # and "fudge" updates in Wong, Moss & Scott (2008).
 #
 # Permission to use, copy, modify and distribute without fee or royalty at
 # any tier, this software and its documentation, for any purpose and without
 # fee or royalty is hereby granted, provided that you agree to comply with
 # the following copyright notice and statements, including the disclaimer,
 # and that the same appear on ALL copies of the software and documentation,
 # including modifications that you make for internal use or for distribution:
 #
 # Copyright 1999-2010 by University of British Columbia.  All rights reserved.
####################################################################################################################
# This file contains a JAX-compatible version of the recfast code base, which has been adapted for use in the 
# DISCO-EB module by Oliver Hahn. The DISCO-EB module is distributed under the GNU GPL license.
####################################################################################################################
 
import diffrax as drx
import jax 
import jax.numpy as jnp
from jax_cosmo.scipy.integrate import romb
from functools import partial 
from typing import Tuple
from discoeb.cosmo import dadtau, dtauda_, get_aprimeoa

from discoeb.ode_integrators_stiff import GRKT4, Rodas5Transformed
# from diffrax import Tsit5

# Pre-compute constants
const_G = 6.67430e-11               # Gravitational constant [m^3/kg/s^2], PDG 2023
const_mH      = 1.67353284e-27      # H atom mass [kg], PDG 2023
const_me      = 9.1093837015e-31    # Electron mass [kg], PDG 2023
const_mHe_mH  = 3.97146570884       # Helium / Hydrogen mass ratio
const_c       = 2.99792458e+08      # Speed of light [m/s]     
const_h       = 6.62607015e-34      # Planck's constant [Js], PDG 2023
const_kB      = 1.380649e-23        # Boltzman's constant [J/K], PDG 2023
const_sigma   = 6.6524587321e-29    # Thomson scattering cross section [m^2], PDG 2023
const_arad = 4 * 5.670374419e-8 / const_c # Radiation constant [Ws/m^3/K^4], PDG 2023

# bigH = 100.0e3/(1.0e6*3.0856775807e16)  # Ho in s-1
bigH = 3.2407792902755102e-18       # Ho in s-1 
const_dens_fac = 11.223810928601939 # 3 * bigH**2 / (8 * jnp.pi * const_G * const_mH)

const_c2ok    = 1.62581581e4 # K / eV
const_c_Mpc_s = 9.71561189e-15 # Mpc/s

## COMMENTS FOR ALL CONSTANTS HAVE BEEN PRESERVED AS MUCH AS POSSIBLE AS THEY ARE IN THE ORIGINAL CODE

const_EionHe12s = 6.314878282674e5  # HeII 1s ionization energy in [K] 

# 2 photon rates and atomic levels
Lambda = 8.2245809                  # H 2s-1s two photon rate
Lambda_He = 51.3                    # HeI 2s-1s two photon rate
L_H_ion = 1.096787737e7             # level for H ionization in m^-1
L_H_alpha = 8.225916453e6           # level for H Ly alpha in m^-1
L_He1_ion = 1.98310772e7            # level for HeI ionization in m^-1
L_He2_ion = 4.389088863e7           # level for HeII ionization in m^-1
L_He_2s = 1.66277434e7              # level for HeI 2s in m^-1
L_He_2p = 1.71134891e7              # level for He 2p (21P1-11S0) in m^-1    

# Atomic data for HeI
A2P_s = 1.798287e9                  # Einstein A coefficient for He 21P1-11S0
A2P_t = 177.58                      # Einstein A coefficient for He 23P1-11S0  
L_He_2Pt = 1.690871466e7            # level for 23P012-11S0 in m^-1
L_He_2St = 1.5985597526e7           # level for 23S1-11S0 in m^-1
L_He2St_ion = 3.8454693845e6        # level for 23S1-continuum in m^-1
sigma_He_2Ps = 1.436289e-22         # H ionization x-section at HeI 21P1-11S0 freq. in m^2
sigma_He_2Pt = 1.484872e-22         # H ionization x-section at HeI 23P1-11S0 freq. in m^2

# Gaussian fits
AGauss1 = -0.14                     # amplitude of the 1st Gaussian for the H fudging
AGauss2 = 0.05                      # amplitude of the 2nd Gaussian for the H fudging
zGauss1 = 7.28                      # ln(1+z) central value of the 1st Gaussian
zGauss2 = 6.75                      # ln(1+z) central value of the 2nd Gaussian
wGauss1 = 0.18                      # width of the 1st Gaussian
wGauss2 = 0.33                      # width of the 2nd Gaussian

H_frac = 1.e-3 

Lalpha = 1.0/L_H_alpha                        # Ly alpha wavelength in SI units
Lalpha_He = 1.0/L_He_2p                       # Helium I 2p-1s wavelength in SI units
DeltaB = const_h*const_c*(L_H_ion-L_H_alpha)  # energy of first excited state from continuum = 3.4eV
CDB = DeltaB/const_kB                         # CDB=DeltaB/k_B			Constants derived from B1,B2,R
DeltaB_He = const_h*const_c*(L_He1_ion-L_He_2s)  # energy of first excited state from cont. for He = 3.4eV
CDB_He = DeltaB_He/const_kB                   # CDB_He=DeltaB_He/k_B n=2-infinity for He in Kelvin
CB1 = const_h*const_c*L_H_ion/const_kB        # CDB*4.			Lalpha and sigma_Th, calculated
CB1_He1 = const_h*const_c*L_He1_ion/const_kB  # CB1 for HeI ionization potential
CB1_He2 = const_h*const_c*L_He2_ion/const_kB  # CB1 for HeII ionization potential

CR = 2.0*jnp.pi*(const_me/const_h)*(const_kB/const_h) 
CK = Lalpha**3/(8.0*jnp.pi)         
CK_He = Lalpha_He**3/(8.0*jnp.pi) 
CL = const_c*const_h/(const_kB*Lalpha) 
CL_He = const_c*const_h/(const_kB/L_He_2s)
CT = (8.0/3.0)*(const_sigma/(const_me*const_c))*const_arad  
Bfact = const_h*const_c*(L_He_2p-L_He_2s)/const_kB  
CL_PSt = const_h*const_c*(L_He_2Pt - L_He_2St)/const_kB
const_h_c_L_He_2St_kB = const_h*const_c*L_He_2St/(const_kB)

# Pequignot, Petitjean & Boisson fitting parameter for Hydrogen
a_PPB = 4.309
b_PPB = -0.6166
c_PPB = 0.6703
d_PPB = 0.5300

# Verner and Ferland type fitting parameter for Helium
a_VF = 10.**(-16.744)
b_VF = 0.711
T_0 = 10**(0.477121)
T_1 = 10**(5.114)

# HeI triplet recombination rate fitting parameters
a_trip = 10.**(-16.306)
b_trip = 0.761
  
def ionization(a, y, params):
  param = params[0]
  z = 1/a - 1
  # Pre-compute derived constants
  H = param['H0']/100.0
  HO = H*bigH
  fu = 1.105
  mu_H = 1.0/(1.0-param['YHe'])
  # mu_T = const_mHe_mH/(const_mHe_mH-(const_mHe_mH-1.0)*param['YHe'])
  fHe = param['YHe']/(const_mHe_mH*(1.0-param['YHe']))
  Nnow = const_dens_fac * H * H * param['Omegab'] / mu_H

  x_H = y[0]
  x_He = y[1]
  x = x_H + fHe * x_He
  Tmat = jnp.abs(y[2])

  # Calculate common terms once
  n = Nnow * (1 + z)**3
  n_He = fHe * n
  Trad = param['Tcmb'] * (1 + z)
  z_term = (1 + z)
  
  # Hubble parameter calculation
  # Hprime = a'/a, dtau = dt/a -> da/dtau/a = da/dt = Ha
  Hz = (1e-5*get_aprimeoa(param=param, aexp=a)) / a * const_c * bigH
  
  # Temperature and rate calculations
  Tmat_1e4 = Tmat / 1e4
  CR_Tmat_15 = (CR * Tmat)**1.5
  
  # Get the radiative rates
  Rdown = 1e-19 * a_PPB * Tmat_1e4**b_PPB / (1 + c_PPB * Tmat_1e4**d_PPB)
  Rup = Rdown * CR_Tmat_15 * jnp.exp(-CDB / Tmat)

  # Calculate He rates using a fit
  sq_0 = jnp.sqrt(Tmat / T_0)
  sq_1 = jnp.sqrt(Tmat / T_1)
  
  sq_0_term = 1 + sq_0
  sq_1_term = 1 + sq_1
  
  Rdown_He_common = a_VF / (sq_0 * sq_0_term**(1 - b_VF) * sq_1_term**(1 + b_VF))
  Rdown_He = Rdown_He_common
  Rup_He = 4 * Rdown_He * CR_Tmat_15 * jnp.exp(-CDB_He / Tmat)

  # Calculate Boltzmann factor with numerical stability
  He_Boltz = jnp.exp(jnp.minimum(680.0, Bfact / Tmat))

  # HeI calculations
  Rdown_trip = a_trip / (sq_0 * sq_0_term**(1 - b_trip) * sq_1_term**(1 + b_trip))
  Rup_trip = Rdown_trip * jnp.exp(-const_h*const_c*L_He2St_ion/(const_kB*Tmat)) * CR_Tmat_15 * (4/3)
  
  # Peebles coefficient calculation
  K_gaussian = 1 + AGauss1*jnp.exp(-((jnp.log(z_term)-zGauss1)/wGauss1)**2) + AGauss2*jnp.exp(-((jnp.log(z_term)-zGauss2)/wGauss2)**2)
  K = CK / Hz * K_gaussian
  
  # He calculations
  one_minus_x_He = 1 - x_He
  one_minus_x_H = 1 - x_H
  n_He_1_minus_x_He = n_He * one_minus_x_He
  
  tauHe_s = A2P_s * CK_He * 3 * n_He_1_minus_x_He / Hz
  pHe_s = (1 - jnp.exp(-tauHe_s)) / tauHe_s
  
  # Doppler calculation
  Doppler_term = jnp.sqrt(2 * const_kB * Tmat / (const_mH * const_mHe_mH * const_c**2))
  Doppler_2p = const_c * L_He_2p * Doppler_term
  Doppler_2Pt = const_c * L_He_2Pt * Doppler_term
  
  # Gamma calculations
  gamma_2Ps_term = 3 * A2P_s * fHe * one_minus_x_He * const_c**2
  gamma_2Ps_denom = jnp.sqrt(jnp.pi) * sigma_He_2Ps * 8 * jnp.pi * Doppler_2p * one_minus_x_H * (const_c * L_He_2p)**2
  gamma_2Ps = gamma_2Ps_term / gamma_2Ps_denom
  
  # AHcon calculation for H continuum opacity
  pb = 0.36  # value from KIV (2007)
  qb = 0.86  # He fudge factor
  AHcon = A2P_s / (1 + pb * (gamma_2Ps**qb))
  
  # K_He calculation with numerical stability
  K_He_cond1 = jnp.logical_or(x_He < 5e-9, x_He > 0.98)
  K_He_cond2 = x_H < 0.9999999
  
  K_He_default = CK_He / Hz
  K_He_case1 = 1.0 / ((A2P_s * pHe_s + AHcon) * 3.0 * n_He_1_minus_x_He)
  K_He_case2 = 1.0 / (A2P_s * pHe_s * 3.0 * n_He_1_minus_x_He)
  
  K_He = jnp.where(K_He_cond1, K_He_default, 
           jnp.where(K_He_cond2, K_He_case1, K_He_case2))
  
  # Triplet calculations
  tauHe_t = A2P_t * n_He_1_minus_x_He * 3 / (8 * jnp.pi * Hz * L_He_2Pt**3)
  pHe_t = (1 - jnp.exp(-tauHe_t)) / tauHe_t
  
  gamma_2Pt_term = 3 * A2P_t * fHe * one_minus_x_He * const_c**2
  gamma_2Pt_denom = jnp.sqrt(jnp.pi) * sigma_He_2Pt * 8 * jnp.pi * Doppler_2Pt * one_minus_x_H * (const_c * L_He_2Pt)**2
  gamma_2Pt = gamma_2Pt_term / gamma_2Pt_denom
  
  # KIV (2007) parameters
  pb_t = 0.66
  qb_t = 0.9
  AHcon_t = A2P_t / (1 + pb_t * gamma_2Pt**qb_t) / 3
  
  # CfHe_t calculation
  exp_CL_PSt = jnp.exp(-CL_PSt / Tmat)
  CfHe_t_case1 = A2P_t * pHe_t * exp_CL_PSt
  CfHe_t_case2 = (A2P_t * pHe_t + AHcon_t) * exp_CL_PSt
  CfHe_t = jnp.where(x_H > 0.99999, CfHe_t_case1, CfHe_t_case2)
  CfHe_t /= (Rup_trip + CfHe_t)  # "C" factor for triplets
  
  # Time calculations
  timeTh = (1 / (CT * Trad**4)) * (1 + x + fHe) / x
  timeH = 2 / (3 * HO * z_term**1.5)
  
  # f0 calculation
  Hz_z_term = Hz * z_term
  x_x_H_n_Rdown = x * x_H * n * Rdown
  Rup_1_minus_x_H_exp_CL = Rup * one_minus_x_H * jnp.exp(-CL / Tmat)
  rate_diff = x_x_H_n_Rdown - Rup_1_minus_x_H_exp_CL
  
  K_Lambda_n_1_minus_x_H = K * Lambda * n * one_minus_x_H
  denom_term = 1 / fu + K_Lambda_n_1_minus_x_H / fu + K * Rup * n * one_minus_x_H
  
  f0_case1 = 0.0
  f0_case2 = rate_diff / Hz_z_term
  f0_case3 = (rate_diff * (1 + K_Lambda_n_1_minus_x_H)) / (Hz_z_term * denom_term)
  
  f0 = jnp.where(x_H > 0.99, f0_case1,
          jnp.where(x_H > 0.985, f0_case2, f0_case3))
  
  # f1 calculation
  x_x_He_n_Rdown_He = x * x_He * n * Rdown_He
  Rup_He_1_minus_x_He_exp_CL_He = Rup_He * one_minus_x_He * jnp.exp(-CL_He / Tmat)
  rate_diff_He = x_x_He_n_Rdown_He - Rup_He_1_minus_x_He_exp_CL_He
  
  K_He_Lambda_He_n_He_1_minus_x_He_He_Boltz = K_He * Lambda_He * n_He_1_minus_x_He * He_Boltz
  He_denom_term = 1 + K_He * (Lambda_He + Rup_He) * n_He_1_minus_x_He * He_Boltz
  
  f1_main = (rate_diff_He * (1 + K_He_Lambda_He_n_He_1_minus_x_He_He_Boltz)) / (Hz_z_term * He_denom_term)
  
  # Triplet contribution
  trip_rate_diff = x * x_He * n * Rdown_trip - one_minus_x_He * 3 * Rup_trip * jnp.exp(-const_h_c_L_He_2St_kB / Tmat)
  trip_contrib = trip_rate_diff * CfHe_t / Hz_z_term
  trip_cond = jnp.logical_or(x_He < 5e-9, x_He > 0.98)
  
  f1 = jnp.where(x_He < 1e-8, 0.0, 
          f1_main + jnp.where(trip_cond, 0.0, trip_contrib))
  
  # f2 calculation
  epsilon = Hz * (1 + x + fHe) / (CT * Trad**3 * x)
  f2_case1 = Tmat / z_term
  f2_case2 = CT * (Trad**4) * x / (1 + x + fHe) * (Tmat - Trad) / Hz_z_term + 2 * Tmat / z_term
  f2 = jnp.where(timeTh < H_frac * timeH, f2_case1, f2_case2)
  
  dzda = -1/a**2
  return jnp.array([f0, f1, f2]) * dzda

def solve_ionization( *, astart : float, aend : float, ystart : jnp.ndarray, rtol : float = 1e-6, atol : float = 1e-8, max_steps : int = 128, param : dict ) -> jnp.ndarray:
  sol =drx.diffeqsolve(
        terms=drx.ODETerm(ionization),
        solver=GRKT4(),
        # solver=Tsit5(),
        t0=astart,
        t1=aend,
        dt0=jnp.abs(astart*1e-3),
        y0=ystart[:3],
        stepsize_controller = drx.PIDController(rtol=rtol,atol=atol), 
        max_steps=max_steps,
        args=(param,),
        # adjoint=drx.RecursiveCheckpointAdjoint(),
        adjoint=drx.ForwardMode(),
        # adjoint=drx.ImplicitAdjoint(),
        throw=False,
    )
  dyda = ionization( aend, sol.ys[-1,:], (param,) )
  return jnp.append(sol.ys[-1,:], dyda)

def Saha_HeII( a, param ):
    """
    Saha equation for HeII recombination
    """
    def NHnow( a, YHe, H0, Omegab ):
      const_Hfac = 1/(1.0e+06*3.0856775807e+13) # 1 km/s/Mpc in 1/s
      mu_H = 1/(1-YHe) 
      rho_c = 3 * (H0*const_Hfac)**2 / (8 * jnp.pi * const_G) 
      return rho_c * Omegab / (const_mH * mu_H) / a**3
    T = param['Tcmb'] / a
    betaE = const_EionHe12s / T
    A = 1 + param['fHe']
    B = 1 + 2*param['fHe']
    R = (2*jnp.pi* const_me * const_kB / const_h**2 * T )**1.5 / NHnow( a, param['YHe'], param['H0'], param['Omegab'] ) * jnp.exp( - betaE )

    xe = jax.lax.cond( R>1e5, 
                      lambda x: param['fHe'] * (1 - B/R + (1 + 5*param['fHe'] + 6*param['fHe']**2)/R**2), # asymptotic expansion to prevent truncation errors
                      lambda x:  -(R-A)/2 + jnp.sqrt( (R-A)**2/4 + R*B ) - A, None )
    
    return xe


def compute_thermal_history(*, a0: float, a1: float, N: int, rtol: float = 1e-3, atol: float = 1e-6, param: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Optimized version of compute_thermal_history that's more GPU-friendly.
    """
    # Generate scale factor array with improved spacing 
    a = jnp.append(jnp.geomspace(a0, 1e-4, 16, endpoint=False), jnp.geomspace(1e-4, a1, N+1-16))
    
    # Precompute common values used throughout the function
    H = param['H0']/100.0
    HO = H * bigH
    mu_H = 1.0/(1.0-param['YHe'])
    Nnow = 3.0 * HO * HO * param['Omegab'] / (8.0 * jnp.pi * const_G * mu_H * const_mH)
    fHe = param['YHe']/(const_mHe_mH*(1.0-param['YHe']))
    Tcmb = param['Tcmb']
    Tcmb2 = Tcmb**2
    Tcmb3 = Tcmb2*Tcmb
    
    # Initial values for the state
    zstart_init = 1.0 / a[0] - 1.0
    initial_state = jnp.array([1.0, 1.0, Tcmb * (1.0 + zstart_init), 0.0, 0.0, -Tcmb * (1.0 + zstart_init)])
    
    # Pre-compute all z values and dzda values to avoid redundant calculations
    z_values = 1.0 / a[1:] - 1.0
    dzda_values = -1.0 / a[1:]**2
    
    # Prepare arrays of boundary conditions for quick masking
    high_z_mask = z_values > 3500.0
    
    def scan_body(prev_state, idx):
        """Process one step with scan instead of fori_loop"""
        i = idx
        astart = a[i]
        aend = a[i+1]
        zend = z_values[i]
        dzda = dzda_values[i]
        
        # Create default outputs for each condition path
        high_z_output = jnp.array([1.0, 1.0, param['Tcmb']*(1.0 + zend), 0.0, 0.0, -param['Tcmb']*(1.0 + zend)])
        
        # Handle HeI recombination path
        x_H0 = 1.0
        rhs_He = (jnp.exp(1.5 * jnp.log(CR * param['Tcmb']/(1.0+zend)) 
                - CB1_He1/(param['Tcmb']*(1.0+zend))) / Nnow) * 4.0
        x_He0 = 0.5*(jnp.sqrt((rhs_He-1.0)**2 + 4.0*(1.0+fHe)*rhs_He) - (rhs_He-1.0))
        
        # Pre-compute the derivative expression for HeI
        dxHeIdz = ((-3*(-(CB1_He1/Tcmb) + CR*Tcmb)**1.5*(Nnow + 2*fHe*Nnow + 
              4*(-((CB1_He1 - CR*Tcmb**2)/(Tcmb*(1+zend))))**1.5 - 
              jnp.sqrt(Nnow**2 + (16*(-CB1_He1 + CR*Tcmb**2)**3)/(Tcmb3*(1 + zend)**3) + 
                8*(1 + 2*fHe)*Nnow*(-((CB1_He1 - CR*Tcmb**2)/(Tcmb*(1+zend))))**1.5)))
           /(Nnow*(1 + zend)**2.5*jnp.sqrt(Nnow**2 + (16*(-CB1_He1 + CR*Tcmb**2)**3)/
               (Tcmb**3*(1 + zend)**3) + 8*(1 + 2*fHe)*Nnow*(-((CB1_He1 - CR*Tcmb**2)/(Tcmb + Tcmb*zend)))**1.5)))
        
        HeI_output = jnp.array([x_H0, (x_He0 - 1.0)/fHe, param['Tcmb']*(1.0+zend), 
                               0.0, dxHeIdz*dzda, -param['Tcmb']*(1.0+zend)])
        
        # Handle HI recombination path
        rhs_H = jnp.exp(1.5*jnp.log(CR*param['Tcmb']/(1.0+zend)) - CB1/(param['Tcmb']*(1.0+zend))) / Nnow
        x_H0_HI = 0.5*(jnp.sqrt(rhs_H**2 + 4.0*rhs_H) - rhs_H)
        
        dxHdz = ((3*((2*(-((CB1 - CR*Tcmb**2)/(Tcmb*(1+zend))))**1.5)/(1 + zend) + 
                   ((CB1 - CR*Tcmb**2)*(2*CB1**2 - 4*CB1*CR*Tcmb**2 + Tcmb**2*
                 (2*CR**2*Tcmb**2 + Nnow*(1 + zend)**2*jnp.sqrt(-((CB1 - CR*Tcmb**2)/(Tcmb*(1+zend)))))))/
            (Tcmb**1.5*(1 + zend)**2.5*jnp.sqrt((-CB1 + CR*Tcmb**2)*
                (CB1**2 - 2*CB1*CR*Tcmb**2 + Tcmb**2*(CR**2*Tcmb**2 + Nnow*(1 + zend)**2*
                                                      jnp.sqrt(-((CB1 - CR*Tcmb**2)/(Tcmb*(1+zend))))))))))/(2.*Nnow))

        # Compute solution by solving ODE
        ode_solution = solve_ionization(astart=astart, aend=aend, ystart=prev_state, 
                                       rtol=rtol, atol=atol, max_steps=128, param=param)
        
        # For HI recombination, override specific elements of the solution
        HI_output = ode_solution.at[0].set(x_H0_HI).at[3].set(dxHdz*dzda)
        
        # Determine which case applies based on conditions
        is_high_z = high_z_mask[i]
        is_HeI_recomb = ~is_high_z & (prev_state[1] > 0.99)
        is_HI_recomb = ~is_high_z & ~is_HeI_recomb & (prev_state[0] > 0.99)
        
        # Combine outputs using masks without attempting to reshape the boolean scalars
        result = jnp.where(is_high_z, high_z_output, 
                 jnp.where(is_HeI_recomb, HeI_output,
                 jnp.where(is_HI_recomb, HI_output, ode_solution)))
        
        return result, result
    
    # Use scan which is often more efficient than fori_loop for stateful operations
    _, y_results = jax.lax.scan(
        scan_body,
        initial_state,
        jnp.arange(N)
    )
    
    return y_results.T, a[1:]


def evaluate_thermo( *, param : dict, num_thermo = 2048 ) -> jax.Array:
    
    param['fHe'] = param['YHe']/(const_mHe_mH*(1.0-param['YHe']))
    
    y, a = compute_thermal_history( a0=param['amin'], a1=param['amax'], N=num_thermo, param=param )

    # extract the relevant quantities from the solution
    xeHI      = y[0,:]
    xeHeI     = y[1,:]
    xeHeII    = jax.vmap( lambda a_: Saha_HeII( a_, param) )( a )
    xe        = xeHI + param['fHe'] * xeHeI + xeHeII
    mu        = 1/(1 + (1/const_mHe_mH-1) * param['YHe'] + (1-param['YHe']) * xe)
    Tm        = y[2,:]

    # extract the derivatives that were also computed, which allows to compute cs2 and dxedtau
    dxeHIda  = y[3,:]
    dxeHeIda = y[4,:]
    dxHeIIda = jax.vmap( lambda a_: jax.grad( lambda aa_: Saha_HeII( aa_, param) )( a_ ) )( a )
    dTmda    = y[5,:]
    daTmda   = Tm + a * dTmda
    cs2      = const_kB/ const_mH / const_c**2 / mu * Tm * (4 - daTmda / (Tm)) /3
    dxedtau  = (dxeHIda + param['fHe'] * dxeHeIda + dxHeIIda) * dadtau(a=a, param=param)

    # # compute conformal times tau for all entries in a
    # def step(cum_tau, a_pair):
    #     a_low, a_high = a_pair
    #     # Integrate dtauda_ between a_low and a_high using romb
    #     dtau = romb(lambda a: dtauda_(a, param['grhom'], param['grhog'], param['grhor'],
    #                     param['Omegam'], param['OmegaDE'],
    #                     param['w_DE_0'], param['w_DE_a'],
    #                     param['Omegak'], param['Neff'], param['Nmnu'],
    #                     param['logrhonu_of_loga_spline']),
    #         a_low, a_high)
    #     new_tau = cum_tau + dtau
    #     return new_tau, new_tau

    # # Stack adjacent pairs of aexp for integration over each interval
    # segments = jnp.stack([a[:-1], a[1:]], axis=1)
    # tau0 = param['taumin']
    # # Use scan to perform the cumulative integration
    # tau_segments = jax.lax.scan(step, tau0, segments)[1]
    # # Prepend the initial tau (0.0) to obtain the tau array corresponding to aexp
    # tau = jnp.concatenate([jnp.array([tau0]), tau_segments], axis=0)

    # compute conformal times tau for all entries in a - PARALLELIZED VERSION
    # Define a function to compute dtau for a single segment
    def compute_dtau(a_pair):
        a_low, a_high = a_pair
        return romb(lambda a: dtauda_(a, param['grhom'], param['grhog'], param['grhor'],
                      param['Omegam'], param['OmegaDE'],
                      param['w_DE_0'], param['w_DE_a'],
                      param['Omegak'], param['Neff'], param['Nmnu'],
                      param['logrhonu_of_loga_spline']),
          a_low, a_high)

    # Stack adjacent pairs of aexp for integration over each interval
    segments = jnp.stack([a[:-1], a[1:]], axis=1)
    tau0 = param['taumin']
    
    # Apply the integration to all segments in parallel using vmap
    all_dtaus = jax.vmap(compute_dtau)(segments)
    
    # Compute cumulative sum of dtau values
    tau_segments = tau0 + jnp.cumsum(all_dtaus)
    
    # Prepend the initial tau value to obtain the full tau array
    tau = jnp.concatenate([jnp.array([tau0]), tau_segments], axis=0)

    return param, tau, a, cs2, Tm, mu, xe, xeHI, xeHeI, xeHeII, dxedtau