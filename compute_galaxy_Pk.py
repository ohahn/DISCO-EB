import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# unfortunately autodiff currently needs double precision!
jax.config.update("jax_enable_x64", True) 

from pylinger_background import evolve_background
from pylinger_perturbations import evolve_one_mode, evolve_perturbations


## Cosmological Parameters
Tcmb    = 2.7255
YHe     = 0.245421
Omegam  = 0.3084961
Omegab  = 0.0488911
# OmegaDE = 1.0-Omegam
w_DE_0  = -0.99
w_DE_a  = 0.0
cs2_DE  = 0.99
num_massive_neutrinos = 1
mnu     = 0.06  #eV
Neff    = 2.046 # -1 if massive neutrino present
standard_neutrino_neff=Neff+num_massive_neutrinos
H0      = 67.742
h       = H0 / 100. 
sigma8  = 0.807192
n_s     = 0.96822

## Galaxy Parameters
b1      = 1.3  # linear bias, should be close to 1.3 for ELGs

## scale factor to evaluate
a       = 1.0

# number of modes
numk    = 256    # number of modes to compute, reduce to speed up calculation
kmin    = 1e-3
kmax    = 1e+1

# list of parameters with respect to which we take derivatives
fieldnames = ['a','b_1','\\Omega_m', '\\Omega_b', '\\sigma_8', 'n_s', 'H_0', 'N_{eff}', 'm_{\\nu}', 'w_0', 'w_a']
fiducial_cosmo_param = jnp.array([a,b1, Omegam, Omegab, sigma8, n_s, H0, Neff, mnu, w_DE_0, w_DE_a])

def Pkbiased( args ):
    """ Compute the matter (b+c) power spectrum for a given set of cosmological parameters"""
    param = {}
    a = args[0]
    b1 = args[1]
    param['Omegam'] = args[2]
    param['Omegab'] = args[3]
    param['OmegaDE'] = 1-args[2]
    param['Omegak'] = 0.0
    sigma8 = args[4]
    n_s = args[5]
    param['H0'] = args[6]
    h = args[6] / 100.0
    param['Tcmb'] = Tcmb
    param['YHe'] = YHe
    param['Neff'] = args[7]
    param['Nmnu'] = num_massive_neutrinos
    param['mnu'] = args[8]
    param['w_DE_0'] = args[9]
    param['w_DE_a'] = args[10]
    param['cs2_DE'] = cs2_DE

    k_p  = 0.05

    ## compute the background evolution
    param = evolve_background(param=param)
    

    # Compute Perturbations
    lmaxg  = 12
    lmaxgp = 12
    lmaxr  = 17
    lmaxnu = 17
    nqmax  = 15

    rtol   = 1e-3
    atol   = 1e-6

    # Compute Perturbations
    aexp_out = jnp.array([a])

    y, kmodes = evolve_perturbations( param=param, kmin=kmin, kmax=kmax, num_k=numk, aexp_out=aexp_out,
                                      lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax,
                                       rtol=rtol, atol=atol )

    iout = -1
    fac = 2.5

    iq0 = 10 + lmaxg + lmaxgp + lmaxr

    rhonu = param['rhonu_of_a_spline'].evaluate(a)
    fnu = (param['grhor'] * rhonu / a**4) / (param['grhom'] / a**3)
    
    Pkc =  (kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,3]**2  
    Pkb =  (kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,5]**2  
    Pknu = (kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,iq0]**2
    
    Pkbc = (param['Omegam']-param['Omegab'])/param['Omegam'] * Pkc + param['Omegab']/param['Omegam'] * Pkb
    
    Pkm =  Pkbc + fnu * Pknu

    Wth = lambda kR : 3*(jnp.sin(kR)-kR*jnp.cos(kR))/(kR)**3
    dsigma = Wth(kmodes*8.0/h)**2 * Pkm * kmodes**2

    sigma8_2_measured = jnp.trapz( y=(kmodes*dsigma), x=jnp.log(kmodes) ) / (2*jnp.pi)

    Pkm  = Pkm  * sigma8**2/ sigma8_2_measured
    # Pkbc = Pkbc * sigma8**2/ sigma8_2_measured

    return Pkm * b1**2

## compute the jacobian 
k  = jnp.geomspace(kmin,kmax,numk) # number of modes to compute, reduce to speed up calculation
Pk = Pkbiased( fiducial_cosmo_param )

dP = jax.jacfwd(Pkbiased)(fiducial_cosmo_param)

fig, ax = plt.subplots()

ax.loglog(k, Pk)
ax.set_xlabel('$k (h/Mpc)$')
ax.set_ylabel('$P (Mpc/h)^3$')
ax.set_title(f'biased power spectrum at z={1/a-1}')
plt.savefig('Pk.pdf')

## make the plot
fig,ax = plt.subplots(4,3,sharex=True,figsize=(13,10),layout='constrained')

for i,ff in enumerate(fieldnames):
    iy = i//3
    ix = i%3
    ax[iy,ix].semilogx(k, dP[:,i],label='$P_{b+c}$')
    ax[iy,ix].axhline(0.0, ls=':', color='k')
    ax[iy,ix].set_title(f'$dP(k) / d{ff}$')
    
for a in ax[-1,:]:
    a.set_xlabel('$k / h Mpc^{-1}$')

plt.savefig('derivatives_bias.pdf')

