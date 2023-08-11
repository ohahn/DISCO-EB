import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from matplotlib import rc
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('axes', titlesize=24)
rc('axes', labelsize=20)
rc('axes', axisbelow=False)
rc('lines',linewidth=2)
# lines.markersize : 10
rc('xtick', labelsize=16)
rc('xtick.major',size=10)
rc('xtick.minor',size=5)
rc('xtick',direction='in')
rc('ytick', labelsize=16)
rc('ytick.major',size=10)
rc('ytick.minor',size=5)
rc('ytick',direction='in')
rc('legend',fontsize='x-large')

#################################

# unfortunately autodiff currently needs double precision!
jax.config.update("jax_enable_x64", True) 

from pylinger_background import evolve_background
from pylinger_perturbations import evolve_one_mode, evolve_perturbations


## Cosmological Parameters
Tcmb    = 2.7255
YHe     = 0.248
Omegam  = 0.3099
Omegab  = 0.0488911
# OmegaDE = 1.0-Omegam
w_DE_0  = -0.999
w_DE_a  = 0.00
cs2_DE  = 0.99
num_massive_neutrinos = 1
mnu     = 0.06  #eV
Neff    = 2.046 # -1 if massive neutrino present
standard_neutrino_neff=Neff+num_massive_neutrinos
H0      = 67.742
A_s     = 2.1064e-09
n_s     = 0.96822


# list of parameters with respect to which we take derivatives
fieldnames = ['H_0', '\\Omega_m', '\\Omega_b', 'N_{eff}', 'm_{\\nu}', 'T_{CMB}', 'Y_{He}', 'A_s', 'n_s', 'w_0', 'w_a', 'c_s^2']
fiducial_cosmo_param = jnp.array([H0, Omegam, Omegab, Neff, mnu, Tcmb, YHe, A_s, n_s,  w_DE_0, w_DE_a, cs2_DE])


def Pk_of_cosmo( args ):
    """ Compute the matter (b+c) power spectrum for a given set of cosmological parameters"""
    param = {}
    param['Omegam'] = args[1]
    param['Omegab'] = args[2]
    param['OmegaDE'] = 1-args[1]
    param['Omegak'] = 0.0
    A_s = args[7]
    n_s = args[8]
    param['H0'] = args[0]
    param['Tcmb'] = args[5]
    param['YHe'] = args[6]
    param['Neff'] = args[3]
    param['Nmnu'] = num_massive_neutrinos
    param['mnu'] = args[4]
    param['w_DE_0'] = args[9]
    param['w_DE_a'] = args[10]
    param['cs2_DE'] = args[11]

    k_p  = 0.05

    ## compute the background evolution
    param = evolve_background(param=param)
    

    # Compute Perturbations
    lmaxg  = 12
    lmaxgp = 12
    lmaxr  = 17
    lmaxnu = 17
    nqmax  = 15

    rtol   = 1e-5
    atol   = 1e-5

    # Compute Perturbations
    nmodes = 256  # number of modes to compute, reduce to speed up calculation
    kmin = 1e-4
    kmax = 1e1
    aexp_out = jnp.geomspace(1e-2,1,2)

    y, kmodes = evolve_perturbations( param=param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out,
                                      lmaxg=lmaxg, lmaxgp=lmaxgp, lmaxr=lmaxr, lmaxnu=lmaxnu, nqmax=nqmax,
                                       rtol=rtol, atol=atol )

    iout = -1
    fac = 2.5
    Pkc = fac * A_s*(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,3]**2 
    Pkb = fac * A_s*(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,5]**2 
    Pkm = (param['Omegam']-param['Omegab']) * Pkc + param['Omegab'] * Pkb
    
    return Pkm


## compute the jacobian 
k  = jnp.geomspace(1e-4,1e1,256) # number of modes to compute, reduce to speed up calculation

dy = jax.jacfwd(Pk_of_cosmo)(fiducial_cosmo_param)
y  = Pk_of_cosmo( fiducial_cosmo_param )

## make the plot
plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = -20  # pad is in points...

fig,ax = plt.subplots(4,3,sharex=True,figsize=(16,18),layout='constrained')
title_bbox = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=0.667)

for i,ff in enumerate(fieldnames):
    iy = i%3
    ix = i//3
    ax[ix,iy].semilogx(k, dy[:,i]/y,label='$P_{b+c}$')
    ax[ix,iy].axhline(0.0, ls=':', color='k')
    ax[ix,iy].set_title(f'$\\mathrm{{d}} \\log P(k) / \\mathrm{{d}} {ff}$', bbox=title_bbox, fontsize=18)
    
for a in ax[-1,:]:
    a.set_xlabel('$k / h \\mathrm{Mpc}^{-1}$')

plt.savefig('derivatives.pdf')
