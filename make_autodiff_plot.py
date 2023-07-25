import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import diffrax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

from pylinger_background import evolve_background
from pylinger_perturbations import evolve_one_mode, evolve_perturbations


## Cosmological Parameters
Tcmb    = 2.7255
YHe     = 0.248
Omegam  = 0.3099
Omegab  = 0.0488911
# OmegaDE = 1.0-Omegam
w_DE_0  = -1.0
w_DE_a  = 0.0
cs2_DE  = 1.0
num_massive_neutrinos = 1
mnu     = 0.06 #0.06 #eV
Neff    = 2.046 # -1 if massive neutrino present
standard_neutrino_neff=Neff+num_massive_neutrinos
H0      = 67.742
A_s     = 2.1064e-09
n_s     = 0.96822


fieldnames = ['$\\Omega_m$', '$\\Omega_b$', '$A_s$', '$n_s$', '$H_0$', '$T_{CMB}$', '$Y_{He}$', '$N_{eff}$', '$m_{\\nu}$', '$w_0$', '$w_a$', '$c_s^2$']
fiducial_cosmo_param = jnp.array([Omegam, Omegab, A_s, n_s, H0, Tcmb, YHe, Neff, mnu, w_DE_0, w_DE_a, cs2_DE])



# @eqx.filter_jit
def f_of_Omegam( args ):
    param = {}
    param['Omegam'] = args[0]
    param['Omegab'] = args[1]
    param['OmegaDE'] = 1-args[0]
    param['Omegak'] = 0.0
    A_s = args[2]
    n_s = args[3]
    param['H0'] = args[4]
    param['Tcmb'] = args[5]
    param['YHe'] = args[6]
    param['Neff'] = args[7]
    param['Nmnu'] = num_massive_neutrinos
    param['mnu'] = args[8]
    param['w_DE_0'] = args[9]
    param['w_DE_a'] = args[10]
    param['cs2_DE'] = args[11]

    k_p  = 0.05

    param = evolve_background(param=param)
    
    k = 1e-2

    # Compute Perturbations
    lmaxg  = 12
    lmaxgp = 12
    lmaxr  = 17
    lmaxnu = 17
    nqmax  = 15

    rtol   = 1e-3
    atol   = 1e-4

    # Compute Perturbations
    nmodes = 256
    kmin = 1e-3
    kmax = 1e1
    # aexp_out = jnp.array([1e-2,1e-1]) 
    aexp_out = jnp.geomspace(1e-2,1,2)

    y, kmodes = evolve_perturbations( param=param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out )

    iout = -1
    fac = 2.5
    Pkc = fac * A_s*(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,3]**2 
    Pkb = fac * A_s*(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,5]**2 
    Pkm = (param['Omegam']-param['Omegab']) * Pkc + param['Omegab'] * Pkb
    
    return Pkm


#print( f_of_Omegam(fiducial_cosmo_param).shape )

## compute the jacobian at few times, but for many k
k  = jnp.geomspace(1e-3,1e1,256)
dy = jax.jacfwd(f_of_Omegam)(fiducial_cosmo_param)

#print( dy.shape )

fig,ax = plt.subplots(4,3,sharex=True,figsize=(13,10),layout='constrained')

for i,ff in enumerate(fieldnames):
    iy = i//3
    ix = i%3
    ax[iy,ix].semilogx(k, dy[:,i],label='$P_{b+c}$')

    ax[iy,ix].set_title(ff)
    

plt.savefig('derivatives.pdf')
