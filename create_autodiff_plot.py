import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import diffrax
import matplotlib.pyplot as plt
%matplotlib widget

jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

from pylinger_background import evolve_background
from pylinger_perturbations import evolve_one_mode, evolve_perturbations


## Cosmological Parameters
Tcmb    = 2.7255
YHe     = 0.248
Omegam  = 0.276
Omegab  = 0.0455
OmegaDE = 1.0-Omegam
w_DE_0  = -1.0
w_DE_a  = 0.0
cs2_DE  = 1.0
num_massive_neutrinos = 1
mnu     = 0.06 #0.06 #eV
Neff    = 2.046 # -1 if massive neutrino present
standard_neutrino_neff=Neff+num_massive_neutrinos
h       = 0.703
A_s     = 2.1e-9
n_s     = 0.965
k_p     = 0.05


# @eqx.filter_jit
def f_of_Omegam( args, k_p ):
    param = {}
    param['Omegam'] = args[0]
    param['Omegab'] = args[1]
    param['OmegaDE'] = 1-args[0]
    param['Omegak'] = 0.0
    param['A_s'] = args[2]
    param['n_s'] = args[3]
    param['H0'] = args[4]
    param['Tcmb'] = args[5]
    param['YHe'] = args[6]
    param['Neff'] = args[7]
    param['Nmnu'] = num_massive_neutrinos
    param['mnu'] = args[8]
    param['w_DE_0'] = args[9]
    param['w_DE_a'] = args[10]
    param['cs2_DE'] = args[11]

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
    Pkc = fac * param['A_s']*(kmodes/k_p)**(param['n_s'] - 1) * kmodes**(-3) * y[:,iout,3]**2 
    Pkb = fac * param['A_s']*(kmodes/k_p)**(param['n_s'] - 1) * kmodes**(-3) * y[:,iout,5]**2 
    Pkm = (param['Omegam']-param['Omegab']) * Pkc + param['Omegab'] * Pkb
    
    return jnp.array([Pkc, Pkb, Pkm])


k  = jnp.geomspace(1e-3,1e1,128)
dy = jax.jacfwd(f_of_Omegam)([Omegam, Omegab, A_s, n_s, h, Tcmb, YHe, Neff, mnu, w_DE_0, w_DE_a, cs2_DE], k_p)

fieldnames = ['$\\Omega_m$', '$\\Omega_b$', '$A_s$', '$n_s$', '$h$', '$T_{CMB}$', '$Y_{He}$', '$N_{eff}$', '$m_{\\nu}$', '$w_0$', '$w_a$', '$c_s^2$']

fig,ax = plt.subplots(3,4,sharex=True,layout='constrained')

for i,ff in enumerate(fieldnames):
    ix = i//3
    iy = i%3
    ax[iy,ix].semilogx(k, dy[i][0],label='$P_c$')
    ax[iy,ix].semilogx(k, dy[i][1],label='$P_b$')
    ax[iy,ix].semilogx(k, dy[i][2],label='$P_m$')

    ax[iy,ix].set_title(ff)
    

plt.savefig('derivatives.pdf')