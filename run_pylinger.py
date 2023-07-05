import jax
import jax.numpy as jnp

from pylinger_cosmo import cosmo
from pylinger_pt_single_mode_TCA import evolve_perturbations

## Cosmological Parameters
Tcmb = 2.7255
YHe = 0.248
Omegam = 0.276
Omegab = 0.0455
OmegaL = 1.0-Omegam
num_massive_neutrinos = 1
mnu=0.06 #eV
Neff=2.046 # -1 if massive neutrino present
standard_neutrino_neff=Neff+num_massive_neutrinos
h = 0.703
A_s = 2.1e-9
n_s = 0.965
k_p = 0.05

## accuracy parameters
lmax_g  = 12
lmax_gp = 12
lmax_r  = 17
lmax_nu = 17
rtol    = 1e-3
atol    = 1e-4


## Compute Background evolution
print('Computing background evolution....')
cp = cosmo(Omegam=Omegam, Omegab=Omegab, OmegaL=OmegaL, H0=100*h, Tcmb=Tcmb, YHe=YHe, Neff=Neff, Nmnu=num_massive_neutrinos, mnu=mnu )

# Compute Perturbations
nmodes = 10
kmin = 1e-3
kmax = 1e0
aexp_out = jnp.geomspace(1e-5,1e-2,2)
print('Computing perturbation evolution....')
y, kmodes = evolve_perturbations( param=cp.param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out )

print(y[:,-1,3])
