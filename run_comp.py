import jax
# jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from classy import Class # import classy module
from scipy.interpolate import interp1d
log_folder = 'runs'


from pylinger_background import evolve_background
from pylinger_perturbations import evolve_perturbations

## Cosmological Parameters
Tcmb    = 2.7255
YHe     = 0.248
Omegam  = 0.276
Omegab  = 0.0455
OmegaDE = 1.0-Omegam
w_DE_0  = -0.9
w_DE_a  = 0.0
num_massive_neutrinos = 1
mnu     = 0.06 #0.06 #eV
Neff    = 2.046 # -1 if massive neutrino present
standard_neutrino_neff=Neff+num_massive_neutrinos
h       = 0.703
A_s     = 2.1e-9
n_s     = 0.965
k_p     = 0.05


## CLASS setup
# create instance of the class "Class"
LambdaCDM = Class()
# pass input parameters
LambdaCDM.set({
    'Omega_k':0.0,
    'Omega_b':Omegab,
    'Omega_cdm':(Omegam-Omegab),
    'N_ur':Neff,
    'N_ncdm':num_massive_neutrinos,
    'm_ncdm':mnu,
    'h':h,
    'A_s':A_s,
    'n_s':n_s,
    'tau_reio':0.06, 
    'gauge':'synchronous',
    # 'reio_parametrization' : 'reio_none',
    'k_per_decade_for_pk' : 100,
    'k_per_decade_for_bao' : 100,
    'compute damping scale' : 'yes',
    'tol_perturbations_integration' : 1e-07,
    'tol_background_integration' : 1e-07,
    'hyper_flat_approximation_nu' : 7000,
    'T_cmb':Tcmb,
    'YHe':YHe,
    'output':'mPk,mTk,vTk',
    'lensing':'no',
    'P_k_max_1/Mpc':10.0,
    'z_max_pk':1000.0,
    # these are high precision reference settings
    'start_small_k_at_tau_c_over_tau_h' : 0.0015, #0.0004,
    'start_large_k_at_tau_h_over_tau_k' : 0.05,
    'tight_coupling_trigger_tau_c_over_tau_h' : 0.005,
    'tight_coupling_trigger_tau_c_over_tau_k' : 0.008,
    'start_sources_at_tau_c_over_tau_h' : 0.006,
    'l_max_g' : 50,
    'l_max_pol_g' : 25,
    'l_max_ur' : 50,
    'l_max_ncdm' : 50,
    'Omega_Lambda' : 0.0,
    'w0_fld' : -1.0, #w_DE_0,
    'wa_fld' : 0.0, #w_DE_a,
  })
# run class
print('Computing CLASS solution...')
LambdaCDM.compute()
thermo = LambdaCDM.get_thermodynamics()

def get_class_power( fieldname, zout ):
  tk, k, z = LambdaCDM.get_transfer_and_k_and_z()
  TT = interp1d( np.log(1/(1+z)), tk[fieldname], axis=1 )( np.log(1/(1+zout)) ) 
  res =  2*np.pi**2*A_s*(k/k_p*h)**(n_s - 1) * k**(-3) * TT**2 *h**3
  return res, k


## Compute Background evolution
param = {}
param['Omegam'] = Omegam
param['Omegab'] = Omegab
param['OmegaDE'] = OmegaDE
param['w_DE_0'] = w_DE_0
param['w_DE_a'] = w_DE_a
param['Omegak'] = 0.0
param['A_s'] = A_s
param['n_s'] = n_s
param['H0'] = 100*h
param['Tcmb'] = Tcmb
param['YHe'] = YHe
param['Neff'] = Neff
param['Nmnu'] = num_massive_neutrinos
param['mnu'] = mnu

for p in param:
  print(f'{p} = {param[p]}')

print('Computing pyLinger background solution...')
param = evolve_background(param=param)


# Compute Perturbations
nmodes = 128
kmin = 1e-4
kmax = 1e1
# aexp_out = jnp.array([1e-2,1e-1]) 
aexp_out = jnp.geomspace(1e-3,1,2)

print('Computing pyLinger perturbations solution...')
y, kmodes = evolve_perturbations( param=param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out )


## evaluate the power spectra
iout = -1
print(f'pyEB zout={1/y[0,iout,0]-1}')
zout = jnp.maximum(1/y[0,iout,0]-1,0.0)
fac = 2.5
Pkc = fac * A_s*(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,3]**2 
Pkb = fac * A_s*(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,5]**2 
Pkg = fac * A_s*(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,7]**2 

Pkc_CLASS, k_CLASS = get_class_power('d_cdm', zout)
Pkb_CLASS, k_CLASS = get_class_power('d_b', zout)

iratio_Pkc = Pkc/np.exp(np.interp( np.log(kmodes), np.log(k_CLASS), np.log(Pkc_CLASS) ))-1.0
iratio_Pkb = Pkb/np.exp(np.interp( np.log(kmodes), np.log(k_CLASS), np.log(Pkb_CLASS) ))-1.0

## output the maximum relative difference
print(f'Pkc mean rel diff = {jnp.mean(jnp.abs(iratio_Pkc))}, max rel diff = {jnp.max(jnp.abs(iratio_Pkc))} at k = {kmodes[jnp.argmax(jnp.abs(iratio_Pkc))]}')
print(f'Pkb mean rel diff = {jnp.mean(jnp.abs(iratio_Pkb))}, max rel diff = {jnp.max(jnp.abs(iratio_Pkb))} at k = {kmodes[jnp.argmax(jnp.abs(iratio_Pkb))]}')


fig, ax = plt.subplots(2,1, sharex=True, figsize=(6,8) )
fac = 2.5
# fac = 2*np.pi**2/ (2*np.pi)**1.5
ax[0].loglog( kmodes, Pkc*fac,label='pyEB cdm', color='C0')
# ax.loglog( kmodes, Pkc_CAMB, label='CAMB cdm', color='C0', ls='--')
ax[0].loglog( kmodes, Pkb*fac,label='pyEB baryon', color='C1')
# ax.loglog( kmodes, Pkb_CAMB, label='CAMB baryon', color='C1', ls='--')
ax[0].loglog( k_CLASS, Pkc_CLASS, label='CLASS cdm', ls='--', color='C0')
ax[0].loglog( k_CLASS, Pkb_CLASS, label='CLASS baryon', ls='--', color='C1')
ax[0].legend()

ax[1].semilogx( kmodes, Pkc/np.exp(np.interp( np.log(kmodes), np.log(k_CLASS), np.log(Pkc_CLASS) ))-1.0, label='cdm err', color='C0')
ax[1].semilogx( kmodes, Pkb/np.exp(np.interp( np.log(kmodes), np.log(k_CLASS), np.log(Pkb_CLASS) ))-1.0, label='baryon err', color='C1')

ax[1].set_ylim((-0.1,0.1))
ax[1].axhline(0.0, ls='--', color='k')
ax[1].axhline(0.01, ls=':', color='k')
ax[1].axhline(-0.01, ls=':', color='k')

ax[1].set_xlim((1e-4,10))
plt.savefig('plot_comparison.pdf',bbox_inches='tight')
