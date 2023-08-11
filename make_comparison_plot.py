import jax
# jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_debug_nans", True)
#jax.config.update("jax_disable_jit", True)
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import time
from classy import Class # import classy module
from scipy.interpolate import interp1d

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

from pylinger_background import evolve_background
from pylinger_perturbations import evolve_perturbations

from jax.lib import xla_bridge
print('platform = ',xla_bridge.get_backend().platform)

## Cosmological Parameters
Tcmb    = 2.7255
YHe     = 0.248
Omegam  = 0.3099
Omegab  = 0.0488911
OmegaDE = 1.0-Omegam
w_DE_0  = -0.9
w_DE_a  = 0.0
cs2_DE  = 1.0
num_massive_neutrinos = 1
mnu     = 0.06  #eV
Neff    = 2.046 # -1 if massive neutrino present
standard_neutrino_neff=Neff+num_massive_neutrinos
h       = 0.67742
A_s     = 2.1064e-09
n_s     = 0.96822
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
    'P_k_max_1/Mpc': 10.0,
    'z_max_pk':1010.0,
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
    'w0_fld' : w_DE_0,
    'wa_fld' : w_DE_a,
    'cs2_fld' : 1.0,
    'use_ppf' : 'no',   # this needs to be switched off, otherwise a weird bump appears at low k when w_0>-1
  })
# run class
print('Computing CLASS solution...')
LambdaCDM.compute()
data_class, k_class, z_class = LambdaCDM.get_transfer_and_k_and_z()
thermo = LambdaCDM.get_thermodynamics()

def get_class_power( fieldname, zout ):
  tk, k, z = LambdaCDM.get_transfer_and_k_and_z()
  TT = interp1d( np.log(1/(1+z)), tk[fieldname], axis=1 )( np.log(1/(1+zout)) )
  res =  2*np.pi**2*A_s*(k/k_p)**(n_s - 1) * k**(-3) * TT**2
  return res, k


## Compute Background evolution
param = {}
param['Omegam']  = Omegam
param['Omegab']  = Omegab
param['OmegaDE'] = OmegaDE
param['w_DE_0']  = w_DE_0
param['w_DE_a']  = w_DE_a
param['cs2_DE']  = cs2_DE
param['Omegak']  = 0.0
param['A_s']     = A_s
param['n_s']     = n_s
param['H0']      = 100*h
param['Tcmb']    = Tcmb
param['YHe']     = YHe
param['Neff']    = Neff
param['Nmnu']    = num_massive_neutrinos
param['mnu']     = mnu

for p in param:
  print(f'{p} = {param[p]}')

print('Computing pyLinger background solution...')
param = evolve_background(param=param)

# Compute Perturbations
nmodes = 256
kmin = 1e-4
kmax = 1e1
# aexp_out = jnp.array([1e-2,1e-1]) 
aexp_out = jnp.geomspace(1e-3,1,9)

print('Computing pyLinger perturbations solution...')
# this is a blind call, to JIT compile, the line below will execute and measure execution time
y, kmodes = evolve_perturbations( param=param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out, rtol=1e-3, atol=1e-3 )#, lmaxg=50, lmaxgp = 25, lmaxr = 50, lmaxnu = 50, rtol=1e-3, atol=1e-3 )

start = time.time()
y, kmodes = evolve_perturbations( param=param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out, rtol=1e-4, atol=1e-4 )#, lmaxg=50, lmaxgp = 25, lmaxr = 50, lmaxnu = 50, rtol=1e-6, atol=1e-6 )
print(f'Time: {time.time() - start}s')

#
# if True:
#   # save the file to disk
#   np.save( 'file_deltac.npy', y[:,:,3] )
#   np.save( 'file_deltac.npy', y[:,:,3] )
#   np.save( 'file_aout.npy', aexp_out )
#   np.save( 'file_kmodes.npy', kmodes )

## evaluate the power spectra


fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,10), layout='constrained', gridspec_kw={'height_ratios': [2, 1]})

colors = pl.cm.jet(np.linspace(0,1,y.shape[1]))

for i,iout in enumerate(range(y.shape[1])):

  print(f'pyEB zout={1/y[0,iout,0]-1}')
  zout = jnp.maximum(1/y[0,iout,0]-1,0.0)
  fac = 2 * np.pi**2 * A_s
  Pkc = fac *(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,3]**2
  Pkb = fac *(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,5]**2
  Pkg = fac *(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,7]**2

  Pkc_CLASS, k_CLASS = get_class_power('d_cdm', zout)
  Pkb_CLASS, k_CLASS = get_class_power('d_b', zout)

  iratio_Pkc = Pkc/np.exp(np.interp( np.log(kmodes), np.log(k_CLASS), np.log(Pkc_CLASS) ))-1.0
  iratio_Pkb = Pkb/np.exp(np.interp( np.log(kmodes), np.log(k_CLASS), np.log(Pkb_CLASS) ))-1.0

  ## output the maximum relative difference
  print(f'Pkc mean rel diff = {jnp.mean(jnp.abs(iratio_Pkc))}, max rel diff = {jnp.max(jnp.abs(iratio_Pkc))} at k = {kmodes[jnp.argmax(jnp.abs(iratio_Pkc))]}')
  print(f'Pkb mean rel diff = {jnp.mean(jnp.abs(iratio_Pkb))}, max rel diff = {jnp.max(jnp.abs(iratio_Pkb))} at k = {kmodes[jnp.argmax(jnp.abs(iratio_Pkb))]}')

  ax[0].loglog( kmodes[::4], Pkc[::4], 'x', color=colors[i], lw=1, ms=5 , alpha=0.5 ) #, label='CDM')
  ax[0].loglog( kmodes[::4], Pkb[::4], '+', color=colors[i], lw=1, ms=5 , alpha=0.5 ) #, label='baryon

  ax[0].loglog( k_CLASS, Pkc_CLASS, ls='-', color=colors[i], lw=1, label=f'z={zout : .2f}' ) #, label='CDM (CLASS)',
  ax[0].loglog( k_CLASS, Pkb_CLASS, ls='--', color=colors[i], lw=1 ) #, label='baryon (CLASS)'

  # ax.loglog( kmodes, Pkc_CAMB, label='CAMB cdm', color='C0', ls='--')
  # ax.loglog( kmodes, Pkb_CAMB, label='CAMB baryon', color='C1', ls='--')

  logPc_interp = interp1d( np.log(k_CLASS), np.log(Pkc_CLASS) )
  logPb_interp = interp1d( np.log(k_CLASS), np.log(Pkb_CLASS) )

  ax[1].semilogx( kmodes, Pkc/np.exp(logPc_interp(np.log(kmodes)))-1.0, lw=1, ls='-', label='CDM err', color=colors[i])
  ax[1].semilogx( kmodes, Pkb/np.exp(logPb_interp(np.log(kmodes)))-1.0, lw=1, ls='--', label='baryon err', color=colors[i])

ax[0].legend(ncol=3)
ax[1].set_ylim((-0.1,0.1))
ax[1].axhline(0.0, ls='--', color='k')
ax[1].axhline(0.01, ls=':', color='k')
ax[1].axhline(-0.01, ls=':', color='k')

ax[1].set_xlabel('$k / {\\rm Mpc}^{-1}$')
ax[1].set_ylabel('rel. err.')
ax[0].set_ylabel('$P(k) / {\\rm Mpc}^{3}$')

ax[1].set_xlim((1e-4,10))
plt.savefig('plot_comparison.pdf',bbox_inches='tight')
