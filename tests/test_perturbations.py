import pytest
import jax
import jax.numpy as jnp


from discoeb.background import evolve_background
from discoeb.perturbations import evolve_perturbations, evolve_perturbations_batched

from conftest import k_CLASS, Pkbc_CLASS

## Cosmological Parameters
Tcmb    = 2.7255
YHe     = 0.248
Omegam  = 0.3099
Omegab  = 0.0488911
Omegac  = Omegam - Omegab
w_DE_0  = -0.99
w_DE_a  = 0.0
cs2_DE  = 1.0
# Initialize neutrinos.
num_massive_neutrinos = 1
mnu     = 0.06  #eV
Tnu     = (4/11)**(1/3) #0.71611 # Tncdm of CLASS
Neff    = 3.046 # -1 if massive neutrino present
N_nu_mass = 1
N_nu_rel = Neff - N_nu_mass * (Tnu/((4/11)**(1/3)))**4
h       = 0.67742
A_s     = 2.1064e-09
n_s     = 0.96822
k_p     = 0.05

# modes to sample
nmodes = 512
kmin = 1e-5
kmax_small = 1e+1
aexp = 0.01

## Compute Background evolution
param = {}
param['Omegam']  = Omegam
param['Omegab']  = Omegab
# param['OmegaDE'] = OmegaDE
param['w_DE_0']  = w_DE_0
param['w_DE_a']  = w_DE_a
param['cs2_DE']  = cs2_DE
param['Omegak']  = 0.0
param['A_s']     = A_s
param['n_s']     = n_s
param['H0']      = 100*h
param['Tcmb']    = Tcmb
param['YHe']     = YHe
param['Neff']    = N_nu_rel
param['Nmnu']    = N_nu_mass
param['mnu']     = mnu

iout = -1
fac = 2 * jnp.pi**2 * A_s

relevant_indices = (k_CLASS >= kmin) & (k_CLASS <= kmax_small)
test_points = k_CLASS[relevant_indices]
test_values = Pkbc_CLASS[relevant_indices]


class TestEvolvePerturbations:

    @pytest.fixture(autouse=True)
    def compute_background_param(self):
        self.param = evolve_background(param=param, thermo_module='RECFAST')

    def test_power_spectrum_small_k_vs_CLASS(self):
        aexp_out = jnp.array([aexp]) 
      
        y, kmodes = evolve_perturbations( param=self.param, kmin=kmin, kmax=kmax_small, num_k=nmodes, aexp_out=aexp_out,
                                  lmaxg = 31, lmaxgp = 31, lmaxr = 31, lmaxnu = 31, nqmax = 5,
                                  max_steps=1024,
                                  rtol=1e-5, atol=1e-5)
        
        
        Pkbc = fac *(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,6]**2
        disco_eb_values = jnp.interp(test_points, kmodes, Pkbc)
        # relative_error = jnp.abs(disco_eb_values - test_values)/test_values
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.01), "relative error exceeded 0.01"
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.005), "relative error exceeded 0.005"
        
    
    @pytest.mark.parametrize("batch_size", [4, 8, 16, 32, 64])
    def test_power_spectrum_small_k_vs_CLASS_batched(self, batch_size):
        aexp_out = jnp.array([aexp]) 

        y, kmodes = evolve_perturbations_batched( param=self.param, kmin=kmin, kmax=kmax_small, num_k=nmodes, aexp_out=aexp_out,
                                  lmaxg = 31, lmaxgp = 31, lmaxr = 31, lmaxnu = 31, nqmax = 5,
                                  max_steps=1024,
                                  rtol=1e-5, atol=1e-5, batch_size=batch_size )
        
        
        Pkbc = fac *(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,6]**2
        disco_eb_values = jnp.interp(test_points, kmodes, Pkbc)
        # relative_error = jnp.abs(disco_eb_values - test_values)/test_values
        # jax.debug.print("relative_error {}", relative_error)

        assert jnp.allclose(disco_eb_values, test_values, rtol=0.01), "relative error exceeded 0.01"
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.005), "relative error exceeded 0.005"
