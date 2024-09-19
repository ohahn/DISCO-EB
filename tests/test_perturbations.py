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
kmax = 1e+2
kmax_small=1e+1
aexp = 0.01
aexp_out = jnp.array([aexp]) 


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

relevant_indices = (k_CLASS >= kmin) & (k_CLASS <= kmax)
test_points = k_CLASS[relevant_indices]
test_values = Pkbc_CLASS[relevant_indices]

small_kmax_indices = (k_CLASS >= kmin) & (k_CLASS <= kmax_small)
small_kmax_test_points = k_CLASS[relevant_indices]
small_kmax_test_values = Pkbc_CLASS[relevant_indices]

@jax.jit
def perturbations_jit(param): 
    return evolve_perturbations( 
        param=param, aexp_out=aexp_out, kmin=kmin, kmax=kmax, num_k=256,
        lmaxg = 11, lmaxgp = 11, lmaxr = 11, lmaxnu  = 8,
        nqmax = 3, rtol = 1e-3, atol = 1e-3,
        pcoeff = 0.25, icoeff = 0.80, dcoeff = 0.0,
        factormax  = 20.0, factormin  = 0.3, max_steps  = 2048)

@jax.jit
def perturbations_batched_jit(param):
    return evolve_perturbations_batched(param=param,
        aexp_out=aexp_out, kmin=kmin, kmax=kmax, num_k=256,
        lmaxg = 11, lmaxgp = 11, lmaxr = 11, lmaxnu  = 8,
        nqmax = 3, rtol = 1e-3, atol = 1e-3,
        pcoeff = 0.25, icoeff = 0.80, dcoeff = 0.0,
        factormax  = 20.0, factormin  = 0.3, max_steps  = 2048, batch_size=32)

@pytest.fixture(scope="session")
def background_param():
    bg_param = evolve_background(param=param, thermo_module='RECFAST')
    yield bg_param

class TestEvolvePerturbations:

    @pytest.mark.parametrize("batch_size", [4, 8, 16, 32, 64])
    def test_power_spectrum_varying_batchsize_small_kmax(self, batch_size, background_param):

        y, kmodes = evolve_perturbations_batched(
            param=background_param, kmin=kmin, kmax=kmax_small, num_k=nmodes, aexp_out=aexp_out,
            lmaxg = 31, lmaxgp = 31, lmaxr = 31, lmaxnu = 31, nqmax = 5,
            max_steps=2048, rtol=1e-4, atol=1e-4, batch_size=batch_size
        )
        
        Pkbc = fac *(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,6]**2
        disco_eb_values = jnp.interp(small_kmax_test_points, kmodes, Pkbc)
        # relative_error = jnp.abs(disco_eb_values - test_values)/test_values
        # jax.debug.print("relative_error {}", relative_error)
        assert jnp.allclose(disco_eb_values, small_kmax_test_values, rtol=0.1), "relative error exceeded 0.1"
        assert jnp.allclose(disco_eb_values, small_kmax_test_values, rtol=0.01), "relative error exceeded 0.01"
        assert jnp.allclose(disco_eb_values, small_kmax_test_values, rtol=0.005), "relative error exceeded 0.005"
        assert jnp.allclose(disco_eb_values, small_kmax_test_values, rtol=0.001), "relative error exceeded 0.005"


    def test_power_spectrum_vs_CLASS_batched(self, background_param):

        y, kmodes = evolve_perturbations_batched(
            param=background_param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out,
            lmaxg = 31, lmaxgp = 31, lmaxr = 31, lmaxnu = 31, nqmax = 5,
            max_steps=2048, rtol=1e-4, atol=1e-4, batch_size=32
        )
        
        Pkbc = fac *(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,6]**2
        disco_eb_values = jnp.interp(test_points, kmodes, Pkbc)
        # relative_error = jnp.abs(disco_eb_values - test_values)/test_values
        # jax.debug.print("relative_error {}", relative_error)
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.1), "relative error exceeded 0.1"
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.01), "relative error exceeded 0.01"
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.005), "relative error exceeded 0.005"
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.001), "relative error exceeded 0.005"


    def test_power_spectrum_vs_CLASS(self, background_param):

        y, kmodes = evolve_perturbations( 
            param=background_param, kmin=kmin, kmax=kmax, num_k=nmodes, aexp_out=aexp_out,
            lmaxg = 31, lmaxgp = 31, lmaxr = 31, lmaxnu = 31, nqmax = 5,
            max_steps=2048, rtol=1e-5, atol=1e-5
        )
        
        
        Pkbc = fac *(kmodes/k_p)**(n_s - 1) * kmodes**(-3) * y[:,iout,6]**2
        disco_eb_values = jnp.interp(test_points, kmodes, Pkbc)
        # relative_error = jnp.abs(disco_eb_values - test_values)/test_values
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.1), "relative error exceeded 0.1"
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.01), "relative error exceeded 0.01"
        assert jnp.allclose(disco_eb_values, test_values, rtol=0.005), "relative error exceeded 0.005"

    def test_benchmark_evolve_perturbations(self, benchmark, background_param):
        _ = perturbations_jit(param=background_param)

        _ = benchmark(perturbations_jit, param=background_param)


    def test_benchmark_evolve_perturbations_batched(self, benchmark, background_param):
        _ = perturbations_batched_jit(param=background_param)

        _ = benchmark(perturbations_batched_jit, param=background_param)