import pytest

import jax
import jax.numpy as jnp


from discoeb.background import evolve_background

from conftest import a_RECFAST, xe_RECFAST

import matplotlib.pyplot as plt

## Cosmological Parameters
Tcmb    = 2.7255
YHe     = 0.248
Omegam  = 0.3099
Omegab  = 0.0488911
w_DE_0  = -0.99
w_DE_a  = 0.0
cs2_DE  = 1.0

# Initialize neutrinos.
mnu     = 0.06  #eV
Tnu     = (4/11)**(1/3) #0.71611 # Tncdm of CLASS
Neff    = 3.046 # -1 if massive neutrino present
N_nu_mass = 1
N_nu_rel = Neff - N_nu_mass * (Tnu/((4/11)**(1/3)))**4
h       = 0.67742
A_s     = 2.1064e-09
n_s     = 0.96822

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



## Compute Background evolution
param = {}
param['Omegam']  = Omegam
param['Omegab']  = Omegab
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

background_recfast_jit = jax.jit(lambda param: evolve_background(param=param, thermo_module='RECFAST'))

class TestEvolveBackground:
    
    @pytest.fixture(autouse=True)
    def jit_functions(self):
        _ = background_recfast_jit(param=param)

    def test_evolve_background_recfast_vs_DISCOEB_baseline(self, benchmark):
        solution_param = benchmark(background_recfast_jit, param=param)


        xe = solution_param['xe_of_tau_spline'].evaluate(solution_param['tau_of_a_spline'].evaluate(solution_param['a']))
        a =  solution_param['a']

        assert a.shape == a_RECFAST.shape
        assert xe.shape == xe_RECFAST.shape
        assert jnp.allclose(a, a_RECFAST, rtol=0.01), "relative error exceeded 0.01"
        assert jnp.allclose(xe, xe_RECFAST, rtol=0.005), "relative error exceeded 0.005"