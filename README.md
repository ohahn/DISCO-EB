# DISCO-EB - The DISCO-DJ Einstein-Boltzmann module
Implementation of a differentiable linear Einstein-Boltzmann solver for cosmology in JAX -- a module of the DISCO-DJ framework (DIfferentiable Simulations for COsmology, Done with Jax).

Note that this program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.

Currently supported features (the list is growing, so check back):

- Autodifferentiable via JAX
- Standard LCDM model with Quintessence DE fluid (w0,wa) and massive neutrinos (one species)
- Thermal history solver based on a simplified Recfast implementation in the module
- Numerous example jupyter notebooks, e.g. for Euclid-like Fisher forecasts
  
New modules/plugins can be easily added (see how to contribute in CONTRIBUTING.md file)


## Contributing and Licensing

See file CONTRIBUTING.md on how to contribute to the development. 

The software is licensed under GPL v3 (see file LICENSE). 

Please note the separate licensing for Panphasia (see external/panphasia/LICENSE).


## Citing in scientific publications or presentations

If you use DISCO-EB in your scientific work, you are required to acknowledge this by linking to this repository and citing the relevant papers:

- Hahn et al. 2024 [arXiv:2311.03291](https://arxiv.org/abs/2311.03291)


## Required Python Packages
- JAX
- diffrax==0.4.1
- equinox
- jaxtyping
- jax_cosmo
