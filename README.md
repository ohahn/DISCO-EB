# DISCO-EB - The DISCO-DJ Einstein-Boltzmann module
Implementation of a differentiable linear Einstein-Boltzmann solver for cosmology in JAX -- a module of the DISCO-DJ framework (DIfferentiable Simulations for COsmology, Done with Jax).

Note that this program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.

Currently supported features (the list is growing, so check back):

- Autodifferentiable via JAX
- Standard LCDM model with Quintessence DE fluid (w0,wa) and massive neutrinos (one species)
- Thermal history solver based on a simplified Recfast implementation in the module
- Numerous example jupyter notebooks, e.g. for Euclid-like Fisher forecasts
  
New modules/plugins can be easily added (see how to contribute in CONTRIBUTING.md file). We are enthusiastic if extensions/improvements that are of broader interest are re-integrated into the master branch and DISCO-EB grows as a community effort.

## Installation

### Simple Install

```bash
pip install git+https://github.com/ohahn/DISCO-EB.git
```
### For Development

In a fresh virtual environment:

```bash
git clone https://github.com/ohahn/DISCO-EB.git
cd DISCO-EB
pip install -e .
```

## Getting started

Start with the sample notebook [nb_minimal_example.ipynb](https://github.com/ohahn/DISCO-EB/blob/master/notebooks/nb_minimal_example.ipynb) in the [notebooks subdirectory](https://github.com/ohahn/DISCO-EB/tree/master/notebooks). It explains how to compute a matter power spectrum and take a derivative w.r.t. a cosmological parameter.

## Contributing and Licensing

See file CONTRIBUTING.md on how to contribute to the development. 

The software is licensed under GPL v3 (see file LICENSE).

## Citing in scientific publications or presentations

If you use DISCO-EB in your scientific work, you are required to acknowledge this by linking to this repository and citing the relevant papers:

- Hahn et al. 2024 [arXiv:2311.03291](https://arxiv.org/abs/2311.03291)


## Required Python Packages
- JAX
- diffrax==0.4.1
- equinox
- jaxtyping
- jax_cosmo
