# Monte Carlo Option Pricing Engine

This project implements a modular Python framework for option pricing using Monte Carlo simulation. It grew out of reading Chapter 9 of "Mathematical Modeling and Computation in Finance" by Oosterlee and Grzelak. See `01_MC_options_pricing.ipynb` for a minimal example illustrating the framework's basic functionality.

Option sensitivities, so called "greeks", can be estimated by the finite difference method (see `03_greeks.ipynb` notebook).
Importance sampling is also implemented for variance reduction (see `02_importance_sampling.ipynb` notebook). 


## Requirements
- numpy
- scipy
- matplotlib
- jupyter
  

## File structure
```text
PricingEngineMC/
│
├── stoch_models.py            # Stochastic model definitions
├── numerical_schemes.py       # Euler, Milstein and exact schemes
├── path_simulation.py         # generates paths 
├── option_contracts.py        # Financial instruments and payoffs
├── estimators.py              # Statistical estimation
├── pricing_engine.py          # Monte Carlo pricing framework
├── greeks.py                  # Numerical Greek estimation
|
├── analytical_results.py      # analytical Black-Scholes prices and Greeks
├── randomness.py              # Random number transformations
├── importance_sampling.py     # Importance sampling strategies
│
└── notebooks/
   ├── 01_MC_options_pricing.ipynb
   ├── 02_importance_sampling.ipynb
   └── 03_greeks.ipynb
