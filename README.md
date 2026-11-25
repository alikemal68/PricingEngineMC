Monte Carlo Option Pricing Engine

This project implements a modular Python framework for pricing options using Monte Carlo simulation. 
It grew out of reading chapter 9 of "Mathematical Modeling and Computation in Finance" by Oosterlee and Grzelak.
See the notebook 01_MC_options_pricing.ipynb for basic functionalities. 

Option sensitivities, so called "greeks", can be estimated by the finite difference method (see 03_greeks.ipynb notebook).
Importance sampling is also implemented for variance reduction (see 02_importance_samplin.ipynb notebook). 

requirements
- numpy
- scipy
- matplotlib
- jupyter

```text
PricingEngineMC/
│
├── stoch_models.py            # Stochastic model definitions
├── numerical_schemes.py       # Euler, Milstein and exact schemes
├── path_simulation.py         # Generic path simulation
├── option_contracts.py        # Financial instruments and payoffs
├── estimators.py              # Statistical estimation
├── pricing_engine.py          # Monte Carlo pricing framework
├── greeks.py                  # Numerical Greek estimation
|
├── analytical_results.py      # Black-Scholes prices and Greeks
├── randomness.py              # Random number transformations
├── importance_sampling.py    # Importance sampling strategies
│
└── notebooks/
   ├── 01_MC_options_pricing.ipynb
   ├── 02_importance_samplin.ipynb
   └── 03_greeks.ipynb
