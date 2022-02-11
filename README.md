# RungeKutta
Module which implements the generic Explicit and Implicit Runge-Kutta algorithm.
The Butcher Tableau can be specified to customize the solvers.

See the examples folder for usage

## Documentation
Documentation is built using Sphinx and Autodoc.
```console
sphinx-build -b html docs/source docs/build
```

## Examples
lotka_volterra.ipynb gives a demonstration of using the ExplicitSolver class.

benchmarking_methods.ipynb compares many different Explicit and Implicit methods on a simple euation.
