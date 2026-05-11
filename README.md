# pymc-bilby

Plugin for using pymc with bilby.

This plugin exposes the `pymc` sampler via the `bilby.samplers` entry point.
Once installed, you can select it in `bilby.run_sampler` using `sampler='pymc'`.


## Installation

The package can be install using pip

```
pip install pymc-bilby
```

or conda

```
conda install conda-forge:pymc-bilby
```


## Changes compared to the original bilby implementation

- `cores` and equivalent keyword arguments are now correctly translated
- `run_sampler` is now wrapped in `signal_wrapper`
