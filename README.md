# PyReactSim

![PyReactSim](https://drive.google.com/uc?export=view&id=19oyWL9UqyvSNqgRT_lu1b-Z_scvV5gBx)

[![PyPI Downloads](https://static.pepy.tech/badge/pyreactsim/month)](https://pepy.tech/projects/pyreactsim)
![PyPI Version](https://img.shields.io/pypi/v/pyreactsim)
![Supported Python Versions](https://img.shields.io/pypi/pyversions/pyreactsim.svg)
![License](https://img.shields.io/pypi/l/pyreactsim)

**PyReactSim** is a Python package for chemical reactor simulation and design. It enables engineers and researchers to model common reactor types such as batch, CSTR, and plug flow reactors, incorporating reaction kinetics, thermodynamics, and transport effects. The initial version focuses on **steady-state simulations**, allowing users to analyze conversion, selectivity, and temperature profiles under different operating conditions. Designed for flexibility and integration, PyReactSim can be used alongside thermodynamic and phase equilibrium tools to build comprehensive reaction and process models, with dynamic simulation capabilities planned for future releases.

## Available Reactor Methods (from examples)

The main reactor workflows follow the same pattern:

1. Define reactor options (`...ReactorOptions`) and heat transfer options.
2. Build a thermo source with `build_thermo_source(...)`.
3. Create a reactor using `create_..._reactor(...)`.
4. Run simulation with `.simulate(...)`.

- `br` (Batch Reactor)
  - API: `create_batch_reactor(...)` and `BatchReactor.simulate(time_span=..., solver_options=...)`
  - Typical use: time-domain simulation for closed systems (no inlet/outlet flow during reaction).
  - Examples:
    - `examples/batch reactor/gas-batch-exe-1.py`
    - `examples/batch reactor/liquid-batch-exe-1.py`

- `cstr` (Continuous Stirred-Tank Reactor)
  - API: `create_cstr_reactor(...)` and `CSTRReactor.simulate(time_span=..., solver_options=...)`
  - Typical use: dynamic holdup behavior with inlet/outlet streams and mixing assumptions.
  - Examples:
    - `examples/cstr/gas-cstr-exp-1.py`
    - `examples/cstr/liquid-cstr-exp-1.py`

- `pfr` (Plug Flow Reactor)
  - API: `create_pfr_reactor(...)` and `PFRReactor.simulate(volume_span=..., solver_options=...)`
  - Typical use: integration along reactor volume for tubular-reactor behavior.
  - Examples:
    - `examples/pfr/gas-pfr-exp-1.py`
    - `examples/pfr/liquid-pfr-exp-1.py`

- `pbr` (Packed-Bed Reactor)
  - API: `create_pbr_reactor(...)` and `PBRReactor.simulate(volume_span=..., solver_options=...)`
  - Typical use: packed-bed modeling with catalyst-bulk effects (for example, `bulk_density` in model inputs).
  - Examples:
    - `examples/pbr/gas-pbr-exp-1.py`
    - `examples/pbr/gas-pbr-exp-2.py`
    - `examples/pbr/liquid-pbr-exp-1.py`
    - `examples/pbr/liquid-pbr-exp-2.py`

If you want to know how each method is configured in practice (options, model inputs, and solver setup), use the example scripts above as the reference.

## Examples

You can also run PyReactSim examples in Google Colab:

- [PyReactSim Colab Examples](https://colab.research.google.com/drive/1_2MVP6EFmcIkqMsNfHG0OybHczuerZVH?usp=sharing)

## 🤝 Contributing

Contributions are highly welcome — bug fixes, new calculation routines, mixture models, extended unit tests, documentation, etc.

## 📝 License

This project is distributed under the Apache License, Version 2.0, which grants you broad freedom to use, modify, and integrate the software into your own applications or projects, provided that you comply with the conditions outlined in the license. Although Apache 2.0 does not require users to retain explicit author credit beyond standard copyright and license notices, I kindly request that if you incorporate this work into your own software, you acknowledge Sina Gilassi as the original author. Referencing the original repository or documentation is appreciated, as it helps recognize the effort invested in developing and maintaining this project.

## ❓ FAQ

For any question, contact me on [LinkedIn](https://www.linkedin.com/in/sina-gilassi/)

## 👨‍💻 Authors

- [@sinagilassi](https://www.github.com/sinagilassi)