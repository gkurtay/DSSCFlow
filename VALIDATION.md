# DSSCFlow v1.0.0 validation

Validation date: 2026-08-16

The first public release candidate was tested against the DSSC16 reference data. The command-line and local browser interfaces call the same workflow functions and therefore share one numerical implementation.

## Automated tests

```bash
pytest -q
```

Result: **6 passed**.

The suite verifies:

- non-negative, finite Gaussian-broadened absorption profiles;
- conservation of the four photon-attribution band fractions;
- D08 as the highest-PCF sensitizer under all five production illumination sources at sigma = 0.30 eV;
- positive BTD main effect on PCF under all five production sources;
- exact and epsilon Pareto membership of D08, D14 and D16 in all 15 source/broadening combinations;
- agreement of the five D08 PCF values with the manuscript reference values within 5e-7;
- loading of the installed DSSC16 GUI demonstration data;
- GUI-backend analysis of all 480 transitions and the expected D08 ranking;
- GUI exposure of the same DSSC16 publication-regression checks;
- generation of a valid numerical result ZIP archive; and
- selection of a bindable local GUI port.

## Local GUI server smoke test

A local v1.0.0 server was launched on the loopback interface. The following checks passed:

- `/api/health` returned status `ok` and version `1.0.0`;
- `/` returned the packaged HTML interface;
- `/api/analyze` completed the DSSC16 demonstration analysis with 16 dyes and 480 transitions;
- the returned top-ranked dye under AM1.5G at sigma = 0.30 eV was D08;
- the GUI regression report returned the expected robust Pareto set D08/D14/D16; and
- `/api/export` returned a valid ZIP archive containing the analysis tables.

## Interface consistency

The GUI does not contain an independent implementation of the photon or decision equations. It calls the same `reconstruct_spectra`, `evaluate_widths`, factorial, robustness and Pareto functions used by the CLI. The default Beer–Lambert concentration (1e-5 M), path length (1 cm), broadening widths (0.20/0.30/0.40 eV), and DSSC16 regression definitions remain unchanged.
