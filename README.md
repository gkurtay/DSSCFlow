# DSSCFlow 1.0.0

DSSCFlow is an installable Python research-software package for **source-conditioned optical screening of molecular dye-sensitized solar-cell sensitizers**. It integrates TD-DFT transition data with illumination spectra to reconstruct molecular absorption, calculate photon-accessibility descriptors, quantify complete-factorial structure effects, test source/broadening robustness, and perform transparent Pareto decision analysis.

The first public release, DSSCFlow 1.0.0, provides two interfaces to the same numerical core:

- a command-line interface (CLI) for reproducible scripted analyses; and
- a local browser-based graphical interface (GUI) for interactive exploration, data upload, visualization and result export.

The package accompanies the DSSC16 study and includes a complete reproducibility example.

## Scientific scope

DSSCFlow does **not** replace Gaussian or Multiwfn. Electronic-structure calculations and state-resolved hole/electron, NTO, IFCT, and fragment-TDM analyses are performed externally. DSSCFlow consumes the resulting tables and performs:

- Gaussian broadening of TD-DFT transitions in energy space;
- Beer–Lambert absorptance reconstruction;
- illumination-to-photon-number conversion;
- photon capture fraction (PCF), centroid (PCC), breadth (PCB), and band attribution;
- 2^4 factorial contrasts;
- broadening/source robustness;
- three-objective and epsilon-Pareto screening using optical compatibility, S1 IFCT CT character, and internal reorganization energy.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install .
```

A prebuilt wheel is included in `dist/`:

```bash
python -m pip install dist/dsscflow-1.0.0-py3-none-any.whl
```

## Command-line interface

```bash
dsscflow --help
dsscflow version
```

Reconstruct spectra:

```bash
dsscflow spectra examples/DSSC16/data/DSSC16_STAGE04B_CAMB3LYP_EXCITED_STATES_LONG.csv \
  --sigma 0.30 \
  --out results/spectra_sigma030.csv
```

Evaluate source-conditioned photon accessibility:

```bash
dsscflow photon examples/DSSC16/data/DSSC16_STAGE04B_CAMB3LYP_EXCITED_STATES_LONG.csv \
  --sources examples/DSSC16/light_sources \
  --sigma 0.20 --sigma 0.30 --sigma 0.40 \
  --out results/photon_panel.csv
```

Reproduce the DSSC16 analysis:

```bash
dsscflow reproduce examples/DSSC16 --out results/DSSC16
```

The command exits non-zero if the publication regression checks fail.

## Browser GUI

Launch the local interface with either command:

```bash
dsscflow gui
```

or

```bash
dsscflow-gui
```

DSSCFlow starts a local web server (default `http://127.0.0.1:8765/`) and opens the interface in the default browser. The GUI is implemented entirely within the package and does not require a remote web service.

The interface provides:

- one-click loading of the bundled DSSC16 demonstration dataset;
- upload of a TD-DFT transition CSV and one or more illumination-source CSV files;
- optional upload of ground-state descriptor, IFCT, and fragment-TDM tables for Pareto analysis;
- adjustable Gaussian broadening, concentration, path length, and illumination source;
- normalized absorption curves with calculated lambda-max values;
- PCF/PCC/PCB rankings;
- factorial main effects;
- Pareto and epsilon-Pareto views when the mechanistic tables are available;
- source-robustness summaries; and
- download of the numerical analysis outputs as a ZIP archive.

The GUI invokes the same Python analysis functions used by the CLI. It is an interface layer, not an independent calculation implementation.

### Upload schemas

Transition CSVs require:

`system, ID, Donor, Aux, Bridge, Anchor, wavelength_nm, oscillator_strength`

Illumination CSVs require:

`wavelength_nm, intensity`

For the complete Pareto view, the optional files must contain the corresponding DSSCFlow descriptor/IFCT/fragment-TDM fields documented in the DSSC16 example.

## DSSC16 reference checks

The reproducibility workflow verifies that:

- D08 is the highest-PCF dye under each of the five production illumination spectra at sigma = 0.30 eV;
- the BTD main effect on PCF is positive under all five production sources; and
- D08, D14, and D16 are members of both the exact and epsilon Pareto fronts in all 15 source/broadening conditions.

The GUI demonstration mode exposes the same regression checks in its overview panel.

## Reproducibility data

`examples/DSSC16/` contains the TD-DFT transition table, ground-state descriptors, IFCT and fragment-TDM summaries, processed illumination spectra, and reference outputs used for regression testing. The same input tables required by the browser demonstration are also installed as package data so that the GUI works from the wheel without relying on the source-tree location.

## Tests

```bash
pytest -q
```

The v1.0.0 test suite includes numerical publication-regression tests, GUI-backend tests, and visible author-attribution checks. See `VALIDATION.md`.

## Citation

Citation metadata are provided in `CITATION.cff`. A repository/release DOI can be added after archiving the release.


## Author and contact

**Gülbin Kurtay**  
Hacettepe University, Department of Chemistry  
Email: gulbinkurtay@hacettepe.edu.tr  
ORCID: https://orcid.org/0000-0003-0920-8409
