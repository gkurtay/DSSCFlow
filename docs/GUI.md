# DSSCFlow browser interface

## Launch

```bash
dsscflow gui
```

Optional launch controls:

```bash
dsscflow gui --port 8765 --address 127.0.0.1 --no-browser
```

The application runs as a local `ThreadingHTTPServer`. Static HTML/CSS/JavaScript files are served from the installed Python package, while numerical analyses execute in the DSSCFlow Python process.

## Data flow

1. The browser reads user-selected CSV files locally.
2. CSV text is posted to the local DSSCFlow server at `/api/analyze`.
3. The server validates the input schemas and invokes the same `dsscflow.workflows` and decision-analysis functions used by the command-line interface.
4. Compact JSON summaries are returned for visualization.
5. The complete numerical outputs can be exported from `/api/export` as a ZIP archive.

No external web API is required for analysis.

## GUI panels

- **Overview:** dataset dimensions, top ranking, DSSC16 regression status in demonstration mode.
- **Spectra:** normalized Gaussian-broadened molecular absorption and calculated envelope maxima.
- **Photon accessibility:** PCF/PCC/PCB ranking for the selected source and broadening.
- **Factorial effects:** main 2^4 structural effects on PCF.
- **Pareto:** PCF vs IFCT CT view and Pareto membership when descriptor, IFCT and TDM tables are supplied.
- **Robustness:** source-conditioned PCF statistics and ranks.

## Local-only design

The browser interface binds to `127.0.0.1` by default. Uploaded inputs remain between the browser and the local Python server. Binding to a non-loopback address is an explicit user action and should only be done in a trusted environment.


## Author

DSSCFlow is developed by **Gülbin Kurtay**, Hacettepe University, Department of Chemistry.  
Contact: gulbinkurtay@hacettepe.edu.tr  
ORCID: https://orcid.org/0000-0003-0920-8409
