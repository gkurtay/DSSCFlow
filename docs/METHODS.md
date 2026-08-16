# Computational definition used by DSSCFlow

DSSCFlow reconstructs molecular absorption from discrete TD-DFT transitions using Gaussian broadening in wavenumber/energy space. The integrated oscillator-strength relation is used to obtain molar absorptivity. Absorptance is then calculated by the Beer–Lambert relation using the configured concentration and optical path length.

Illumination spectra are converted to relative photon-number weighting proportional to `I(lambda) * lambda`. Within the analysis window, the capturable-photon distribution is the product of molecular absorptance and source photon weighting. PCF is the captured-photon integral divided by the source-photon integral. PCC and PCB are the first moment and square root of the second central moment of the capturable-photon distribution.

Factorial effects are descriptive orthogonal contrasts of the complete 2^4 design. No inferential p-values are assigned. The core Pareto analysis maximizes optical compatibility and S1 IFCT CT character while minimizing total internal reorganization energy. Fragment-TDM and anchor-electron descriptors are treated as mechanistic annotations rather than mandatory Pareto objectives.
