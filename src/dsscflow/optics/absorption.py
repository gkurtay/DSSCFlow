from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpticalGrid:
    lower_nm: float = 200.0
    upper_nm: float = 800.0
    points: int = 10000

    def wavelengths(self) -> np.ndarray:
        if self.lower_nm <= 0:
            raise ValueError("lower_nm must be positive")

        if self.upper_nm <= self.lower_nm:
            raise ValueError(
                "upper_nm must exceed lower_nm"
            )

        if self.points < 50:
            raise ValueError(
                "points must be >= 50"
            )

        return np.linspace(
            self.lower_nm,
            self.upper_nm,
            self.points,
            dtype=np.float64,
        )


@dataclass(frozen=True)
class GaussianBroadening:
    sigma_ev: float = 0.30
    ev_to_cm1: float = 8065.544

    @property
    def sigma_cm1(self) -> float:
        if self.sigma_ev <= 0:
            raise ValueError(
                "sigma_ev must be positive"
            )

        return (
            self.sigma_ev
            * self.ev_to_cm1
        )


@dataclass(frozen=True)
class AbsorptionScale:
    integrated_coefficient: float = 2.315e8

    def validate(self) -> None:
        if self.integrated_coefficient <= 0:
            raise ValueError(
                "integrated coefficient must be positive"
            )


@dataclass(frozen=True)
class OpticalPath:
    concentration_molar: float = 1.0e-5
    path_length_cm: float = 1.0

    def validate(self) -> None:
        if self.concentration_molar < 0:
            raise ValueError(
                "concentration cannot be negative"
            )

        if self.path_length_cm < 0:
            raise ValueError(
                "path length cannot be negative"
            )

def validate_transition_table(
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate TD-DFT transition data.

    Required columns:
        wavelength_nm
        oscillator_strength
    """

    required = {
        "wavelength_nm",
        "oscillator_strength",
    }

    missing = required - set(
        transitions.columns
    )

    if missing:
        raise ValueError(
            f"Missing transition columns: {sorted(missing)}"
        )

    out = transitions.copy()

    out["wavelength_nm"] = pd.to_numeric(
        out["wavelength_nm"],
        errors="raise",
    )

    out["oscillator_strength"] = pd.to_numeric(
        out["oscillator_strength"],
        errors="raise",
    )

    wavelength = out[
        "wavelength_nm"
    ].to_numpy(dtype=np.float64)

    oscillator = out[
        "oscillator_strength"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(wavelength).all():
        raise ValueError(
            "Transition wavelengths must be finite"
        )

    if not np.isfinite(oscillator).all():
        raise ValueError(
            "Oscillator strengths must be finite"
        )

    if np.any(wavelength <= 0):
        raise ValueError(
            "Transition wavelengths must be positive"
        )

    if np.any(oscillator < 0):
        raise ValueError(
            "Oscillator strengths cannot be negative"
        )

    return out


def gaussian_kernel_matrix(
    transition_wavelength_nm: np.ndarray,
    grid_wavelength_nm: np.ndarray,
    *,
    broadening: GaussianBroadening,
) -> np.ndarray:
    """
    Construct the complete transition x spectral-grid Gaussian matrix.

    No Python loop over individual transitions is used.
    """

    centres_cm1 = (
        1.0e7
        / np.asarray(
            transition_wavelength_nm,
            dtype=np.float64,
        )
    )

    grid_cm1 = (
        1.0e7
        / np.asarray(
            grid_wavelength_nm,
            dtype=np.float64,
        )
    )

    sigma = broadening.sigma_cm1

    displacement = (
        grid_cm1[None, :]
        - centres_cm1[:, None]
    )

    normalization = (
        sigma
        * np.sqrt(
            2.0 * np.pi
        )
    )

    return (
        np.exp(
            -0.5
            * (
                displacement
                / sigma
            ) ** 2
        )
        / normalization
    )

def reconstruct_molar_absorptivity(
    transitions: pd.DataFrame,
    *,
    grid: OpticalGrid = OpticalGrid(),
    broadening: GaussianBroadening = GaussianBroadening(),
    scale: AbsorptionScale = AbsorptionScale(),
) -> pd.DataFrame:
    """
    Reconstruct a Gaussian-broadened molar-absorptivity profile.
    """

    scale.validate()

    td = validate_transition_table(
        transitions
    )

    wavelength_nm = (
        grid.wavelengths()
    )

    if len(td) == 0:
        epsilon = np.zeros_like(
            wavelength_nm
        )

    else:
        centres = td[
            "wavelength_nm"
        ].to_numpy(dtype=np.float64)

        strengths = td[
            "oscillator_strength"
        ].to_numpy(dtype=np.float64)

        kernel = gaussian_kernel_matrix(
            centres,
            wavelength_nm,
            broadening=broadening,
        )

        epsilon = (
            scale.integrated_coefficient
            * np.einsum(
                "i,ij->j",
                strengths,
                kernel,
                optimize=True,
            )
        )

    return pd.DataFrame({
        "wavelength_nm":
            wavelength_nm,

        "molar_absorptivity":
            epsilon,
    })


def beer_lambert_absorptance(
    spectrum: pd.DataFrame,
    *,
    optical_path: OpticalPath = OpticalPath(),
) -> pd.DataFrame:
    """
    Convert molar absorptivity to absorbance and absorptance.
    """

    optical_path.validate()

    required = {
        "wavelength_nm",
        "molar_absorptivity",
    }

    missing = required - set(
        spectrum.columns
    )

    if missing:
        raise ValueError(
            f"Missing spectrum columns: {sorted(missing)}"
        )

    wavelength_nm = spectrum[
        "wavelength_nm"
    ].to_numpy(dtype=np.float64)

    epsilon = spectrum[
        "molar_absorptivity"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(epsilon).all():
        raise ValueError(
            "Molar absorptivity must be finite"
        )

    if np.any(epsilon < 0):
        raise ValueError(
            "Molar absorptivity cannot be negative"
        )

    absorbance = (
        epsilon
        * optical_path.concentration_molar
        * optical_path.path_length_cm
    )

    # Numerically stable form of 1 - 10^(-A)
    absorptance = (
        -np.expm1(
            -np.log(10.0)
            * absorbance
        )
    )

    absorptance = np.clip(
        absorptance,
        0.0,
        1.0,
    )

    return pd.DataFrame({
        "wavelength_nm":
            wavelength_nm,

        "molar_absorptivity":
            epsilon,

        "absorbance":
            absorbance,

        "absorptance":
            absorptance,
    })


def td_to_absorptance(
    transitions: pd.DataFrame,
    *,
    grid: OpticalGrid = OpticalGrid(),
    broadening: GaussianBroadening = GaussianBroadening(),
    scale: AbsorptionScale = AbsorptionScale(),
    optical_path: OpticalPath = OpticalPath(),
) -> pd.DataFrame:

    spectrum = (
        reconstruct_molar_absorptivity(
            transitions,
            grid=grid,
            broadening=broadening,
            scale=scale,
        )
    )

    return beer_lambert_absorptance(
        spectrum,
        optical_path=optical_path,
    )
