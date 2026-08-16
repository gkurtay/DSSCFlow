from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PhotonConstants:
    """
    Physical constants used for photon-number conversion.
    """

    planck_j_s: float = 6.62607015e-34
    speed_of_light_m_s: float = 299792458.0
    nm_to_m: float = 1.0e-9


DEFAULT_CONSTANTS = PhotonConstants()


def validate_source_table(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and return a wavelength-sorted illumination table.

    Required columns
    ----------------
    wavelength_nm
    intensity
    """

    required = {
        "wavelength_nm",
        "intensity",
    }

    missing = required - set(source.columns)

    if missing:
        raise ValueError(
            f"Missing source columns: {sorted(missing)}"
        )

    out = source[
        ["wavelength_nm", "intensity"]
    ].copy()

    out["wavelength_nm"] = pd.to_numeric(
        out["wavelength_nm"],
        errors="raise",
    )

    out["intensity"] = pd.to_numeric(
        out["intensity"],
        errors="raise",
    )

    if not np.isfinite(
        out["wavelength_nm"].to_numpy()
    ).all():
        raise ValueError(
            "Source wavelengths must be finite"
        )

    if not np.isfinite(
        out["intensity"].to_numpy()
    ).all():
        raise ValueError(
            "Source intensities must be finite"
        )

    if (
        out["wavelength_nm"] <= 0
    ).any():
        raise ValueError(
            "Source wavelengths must be positive"
        )

    if (
        out["intensity"] < 0
    ).any():
        raise ValueError(
            "Source intensities cannot be negative"
        )

    out = (
        out
        .sort_values("wavelength_nm")
        .reset_index(drop=True)
    )

    if out["wavelength_nm"].duplicated().any():
        raise ValueError(
            "Source wavelength grid contains duplicates"
        )

    return out


def photon_number_spectrum(
    source: pd.DataFrame,
    *,
    absolute: bool = False,
    constants: PhotonConstants = DEFAULT_CONSTANTS,
) -> pd.DataFrame:
    """
    Convert a spectral-power distribution into photon-number weighting.

    For an absolute spectral-power distribution,

        Phi(lambda) = I(lambda) * lambda / (h c)

    If absolute=False, the common dimensional prefactor is omitted and
    relative photon weighting proportional to I(lambda)*lambda is returned.

    Relative weighting is sufficient for normalized descriptors such as PCF.
    """

    src = validate_source_table(source)

    wavelength_nm = src[
        "wavelength_nm"
    ].to_numpy(dtype=np.float64)

    intensity = src[
        "intensity"
    ].to_numpy(dtype=np.float64)

    if absolute:
        wavelength_m = (
            wavelength_nm
            * constants.nm_to_m
        )

        photon_weight = (
            intensity
            * wavelength_m
            / (
                constants.planck_j_s
                * constants.speed_of_light_m_s
            )
        )

        mode = "absolute"

    else:
        photon_weight = (
            intensity * wavelength_nm
        )

        mode = "relative"

    return pd.DataFrame({
        "wavelength_nm": wavelength_nm,
        "source_intensity": intensity,
        "photon_weight": photon_weight,
        "photon_mode": mode,
    })


def normalize_photon_distribution(
    photon_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize a photon-number spectrum to unit integrated area.
    """

    required = {
        "wavelength_nm",
        "photon_weight",
    }

    missing = required - set(
        photon_table.columns
    )

    if missing:
        raise ValueError(
            f"Missing photon columns: {sorted(missing)}"
        )

    out = photon_table.copy()

    wavelength = out[
        "wavelength_nm"
    ].to_numpy(dtype=np.float64)

    weight = out[
        "photon_weight"
    ].to_numpy(dtype=np.float64)

    area = float(
        np.trapezoid(
            weight,
            wavelength,
        )
    )

    if area <= 0:
        out["photon_probability_density"] = (
            np.zeros_like(weight)
        )
        return out

    out["photon_probability_density"] = (
        weight / area
    )

    return out
