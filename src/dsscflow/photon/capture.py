from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CaptureWindow:
    lower_nm: float = 380.0
    upper_nm: float = 780.0

    def validate(self) -> None:
        if self.lower_nm <= 0:
            raise ValueError("lower_nm must be positive")
        if self.upper_nm <= self.lower_nm:
            raise ValueError("upper_nm must exceed lower_nm")


@dataclass(frozen=True)
class CaptureBand:
    name: str
    lower_nm: float
    upper_nm: float


DEFAULT_VISIBLE_BANDS = (
    CaptureBand("B1_380_500", 380.0, 500.0),
    CaptureBand("B2_500_600", 500.0, 600.0),
    CaptureBand("B3_600_700", 600.0, 700.0),
    CaptureBand("B4_700_780", 700.0, 780.0),
)

def _validated_table(
    table: pd.DataFrame,
    value_column: str,
    *,
    bounded_unit_interval: bool = False,
) -> pd.DataFrame:
    required = {"wavelength_nm", value_column}
    missing = required - set(table.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    out = table[
        ["wavelength_nm", value_column]
    ].copy()

    out["wavelength_nm"] = pd.to_numeric(
        out["wavelength_nm"],
        errors="raise",
    )
    out[value_column] = pd.to_numeric(
        out[value_column],
        errors="raise",
    )

    wavelength = out[
        "wavelength_nm"
    ].to_numpy(dtype=np.float64)

    values = out[
        value_column
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(wavelength).all():
        raise ValueError(
            "Wavelength values must be finite"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{value_column} values must be finite"
        )

    if np.any(wavelength <= 0):
        raise ValueError(
            "Wavelength values must be positive"
        )

    if np.any(values < 0):
        raise ValueError(
            f"{value_column} values cannot be negative"
        )

    if (
        bounded_unit_interval
        and np.any(values > 1)
    ):
        raise ValueError(
            f"{value_column} values must lie in [0, 1]"
        )

    out = (
        out
        .sort_values("wavelength_nm")
        .reset_index(drop=True)
    )

    if out["wavelength_nm"].duplicated().any():
        raise ValueError(
            "Wavelength grid contains duplicates"
        )

    return out


def _windowed_arrays(
    wavelength_nm: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    lower_nm: float,
    upper_nm: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:

    if lower_nm < float(wavelength_nm.min()):
        raise ValueError(
            "Capture window extends below available wavelength domain"
        )

    if upper_nm > float(wavelength_nm.max()):
        raise ValueError(
            "Capture window extends above available wavelength domain"
        )

    inside = (
        (wavelength_nm > lower_nm)
        & (wavelength_nm < upper_nm)
    )

    x = np.concatenate(
        ([lower_nm], wavelength_nm[inside], [upper_nm])
    )

    outputs = []

    for values in arrays:
        y = np.concatenate((
            [np.interp(lower_nm, wavelength_nm, values)],
            values[inside],
            [np.interp(upper_nm, wavelength_nm, values)],
        ))
        outputs.append(y)

    return x, tuple(outputs)


def capturable_photon_distribution(
    absorptance: pd.DataFrame,
    photons: pd.DataFrame,
    *,
    window: CaptureWindow = CaptureWindow(),
) -> pd.DataFrame:
    """
    Construct the source-conditioned capturable-photon distribution

        C(lambda) = alpha(lambda) * Phi(lambda)

    using the photon wavelength grid as the integration grid.
    """

    window.validate()

    alpha_table = _validated_table(
        absorptance,
        "absorptance",
        bounded_unit_interval=True,
    )

    photon_table = _validated_table(
        photons,
        "photon_weight",
    )

    photon_nm = photon_table[
        "wavelength_nm"
    ].to_numpy(dtype=np.float64)

    photon_weight = photon_table[
        "photon_weight"
    ].to_numpy(dtype=np.float64)

    alpha_on_photon_grid = np.interp(
        photon_nm,
        alpha_table[
            "wavelength_nm"
        ].to_numpy(dtype=np.float64),
        alpha_table[
            "absorptance"
        ].to_numpy(dtype=np.float64),
        left=0.0,
        right=0.0,
    )

    wavelength_nm, (
        alpha_window,
        photon_window,
    ) = _windowed_arrays(
        photon_nm,
        (
            alpha_on_photon_grid,
            photon_weight,
        ),
        window.lower_nm,
        window.upper_nm,
    )

    captured = (
        alpha_window
        * photon_window
    )

    return pd.DataFrame({
        "wavelength_nm": wavelength_nm,
        "absorptance": alpha_window,
        "photon_weight": photon_window,
        "captured_photon_weight": captured,
    })


def photon_capture_fraction(
    capture: pd.DataFrame,
) -> float:
    """
    Photon Capture Fraction (PCF).

    Returns the fraction of source photons in the analysis window
    weighted by molecular absorptance.
    """

    wavelength = capture[
        "wavelength_nm"
    ].to_numpy(dtype=np.float64)

    photon = capture[
        "photon_weight"
    ].to_numpy(dtype=np.float64)

    captured = capture[
        "captured_photon_weight"
    ].to_numpy(dtype=np.float64)

    available_area = float(
        np.trapezoid(
            photon,
            wavelength,
        )
    )

    if available_area <= 0.0:
        return float("nan")

    captured_area = float(
        np.trapezoid(
            captured,
            wavelength,
        )
    )

    value = (
        captured_area
        / available_area
    )

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def photon_capture_centroid(
    capture: pd.DataFrame,
) -> float:
    """
    Photon Capture Centroid (PCC), in nm.

    Returns NaN when no photons are captured.
    """

    wavelength = capture[
        "wavelength_nm"
    ].to_numpy(dtype=np.float64)

    captured = capture[
        "captured_photon_weight"
    ].to_numpy(dtype=np.float64)

    captured_area = float(
        np.trapezoid(
            captured,
            wavelength,
        )
    )

    if captured_area <= 0.0:
        return float("nan")

    first_moment = float(
        np.trapezoid(
            wavelength
            * captured,
            wavelength,
        )
    )

    return (
        first_moment
        / captured_area
    )


def photon_capture_breadth(
    capture: pd.DataFrame,
    *,
    centroid_nm: float | None = None,
) -> float:
    """
    Photon Capture Breadth (PCB), in nm.

    PCB is the square root of the second central moment of the
    capturable-photon distribution.
    """

    wavelength = capture[
        "wavelength_nm"
    ].to_numpy(dtype=np.float64)

    captured = capture[
        "captured_photon_weight"
    ].to_numpy(dtype=np.float64)

    captured_area = float(
        np.trapezoid(
            captured,
            wavelength,
        )
    )

    if captured_area <= 0.0:
        return float("nan")

    if centroid_nm is None:
        centroid_nm = (
            photon_capture_centroid(
                capture
            )
        )

    second_central_moment = float(
        np.trapezoid(
            (
                wavelength
                - centroid_nm
            ) ** 2
            * captured,
            wavelength,
        )
    )

    variance = max(
        second_central_moment
        / captured_area,
        0.0,
    )

    return float(
        np.sqrt(variance)
    )

def _integrate_interval(
    wavelength: np.ndarray,
    values: np.ndarray,
    lower_nm: float,
    upper_nm: float,
) -> float:

    x, (y,) = _windowed_arrays(
        wavelength,
        (values,),
        lower_nm,
        upper_nm,
    )

    return float(
        np.trapezoid(y, x)
    )


def band_capture_decomposition(
    capture: pd.DataFrame,
    *,
    bands: tuple[
        CaptureBand, ...
    ] = DEFAULT_VISIBLE_BANDS,
) -> pd.DataFrame:

    wavelength = capture[
        "wavelength_nm"
    ].to_numpy(dtype=np.float64)

    captured = capture[
        "captured_photon_weight"
    ].to_numpy(dtype=np.float64)

    rows = []

    for band in bands:

        if band.upper_nm <= band.lower_nm:
            raise ValueError(
                f"Invalid band: {band.name}"
            )

        if (
            band.lower_nm < wavelength.min()
            or band.upper_nm > wavelength.max()
        ):
            raise ValueError(
                f"Band {band.name} lies outside capture window"
            )

        area = _integrate_interval(
            wavelength,
            captured,
            band.lower_nm,
            band.upper_nm,
        )

        rows.append({
            "band": band.name,
            "lower_nm": band.lower_nm,
            "upper_nm": band.upper_nm,
            "captured_area": area,
        })

    out = pd.DataFrame(rows)

    total = float(
        out["captured_area"].sum()
    )

    if total > 0:
        out["captured_fraction"] = (
            out["captured_area"]
            / total
        )
    else:
        out["captured_fraction"] = np.nan

    return out


def summarize_photon_capture(
    absorptance: pd.DataFrame,
    photons: pd.DataFrame,
    *,
    window: CaptureWindow = CaptureWindow(),
    bands: tuple[
        CaptureBand, ...
    ] = DEFAULT_VISIBLE_BANDS,
) -> tuple[
    dict[str, float],
    pd.DataFrame,
    pd.DataFrame,
]:

    capture = (
        capturable_photon_distribution(
            absorptance,
            photons,
            window=window,
        )
    )

    pcf = photon_capture_fraction(capture)
    pcc = photon_capture_centroid(capture)

    pcb = photon_capture_breadth(
        capture,
        centroid_nm=pcc,
    )

    band_table = (
        band_capture_decomposition(
            capture,
            bands=bands,
        )
    )

    summary = {
        "PCF": pcf,
        "PCC_nm": pcc,
        "PCB_nm": pcb,
    }

    for _, row in band_table.iterrows():
        value = row["captured_fraction"]

        summary[
            f"{row['band']}_fraction"
        ] = (
            float(value)
            if pd.notna(value)
            else float("nan")
        )

    return summary, capture, band_table
