from __future__ import annotations

import numpy as np
import pandas as pd

from dsscflow.photon.capture import (
    CaptureWindow,
    DEFAULT_VISIBLE_BANDS,
)


def band_pcf_contributions(
    absorptance: pd.DataFrame,
    photon_table: pd.DataFrame,
    *,
    window: CaptureWindow = CaptureWindow(),
    bands=DEFAULT_VISIBLE_BANDS,
) -> pd.DataFrame:
    """
    Decompose total PCF into absolute band contributions.

    band_pcf_contribution:
        Band-captured photon area divided by the TOTAL
        source-photon area over the analysis window.
        Therefore band contributions sum to PCF.

    band_capture_fraction:
        Band-captured photon area divided by TOTAL
        captured-photon area.
        Therefore fractions sum to 1 for nonzero capture.
    """

    for frame, required, label in [
        (
            absorptance,
            {"wavelength_nm", "absorptance"},
            "absorptance",
        ),
        (
            photon_table,
            {"wavelength_nm", "photon_weight"},
            "photon",
        ),
    ]:
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"Missing {label} columns: "
                f"{sorted(missing)}"
            )

    ax = absorptance[
        "wavelength_nm"
    ].to_numpy(dtype=float)

    ay = absorptance[
        "absorptance"
    ].to_numpy(dtype=float)

    px = photon_table[
        "wavelength_nm"
    ].to_numpy(dtype=float)

    py = photon_table[
        "photon_weight"
    ].to_numpy(dtype=float)

    if not (
        np.isfinite(ax).all()
        and np.isfinite(ay).all()
        and np.isfinite(px).all()
        and np.isfinite(py).all()
    ):
        raise ValueError(
            "Band inputs must be finite"
        )

    if np.any(
        (ay < 0.0)
        | (ay > 1.0)
    ):
        raise ValueError(
            "Absorptance must lie in [0, 1]"
        )

    if np.any(py < 0.0):
        raise ValueError(
            "Photon weights cannot be negative"
        )

    edges = [
        window.lower_nm,
        window.upper_nm,
    ]

    for band in bands:
        edges.extend(
            [
                band.lower_nm,
                band.upper_nm,
            ]
        )

    interior_a = ax[
        (ax > window.lower_nm)
        & (ax < window.upper_nm)
    ]

    interior_p = px[
        (px > window.lower_nm)
        & (px < window.upper_nm)
    ]

    x = np.unique(
        np.concatenate(
            [
                interior_a,
                interior_p,
                np.asarray(
                    edges,
                    dtype=float,
                ),
            ]
        )
    )

    x = x[
        (x >= window.lower_nm)
        & (x <= window.upper_nm)
    ]

    alpha = np.interp(
        x,
        ax,
        ay,
        left=0.0,
        right=0.0,
    )

    phi = np.interp(
        x,
        px,
        py,
        left=0.0,
        right=0.0,
    )

    captured = (
        alpha
        * phi
    )

    photon_area = float(
        np.trapezoid(
            phi,
            x,
        )
    )

    captured_area = float(
        np.trapezoid(
            captured,
            x,
        )
    )

    pcf = (
        captured_area
        / photon_area
        if photon_area > 0.0
        else np.nan
    )

    rows = []

    for band in bands:
        mask = (
            (x >= band.lower_nm)
            & (x <= band.upper_nm)
        )

        xb = x[mask]
        cb = captured[mask]

        area = float(
            np.trapezoid(
                cb,
                xb,
            )
        )

        rows.append({
            "band": band.name,
            "lower_nm": band.lower_nm,
            "upper_nm": band.upper_nm,
            "captured_photon_area": area,
            "band_pcf_contribution": (
                area / photon_area
                if photon_area > 0.0
                else np.nan
            ),
            "band_capture_fraction": (
                area / captured_area
                if captured_area > 0.0
                else np.nan
            ),
            "total_PCF": pcf,
        })

    result = pd.DataFrame(rows)

    if np.isfinite(pcf):
        reconstructed = float(
            result[
                "band_pcf_contribution"
            ].sum()
        )

        if not np.isclose(
            reconstructed,
            pcf,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Band PCF contributions do not "
                "reconstruct total PCF"
            )

    return result
