from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from dsscflow.optics.absorption import (
    GaussianBroadening,
    OpticalGrid,
    OpticalPath,
    td_to_absorptance,
)
from dsscflow.photon.capture import (
    CaptureWindow,
    DEFAULT_VISIBLE_BANDS,
)
from dsscflow.photon.factorial import (
    FACTOR_CODING,
    FACTOR_ORDER,
    validate_full_factorial,
)
from dsscflow.photon.source import (
    photon_number_spectrum,
)


def build_absorptance_matrix(
    excited_states: pd.DataFrame,
    *,
    sigma_ev: float = 0.30,
    grid: OpticalGrid = OpticalGrid(),
    optical_path: OpticalPath = OpticalPath(),
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Return:
      design table,
      common wavelength grid,
      absorptance matrix [system x wavelength].
    """

    metadata = (
        excited_states[
            [
                "system",
                "ID",
                "Donor",
                "Aux",
                "Bridge",
                "Anchor",
            ]
        ]
        .drop_duplicates("system")
        .sort_values("system")
        .reset_index(drop=True)
    )

    validate_full_factorial(metadata)

    wavelength = None
    rows = []

    for system in metadata["system"]:
        td = excited_states[
            excited_states["system"] == system
        ]

        optical = td_to_absorptance(
            td,
            grid=grid,
            broadening=GaussianBroadening(
                sigma_ev=sigma_ev
            ),
            optical_path=optical_path,
        )

        x = optical[
            "wavelength_nm"
        ].to_numpy(dtype=np.float64)

        alpha = optical[
            "absorptance"
        ].to_numpy(dtype=np.float64)

        if wavelength is None:
            wavelength = x
        elif not np.array_equal(
            wavelength,
            x,
        ):
            raise RuntimeError(
                "Optical grids are inconsistent"
            )

        rows.append(alpha)

    return (
        metadata,
        wavelength,
        np.vstack(rows),
    )


def wavelength_factor_effects(
    design: pd.DataFrame,
    wavelength_nm: np.ndarray,
    absorptance_matrix: np.ndarray,
    *,
    max_order: int = 2,
) -> pd.DataFrame:
    """
    Compute wavelength-resolved factorial effects on absorptance.
    """

    validate_full_factorial(design)

    if absorptance_matrix.shape != (
        len(design),
        len(wavelength_nm),
    ):
        raise ValueError(
            "Absorptance matrix shape mismatch"
        )

    frames = []

    for order in range(
        1,
        max_order + 1,
    ):
        for term in itertools.combinations(
            FACTOR_ORDER,
            order,
        ):
            coded = np.ones(
                len(design),
                dtype=np.float64,
            )

            for factor in term:
                mapped = design[
                    factor
                ].map(
                    FACTOR_CODING[factor]
                )

                if mapped.isna().any():
                    raise ValueError(
                        f"Unknown level for {factor}"
                    )

                coded *= mapped.to_numpy(
                    dtype=np.float64
                )

            effect = (
                2.0
                * np.mean(
                    coded[:, None]
                    * absorptance_matrix,
                    axis=0,
                )
            )

            frames.append(
                pd.DataFrame({
                    "order": order,
                    "term": "×".join(term),
                    "wavelength_nm":
                        wavelength_nm,
                    "delta_absorptance":
                        effect,
                })
            )

    return pd.concat(
        frames,
        ignore_index=True,
    )



def _restrict_with_boundaries(
    wavelength_nm: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    lower_nm: float,
    upper_nm: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """
    Restrict arrays to an exact wavelength interval while inserting
    linearly interpolated values at both requested boundaries.
    """

    if lower_nm < float(wavelength_nm.min()):
        raise ValueError("lower boundary lies outside wavelength domain")

    if upper_nm > float(wavelength_nm.max()):
        raise ValueError("upper boundary lies outside wavelength domain")

    if upper_nm <= lower_nm:
        raise ValueError("upper boundary must exceed lower boundary")

    interior = (
        (wavelength_nm > lower_nm)
        & (wavelength_nm < upper_nm)
    )

    x = np.concatenate((
        [lower_nm],
        wavelength_nm[interior],
        [upper_nm],
    ))

    outputs = []

    for values in arrays:
        y = np.concatenate((
            [np.interp(lower_nm, wavelength_nm, values)],
            values[interior],
            [np.interp(upper_nm, wavelength_nm, values)],
        ))

        outputs.append(y)

    return x, tuple(outputs)

def source_weighted_factor_attribution(
    wavelength_effects: pd.DataFrame,
    illumination_sources: dict[str, pd.DataFrame],
    *,
    window: CaptureWindow = CaptureWindow(),
) -> pd.DataFrame:
    """
    Weight wavelength-resolved absorptance effects by source photon number.

    The returned contribution_density integrates to the
    source-specific PCF factorial effect.
    """

    rows = []

    for source_name in sorted(
        illumination_sources
    ):
        photons = photon_number_spectrum(
            illumination_sources[source_name],
            absolute=False,
        )

        source_nm = photons[
            "wavelength_nm"
        ].to_numpy(dtype=np.float64)

        source_weight = photons[
            "photon_weight"
        ].to_numpy(dtype=np.float64)

        x, (phi,) = _restrict_with_boundaries(
            source_nm,
            (source_weight,),
            window.lower_nm,
            window.upper_nm,
        )

        denominator = float(
            np.trapezoid(
                phi,
                x,
            )
        )

        if denominator <= 0:
            raise ValueError(
                f"Zero photon area for {source_name}"
            )

        for (
            order,
            term,
        ), sub in wavelength_effects.groupby(
            ["order", "term"],
            sort=True,
        ):
            delta_alpha = np.interp(
                x,
                sub["wavelength_nm"].to_numpy(
                    dtype=np.float64
                ),
                sub["delta_absorptance"].to_numpy(
                    dtype=np.float64
                ),
                left=0.0,
                right=0.0,
            )

            density = (
                delta_alpha
                * phi
                / denominator
            )

            rows.append(
                pd.DataFrame({
                    "light_source":
                        source_name,
                    "order":
                        order,
                    "term":
                        term,
                    "wavelength_nm":
                        x,
                    "delta_absorptance":
                        delta_alpha,
                    "photon_weight":
                        phi,
                    "pcf_contribution_density":
                        density,
                })
            )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def summarize_factor_attribution(
    attribution: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce integrated source-term effects and band-level attribution.

    positive_gain:
        integral of positive contribution density

    negative_loss:
        magnitude of negative contribution density

    net_effect:
        positive_gain - negative_loss
    """

    summary_rows = []
    band_rows = []

    for (
        source,
        order,
        term,
    ), sub in attribution.groupby(
        [
            "light_source",
            "order",
            "term",
        ],
        sort=True,
    ):
        x = sub[
            "wavelength_nm"
        ].to_numpy(dtype=np.float64)

        density = sub[
            "pcf_contribution_density"
        ].to_numpy(dtype=np.float64)

        positive = np.clip(
            density,
            0.0,
            None,
        )

        negative = np.clip(
            -density,
            0.0,
            None,
        )

        net = float(
            np.trapezoid(
                density,
                x,
            )
        )

        positive_total = float(
            np.trapezoid(
                positive,
                x,
            )
        )

        negative_total = float(
            np.trapezoid(
                negative,
                x,
            )
        )

        summary_rows.append({
            "light_source": source,
            "order": order,
            "term": term,
            "net_effect": net,
            "positive_gain": positive_total,
            "negative_loss": negative_total,
        })

        for band in DEFAULT_VISIBLE_BANDS:
            xb, (
                band_density,
                band_positive_density,
                band_negative_density,
            ) = _restrict_with_boundaries(
                x,
                (
                    density,
                    positive,
                    negative,
                ),
                band.lower_nm,
                band.upper_nm,
            )

            band_net = float(
                np.trapezoid(
                    band_density,
                    xb,
                )
            )

            band_positive = float(
                np.trapezoid(
                    band_positive_density,
                    xb,
                )
            )

            band_negative = float(
                np.trapezoid(
                    band_negative_density,
                    xb,
                )
            )

            band_rows.append({
                "light_source":
                    source,
                "order":
                    order,
                "term":
                    term,
                "band":
                    band.name,
                "band_net_effect":
                    band_net,
                "band_positive_gain":
                    band_positive,
                "band_negative_loss":
                    band_negative,
                "positive_gain_fraction":
                    (
                        band_positive
                        / positive_total
                        if positive_total > 0
                        else np.nan
                    ),
                "negative_loss_fraction":
                    (
                        band_negative
                        / negative_total
                        if negative_total > 0
                        else np.nan
                    ),
            })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(band_rows),
    )
