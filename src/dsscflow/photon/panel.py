from __future__ import annotations

import numpy as np
import pandas as pd

from dsscflow.optics.absorption import (
    GaussianBroadening,
    OpticalGrid,
    OpticalPath,
    td_to_absorptance,
)
from dsscflow.photon.source import (
    photon_number_spectrum,
)
from dsscflow.photon.capture import (
    CaptureWindow,
    summarize_photon_capture,
)


REQUIRED_METADATA = (
    "system",
    "ID",
    "Donor",
    "Aux",
    "Bridge",
    "Anchor",
)


def evaluate_photon_panel(
    excited_states: pd.DataFrame,
    illumination_sources: dict[str, pd.DataFrame],
    *,
    sigma_ev: float = 0.30,
    optical_grid: OpticalGrid = OpticalGrid(),
    optical_path: OpticalPath = OpticalPath(),
    capture_window: CaptureWindow = CaptureWindow(),
) -> pd.DataFrame:
    """
    Evaluate source-conditioned photon-harvesting descriptors.

    One output row is produced for every sensitizer-source pair.
    """

    missing = (
        set(REQUIRED_METADATA)
        - set(excited_states.columns)
    )

    if missing:
        raise ValueError(
            f"Missing excited-state metadata: {sorted(missing)}"
        )

    if not illumination_sources:
        raise ValueError(
            "At least one illumination source is required"
        )

    broadening = GaussianBroadening(
        sigma_ev=sigma_ev
    )

    rows = []

    for system, td in excited_states.groupby(
        "system",
        sort=True,
    ):
        meta = (
            td.iloc[0][
                list(REQUIRED_METADATA)
            ]
            .to_dict()
        )

        absorption = td_to_absorptance(
            td,
            grid=optical_grid,
            broadening=broadening,
            optical_path=optical_path,
        )

        absorptance_table = absorption[
            [
                "wavelength_nm",
                "absorptance",
            ]
        ].copy()

        for source_name in sorted(
            illumination_sources
        ):
            photon_table = photon_number_spectrum(
                illumination_sources[source_name],
                absolute=False,
            )

            summary, _, _ = summarize_photon_capture(
                absorptance_table,
                photon_table,
                window=capture_window,
            )

            rows.append({
                **meta,
                "light_source": source_name,
                "sigma_ev": sigma_ev,
                "PCF": summary["PCF"],
                "PCC_nm": summary["PCC_nm"],
                "PCB_nm": summary["PCB_nm"],
                "B1_380_500_fraction":
                    summary["B1_380_500_fraction"],
                "B2_500_600_fraction":
                    summary["B2_500_600_fraction"],
                "B3_600_700_fraction":
                    summary["B3_600_700_fraction"],
                "B4_700_780_fraction":
                    summary["B4_700_780_fraction"],
            })

    return pd.DataFrame(rows)


def summarize_source_robustness(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize sensitizer PCF robustness across illumination sources.
    """

    required = {
        "system",
        "ID",
        "sigma_ev",
        "light_source",
        "PCF",
    }

    missing = required - set(panel.columns)

    if missing:
        raise ValueError(
            f"Missing panel columns: {sorted(missing)}"
        )

    ranked = panel.copy()

    ranked["source_rank"] = (
        ranked
        .groupby(
            [
                "sigma_ev",
                "light_source",
            ]
        )["PCF"]
        .rank(
            method="min",
            ascending=False,
        )
    )

    rows = []

    for (
        sigma_ev,
        system,
        ID,
    ), sub in ranked.groupby(
        [
            "sigma_ev",
            "system",
            "ID",
        ],
        sort=True,
    ):
        values = sub[
            "PCF"
        ].to_numpy(dtype=np.float64)

        ranks = sub[
            "source_rank"
        ].to_numpy(dtype=np.float64)

        mean = float(
            np.mean(values)
        )

        sd = float(
            np.std(
                values,
                ddof=0,
            )
        )

        cv = (
            sd / mean
            if mean > 0
            else np.nan
        )

        rows.append({
            "system": system,
            "ID": ID,
            "sigma_ev": sigma_ev,
            "mean_PCF": mean,
            "min_PCF": float(np.min(values)),
            "max_PCF": float(np.max(values)),
            "sd_PCF": sd,
            "cv_PCF": cv,
            "mean_rank": float(np.mean(ranks)),
            "rank_min": int(np.min(ranks)),
            "rank_max": int(np.max(ranks)),
            "rank_range": int(
                np.max(ranks)
                - np.min(ranks)
            ),
        })

    return pd.DataFrame(rows)
