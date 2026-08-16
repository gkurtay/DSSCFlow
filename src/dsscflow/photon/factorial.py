from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


FACTOR_ORDER = (
    "Donor",
    "Aux",
    "Bridge",
    "Anchor",
)

FACTOR_CODING = {
    "Donor": {
        "TPA": -1,
        "PTZ": +1,
    },
    "Aux": {
        "NONE": -1,
        "BTD": +1,
    },
    "Bridge": {
        "T": -1,
        "TT": +1,
    },
    "Anchor": {
        "CAA": -1,
        "RAA": +1,
    },
}


def validate_full_factorial(
    design: pd.DataFrame,
) -> None:
    """
    Require a complete unreplicated 2^4 design:
    16 unique factor combinations.
    """

    missing = (
        set(FACTOR_ORDER)
        - set(design.columns)
    )

    if missing:
        raise ValueError(
            f"Missing factors: {sorted(missing)}"
        )

    combinations = (
        design[
            list(FACTOR_ORDER)
        ]
        .drop_duplicates()
    )

    if len(combinations) != 16:
        raise ValueError(
            "Expected exactly 16 unique 2^4 "
            "factor combinations"
        )

    for factor in FACTOR_ORDER:
        observed = set(
            design[factor]
            .dropna()
            .unique()
        )

        expected = set(
            FACTOR_CODING[factor]
        )

        if observed != expected:
            raise ValueError(
                f"Unexpected levels for {factor}: "
                f"{sorted(observed)}"
            )


def _coded_vector(
    design: pd.DataFrame,
    term: tuple[str, ...],
) -> np.ndarray:

    coded = np.ones(
        len(design),
        dtype=np.float64,
    )

    for factor in term:
        values = (
            design[factor]
            .map(
                FACTOR_CODING[factor]
            )
        )

        if values.isna().any():
            raise ValueError(
                f"Unrecognized level in {factor}"
            )

        coded *= values.to_numpy(
            dtype=np.float64
        )

    return coded


def factorial_effect_table(
    design: pd.DataFrame,
    *,
    response: str,
    max_order: int = 2,
) -> pd.DataFrame:
    """
    Compute orthogonal descriptive factorial contrasts.

    Effect = mean(response | coded term = +1)
           - mean(response | coded term = -1)

           = 2 * mean(x_term * response)

    No inferential p-values are assigned.
    """

    validate_full_factorial(design)

    if response not in design.columns:
        raise ValueError(
            f"Missing response column: {response}"
        )

    y = design[
        response
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(y).all():
        raise ValueError(
            "Response values must be finite"
        )

    rows = []

    for order in range(
        1,
        max_order + 1,
    ):
        for term in itertools.combinations(
            FACTOR_ORDER,
            order,
        ):
            x = _coded_vector(
                design,
                term,
            )

            plus = y[x > 0]
            minus = y[x < 0]

            effect = float(
                plus.mean()
                - minus.mean()
            )

            orthogonal_effect = float(
                2.0
                * np.mean(
                    x * y
                )
            )

            if not np.isclose(
                effect,
                orthogonal_effect,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "Factorial contrast identity failed"
                )

            rows.append({
                "order": order,
                "term": "×".join(term),
                "response": response,
                "mean_plus": float(
                    plus.mean()
                ),
                "mean_minus": float(
                    minus.mean()
                ),
                "effect": effect,
                "abs_effect": abs(effect),
            })

    return pd.DataFrame(rows)


def source_specific_pcf_effects(
    panel: pd.DataFrame,
    *,
    max_order: int = 2,
) -> pd.DataFrame:
    """
    Compute source-specific factorial effects on PCF.
    """

    required = {
        "system",
        "ID",
        "light_source",
        "sigma_ev",
        "PCF",
        *FACTOR_ORDER,
    }

    missing = required - set(
        panel.columns
    )

    if missing:
        raise ValueError(
            f"Missing panel columns: {sorted(missing)}"
        )

    rows = []

    for (
        sigma_ev,
        source,
    ), sub in panel.groupby(
        [
            "sigma_ev",
            "light_source",
        ],
        sort=True,
    ):
        if sub["system"].duplicated().any():
            raise ValueError(
                f"Duplicate systems for {source}"
            )

        effects = factorial_effect_table(
            sub,
            response="PCF",
            max_order=max_order,
        )

        effects.insert(
            0,
            "light_source",
            source,
        )

        effects.insert(
            0,
            "sigma_ev",
            sigma_ev,
        )

        rows.append(effects)

    return (
        pd.concat(
            rows,
            ignore_index=True,
        )
        .sort_values(
            [
                "sigma_ev",
                "light_source",
                "order",
                "abs_effect",
            ],
            ascending=[
                True,
                True,
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )
