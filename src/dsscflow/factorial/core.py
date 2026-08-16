from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

DEFAULT_CODING = {
    "Donor":  {"TPA": -1, "PTZ": +1},
    "Aux":    {"NONE": -1, "BTD": +1},
    "Bridge": {"T": -1, "TT": +1},
    "Anchor": {"CAA": -1, "RAA": +1},
}


def factorial_effects(
    df: pd.DataFrame,
    response: str,
    factors: list[str] | None = None,
    coding: dict | None = None,
    max_order: int | None = None,
) -> pd.DataFrame:
    coding = coding or DEFAULT_CODING
    factors = factors or list(coding.keys())

    if max_order is None:
        max_order = len(factors)

    rows = []

    for order in range(1, max_order + 1):
        for combo in itertools.combinations(factors, order):
            x = np.ones(len(df), dtype=float)

            for factor in combo:
                x *= df[factor].map(coding[factor]).to_numpy(float)

            y = df[response].to_numpy(float)

            effect = 2.0 * np.mean(x * y)

            plus = y[x == 1]
            minus = y[x == -1]

            rows.append({
                "order": order,
                "term": "×".join(combo),
                "response": response,
                "mean_plus": float(np.mean(plus)),
                "mean_minus": float(np.mean(minus)),
                "effect": float(effect),
                "abs_effect": float(abs(effect)),
            })

    return pd.DataFrame(rows)
