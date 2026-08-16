from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_front(
    df: pd.DataFrame,
    maximize: list[str],
    minimize: list[str],
) -> pd.DataFrame:
    records = df.to_dict("records")
    dominated = []

    def dominates(a, b):
        no_worse = True
        strictly_better = False

        for c in maximize:
            if a[c] < b[c]:
                no_worse = False
                break
            if a[c] > b[c]:
                strictly_better = True

        if no_worse:
            for c in minimize:
                if a[c] > b[c]:
                    no_worse = False
                    break
                if a[c] < b[c]:
                    strictly_better = True

        return no_worse and strictly_better

    for i, a in enumerate(records):
        count = 0

        for j, b in enumerate(records):
            if i == j:
                continue

            if dominates(b, a):
                count += 1

        dominated.append(count)

    out = df.copy()
    out["dominated_by_count"] = dominated
    out["pareto_front"] = np.asarray(dominated) == 0

    return out
