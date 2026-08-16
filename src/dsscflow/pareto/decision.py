from __future__ import annotations

import numpy as np
import pandas as pd


def _benefit_scale(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()

    if np.isclose(lo, hi):
        return pd.Series(1.0, index=s.index)

    return (s - lo) / (hi - lo)


def _cost_scale(s: pd.Series) -> pd.Series:
    return 1.0 - _benefit_scale(s)


def core_pareto_front(
    df: pd.DataFrame,
    optical_col: str = "optical_value",
    ct_col: str = "CT_percent",
    reorg_col: str = "lambda_total_eV",
) -> pd.DataFrame:
    """
    Three-objective DSSC screening front:

    maximize optical compatibility
    maximize S1 IFCT CT_percent
    minimize total reorganization energy

    CT_percent is an interfragment redistribution descriptor,
    not an injection-efficiency measure.
    """
    records = df.to_dict("records")
    counts = []

    def dominates(a, b):
        no_worse = (
            a[optical_col] >= b[optical_col]
            and a[ct_col] >= b[ct_col]
            and a[reorg_col] <= b[reorg_col]
        )

        better = (
            a[optical_col] > b[optical_col]
            or a[ct_col] > b[ct_col]
            or a[reorg_col] < b[reorg_col]
        )

        return no_worse and better

    for i, a in enumerate(records):
        count = 0

        for j, b in enumerate(records):
            if i != j and dominates(b, a):
                count += 1

        counts.append(count)

    out = df.copy()
    out["dominated_by_count"] = counts
    out["pareto_front"] = np.asarray(counts) == 0

    return out


def epsilon_pareto_front(
    df: pd.DataFrame,
    optical_col: str = "optical_value",
    ct_col: str = "CT_percent",
    reorg_col: str = "lambda_total_eV",
    optical_epsilon_fraction: float = 0.02,
    ct_epsilon_fraction: float = 0.01,
    reorg_epsilon_fraction: float = 0.03,
) -> pd.DataFrame:
    """
    ε-dominance front using fractions of the observed
    descriptor ranges.
    """
    opt_range = float(
        df[optical_col].max() - df[optical_col].min()
    )
    ct_range = float(
        df[ct_col].max() - df[ct_col].min()
    )
    reorg_range = float(
        df[reorg_col].max() - df[reorg_col].min()
    )

    eps_opt = optical_epsilon_fraction * opt_range
    eps_ct = ct_epsilon_fraction * ct_range
    eps_reorg = reorg_epsilon_fraction * reorg_range

    records = df.to_dict("records")
    counts = []

    def dominates(a, b):
        no_worse = (
            a[optical_col] >= b[optical_col] - eps_opt
            and a[ct_col] >= b[ct_col] - eps_ct
            and a[reorg_col] <= b[reorg_col] + eps_reorg
        )

        materially_better = (
            a[optical_col] > b[optical_col] + eps_opt
            or a[ct_col] > b[ct_col] + eps_ct
            or a[reorg_col] < b[reorg_col] - eps_reorg
        )

        return no_worse and materially_better

    for i, a in enumerate(records):
        count = 0

        for j, b in enumerate(records):
            if i != j and dominates(b, a):
                count += 1

        counts.append(count)

    out = df.copy()
    out["epsilon_dominated_by_count"] = counts
    out["epsilon_pareto_front"] = np.asarray(counts) == 0

    return out


def add_decision_scores(
    df: pd.DataFrame,
    optical_col: str = "optical_value",
    ct_col: str = "CT_percent",
    reorg_col: str = "lambda_total_eV",
    anchor_col: str | None = "Anchor_electron_percent",
    tdm_col: str | None = "off_diagonal_fraction",
) -> pd.DataFrame:
    """
    Add transparent secondary scores.

    The core score is NOT used to define Pareto membership.
    Anchor-electron and fragment-TDM descriptors are treated
    as mechanistic annotations rather than mandatory objectives.
    """
    out = df.copy()

    out["z_optical"] = _benefit_scale(out[optical_col])
    out["z_CT"] = _benefit_scale(out[ct_col])
    out["z_reorg"] = _cost_scale(out[reorg_col])

    out["core_compromise_score"] = (
        out["z_optical"]
        + out["z_CT"]
        + out["z_reorg"]
    ) / 3.0

    out["distance_to_ideal"] = np.sqrt(
        (1.0 - out["z_optical"]) ** 2
        + (1.0 - out["z_CT"]) ** 2
        + (1.0 - out["z_reorg"]) ** 2
    )

    if (
        anchor_col is not None
        and tdm_col is not None
        and anchor_col in out.columns
        and tdm_col in out.columns
    ):
        out["z_anchor"] = _benefit_scale(out[anchor_col])
        out["z_TDM"] = _benefit_scale(out[tdm_col])

        out["mechanistic_support_score"] = (
            out["z_anchor"] + out["z_TDM"]
        ) / 2.0

    return out


def robust_membership_summary(
    runs: pd.DataFrame,
    system_col: str = "system",
    id_col: str = "ID",
) -> pd.DataFrame:
    """
    Summarize Pareto membership across source/metric runs.
    """
    required = {
        "pareto_front",
        "epsilon_pareto_front",
        "core_compromise_score",
        "distance_to_ideal",
    }

    missing = required - set(runs.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    agg = {
        "pareto_front": "sum",
        "epsilon_pareto_front": "sum",
        "core_compromise_score": "mean",
        "distance_to_ideal": "mean",
    }

    if "mechanistic_support_score" in runs.columns:
        agg["mechanistic_support_score"] = "mean"

    out = (
        runs
        .groupby([system_col, id_col])
        .agg(agg)
        .reset_index()
        .rename(columns={
            "pareto_front": "pareto_membership",
            "epsilon_pareto_front":
                "epsilon_pareto_membership",
            "core_compromise_score": "mean_core_score",
            "distance_to_ideal": "mean_ideal_distance",
            "mechanistic_support_score":
                "mean_mechanistic_support",
        })
    )

    total_runs = (
        runs[["source", "optical_metric"]]
        .drop_duplicates()
        .shape[0]
        if {"source", "optical_metric"}.issubset(runs.columns)
        else None
    )

    if total_runs:
        out["possible_runs"] = total_runs
        out["robust_pareto_fraction"] = (
            out["pareto_membership"] / total_runs
        )
        out["robust_epsilon_fraction"] = (
            out["epsilon_pareto_membership"] / total_runs
        )

    return out
