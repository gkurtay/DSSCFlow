from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_transition_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"system", "ID", "Donor", "Aux", "Bridge", "Anchor", "wavelength_nm", "oscillator_strength"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing transition columns: {sorted(missing)}")
    return df


def read_sources(directory: str | Path) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    sources: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("*.csv")):
        name = path.name.split("_200_800nm")[0]
        sources[name] = pd.read_csv(path)
    if not sources:
        raise ValueError(f"No source CSV files found in {directory}")
    return sources
