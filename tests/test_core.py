from pathlib import Path
import numpy as np
import pandas as pd

from dsscflow.optics.absorption import OpticalGrid, GaussianBroadening, reconstruct_molar_absorptivity
from dsscflow.photon.panel import evaluate_photon_panel
from dsscflow.io import read_sources

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "DSSC16"


def test_absorption_nonnegative():
    td = pd.read_csv(EX / "data/DSSC16_STAGE04B_CAMB3LYP_EXCITED_STATES_LONG.csv")
    d01 = td[td["ID"] == "D01"]
    spec = reconstruct_molar_absorptivity(d01, grid=OpticalGrid(points=1000), broadening=GaussianBroadening(0.30))
    assert np.isfinite(spec["molar_absorptivity"]).all()
    assert (spec["molar_absorptivity"] >= 0).all()


def test_band_conservation_and_d08_winner():
    td = pd.read_csv(EX / "data/DSSC16_STAGE04B_CAMB3LYP_EXCITED_STATES_LONG.csv")
    src = read_sources(EX / "light_sources")
    panel = evaluate_photon_panel(td, src, sigma_ev=0.30)
    bands = ["B1_380_500_fraction","B2_500_600_fraction","B3_600_700_fraction","B4_700_780_fraction"]
    assert (panel[bands].sum(axis=1).sub(1.0).abs().max()) < 1e-12
    winners = panel.sort_values(["light_source","PCF"],ascending=[True,False]).groupby("light_source").first()["ID"]
    assert set(winners) == {"D08"}
