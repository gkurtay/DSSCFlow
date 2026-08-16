from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from io import StringIO, BytesIO
from pathlib import Path
import json
import math
import re
import zipfile

import numpy as np
import pandas as pd

from dsscflow.io import read_transition_csv
from dsscflow.optics.absorption import OpticalPath
from dsscflow.photon.factorial import source_specific_pcf_effects
from dsscflow.photon.panel import summarize_source_robustness
from dsscflow.photon.source import validate_source_table
from dsscflow.workflows import reconstruct_spectra, evaluate_widths, pareto_master


@dataclass
class AnalysisBundle:
    transitions: pd.DataFrame
    sources: dict[str, pd.DataFrame]
    spectra: pd.DataFrame
    panel: pd.DataFrame
    effects: pd.DataFrame
    robustness: pd.DataFrame
    pareto_master_table: pd.DataFrame | None
    pareto_summary: pd.DataFrame | None
    active_sigma: float
    active_source: str


def _resource_csv(path_parts: tuple[str, ...]) -> pd.DataFrame:
    ref = resources.files("dsscflow")
    for part in path_parts:
        ref = ref.joinpath(part)
    with ref.open("rb") as handle:
        return pd.read_csv(handle)


def load_demo_data() -> dict[str, object]:
    data_names = {
        "transitions": "DSSC16_STAGE04B_CAMB3LYP_EXCITED_STATES_LONG.csv",
        "descriptors": "DSSC16_GLOBAL_DESCRIPTORS.csv",
        "ifct": "DSSC16_IFCT_BATCH24_RESULTS.csv",
        "tdm": "DSSC16_FRAGMENT_TDM_BATCH24_SUMMARY.csv",
    }
    out: dict[str, object] = {}
    for key, name in data_names.items():
        out[key] = _resource_csv(("demo", "DSSC16", "data", name))

    src_root = resources.files("dsscflow").joinpath("demo", "DSSC16", "light_sources")
    sources: dict[str, pd.DataFrame] = {}
    for ref in sorted(src_root.iterdir(), key=lambda x: x.name):
        if ref.name.lower().endswith(".csv"):
            name = ref.name.split("_200_800nm")[0]
            with ref.open("rb") as handle:
                sources[name] = validate_source_table(pd.read_csv(handle))
    out["sources"] = sources
    return out


def parse_csv_text(text: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(text))


def source_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_200_800nm_10000pt$", "", stem, flags=re.I)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return stem or "source"


def parse_sources_payload(items: list[dict[str, str]]) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    for item in items:
        filename = str(item.get("name", "source.csv"))
        text = str(item.get("text", ""))
        if not text.strip():
            continue
        name = source_name_from_filename(filename)
        if name in sources:
            raise ValueError(f"Duplicate source name after normalization: {name}")
        sources[name] = validate_source_table(parse_csv_text(text))
    if not sources:
        raise ValueError("At least one illumination-source CSV is required.")
    return sources


def validate_optional_pareto_tables(
    descriptors: pd.DataFrame | None,
    ifct: pd.DataFrame | None,
    tdm: pd.DataFrame | None,
) -> bool:
    if descriptors is None or ifct is None or tdm is None:
        return False
    d_req = {"system", "ID", "lambda_total_eV"}
    i_req = {"system", "state", "status", "CT_percent", "Anchor_electron_percent", "Donor_to_Anchor_net_e"}
    t_req = {"system", "state", "status", "off_diagonal_fraction"}
    return d_req.issubset(descriptors.columns) and i_req.issubset(ifct.columns) and t_req.issubset(tdm.columns)


def analyse(
    transitions: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    *,
    widths: tuple[float, ...] = (0.20, 0.30, 0.40),
    active_sigma: float = 0.30,
    active_source: str | None = None,
    concentration_molar: float = 1.0e-5,
    path_length_cm: float = 1.0,
    descriptors: pd.DataFrame | None = None,
    ifct: pd.DataFrame | None = None,
    tdm: pd.DataFrame | None = None,
) -> AnalysisBundle:
    if not widths:
        raise ValueError("At least one broadening width is required.")
    widths = tuple(sorted({float(x) for x in widths}))
    if any(x <= 0 for x in widths):
        raise ValueError("Broadening widths must be positive.")
    if not any(np.isclose(active_sigma, x) for x in widths):
        widths = tuple(sorted(widths + (float(active_sigma),)))
    if concentration_molar < 0 or path_length_cm < 0:
        raise ValueError("Concentration and path length must be non-negative.")

    transitions = transitions.copy()
    required = {"system", "ID", "Donor", "Aux", "Bridge", "Anchor", "wavelength_nm", "oscillator_strength"}
    missing = required - set(transitions.columns)
    if missing:
        raise ValueError(f"Missing transition columns: {sorted(missing)}")

    source_names = sorted(sources)
    if not source_names:
        raise ValueError("At least one illumination source is required.")
    active_source = active_source or source_names[0]
    if active_source not in sources:
        active_source = source_names[0]

    optical_path = OpticalPath(
        concentration_molar=float(concentration_molar),
        path_length_cm=float(path_length_cm),
    )
    spectra = reconstruct_spectra(transitions, sigma_ev=float(active_sigma))
    panel, effects = evaluate_widths(
        transitions,
        sources,
        widths=widths,
        optical_path=optical_path,
    )
    active_panel = panel[np.isclose(panel["sigma_ev"], float(active_sigma))].copy()
    robustness = summarize_source_robustness(active_panel)

    pmaster = None
    psummary = None
    if validate_optional_pareto_tables(descriptors, ifct, tdm):
        pmaster, psummary = pareto_master(panel, descriptors.copy(), ifct.copy(), tdm.copy())

    return AnalysisBundle(
        transitions=transitions,
        sources=sources,
        spectra=spectra,
        panel=panel,
        effects=effects,
        robustness=robustness,
        pareto_master_table=pmaster,
        pareto_summary=psummary,
        active_sigma=float(active_sigma),
        active_source=active_source,
    )


def lambda_max_table(spectra: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (system, ident), sub in spectra.groupby(["system", "ID"], sort=True):
        idx = sub["molar_absorptivity"].astype(float).idxmax()
        row = sub.loc[idx]
        rows.append({
            "system": system,
            "ID": ident,
            "lambda_max_nm": float(row["wavelength_nm"]),
            "epsilon_max": float(row["molar_absorptivity"]),
        })
    return pd.DataFrame(rows)


def publication_checks(bundle: AnalysisBundle) -> dict[str, object]:
    active = bundle.panel[np.isclose(bundle.panel["sigma_ev"], 0.30)].copy()
    if active.empty:
        return {"available": False}
    top = (
        active.sort_values(["light_source", "PCF"], ascending=[True, False])
        .groupby("light_source", as_index=False)
        .first()
    )
    top_ids = dict(zip(top["light_source"], top["ID"]))
    aux = source_specific_pcf_effects(active, max_order=1)
    aux_effect = aux[(aux["order"] == 1) & (aux["term"] == "Aux")]
    result: dict[str, object] = {
        "available": True,
        "D08_top_all_sources": bool(set(top_ids.values()) == {"D08"}),
        "BTD_effect_positive_all_sources": bool((aux_effect["effect"] > 0).all()),
        "top_dye_by_source": top_ids,
    }
    if bundle.pareto_summary is not None:
        s = bundle.pareto_summary
        possible = int(s["possible_runs"].max()) if "possible_runs" in s else 0
        robust = sorted(s.loc[(s["exact_membership"] == possible) & (s["epsilon_membership"] == possible), "ID"])
        result["robust_pareto_all_runs"] = robust
        result["robust_pareto_matches_D08_D14_D16"] = robust == ["D08", "D14", "D16"]
    return result


def dataframe_records(df: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> list[dict[str, object]]:
    x = df if columns is None else df[columns]
    if limit is not None:
        x = x.head(limit)
    # JSON-friendly NaN handling
    return json.loads(x.to_json(orient="records"))


def downsample_spectra(spectra: pd.DataFrame, ids: list[str], max_points: int = 450) -> dict[str, list[list[float]]]:
    output: dict[str, list[list[float]]] = {}
    for ident in ids:
        sub = spectra[spectra["ID"] == ident].sort_values("wavelength_nm")
        if sub.empty:
            continue
        stride = max(1, math.ceil(len(sub) / max_points))
        sub = sub.iloc[::stride]
        y = sub["molar_absorptivity"].to_numpy(float)
        ymax = float(y.max()) if len(y) else 0.0
        yn = y / ymax if ymax > 0 else y
        output[ident] = [[float(x), float(v)] for x, v in zip(sub["wavelength_nm"], yn)]
    return output


def bundle_summary(bundle: AnalysisBundle, selected_ids: list[str] | None = None) -> dict[str, object]:
    ids = sorted(bundle.transitions["ID"].dropna().astype(str).unique())
    selected_ids = [x for x in (selected_ids or []) if x in ids]
    if not selected_ids:
        # choose highest-PCF three under active source/sigma
        ranking = bundle.panel[
            np.isclose(bundle.panel["sigma_ev"], bundle.active_sigma)
            & (bundle.panel["light_source"] == bundle.active_source)
        ].sort_values("PCF", ascending=False)
        selected_ids = ranking["ID"].astype(str).head(3).tolist()

    ranking = bundle.panel[
        np.isclose(bundle.panel["sigma_ev"], bundle.active_sigma)
        & (bundle.panel["light_source"] == bundle.active_source)
    ].sort_values("PCF", ascending=False).copy()
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    eff = bundle.effects[
        np.isclose(bundle.effects["sigma_ev"], bundle.active_sigma)
        & (bundle.effects["light_source"] == bundle.active_source)
        & (bundle.effects["order"] == 1)
    ].sort_values("abs_effect", ascending=False)

    result: dict[str, object] = {
        "meta": {
            "dyes": int(bundle.transitions["ID"].nunique()),
            "transitions": int(len(bundle.transitions)),
            "sources": sorted(bundle.sources),
            "widths": sorted(float(x) for x in bundle.panel["sigma_ev"].unique()),
            "active_sigma": bundle.active_sigma,
            "active_source": bundle.active_source,
            "ids": ids,
            "selected_ids": selected_ids,
        },
        "lambda_max": dataframe_records(lambda_max_table(bundle.spectra)),
        "spectra": downsample_spectra(bundle.spectra, selected_ids),
        "ranking": dataframe_records(ranking, ["rank", "ID", "system", "PCF", "PCC_nm", "PCB_nm"]),
        "factorial_main_effects": dataframe_records(eff, ["term", "effect", "mean_plus", "mean_minus"]),
        "robustness": dataframe_records(bundle.robustness.sort_values(["mean_rank", "mean_PCF"], ascending=[True, False])),
    }
    if bundle.pareto_master_table is not None:
        p = bundle.pareto_master_table[
            np.isclose(bundle.pareto_master_table["sigma_ev"], bundle.active_sigma)
            & (bundle.pareto_master_table["source"] == bundle.active_source)
        ].copy()
        result["pareto"] = dataframe_records(
            p,
            ["ID", "optical_value", "CT_percent", "lambda_total_eV", "pareto_front", "epsilon_pareto_front", "core_compromise_score"],
        )
        result["pareto_summary"] = dataframe_records(bundle.pareto_summary)
    else:
        result["pareto"] = None
        result["pareto_summary"] = None
    return result


def export_zip_bytes(bundle: AnalysisBundle, checks: dict[str, object] | None = None) -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        tables = {
            "spectra.csv": bundle.spectra,
            "photon_panel.csv": bundle.panel,
            "factorial_effects.csv": bundle.effects,
            "source_robustness.csv": bundle.robustness,
            "lambda_max.csv": lambda_max_table(bundle.spectra),
        }
        if bundle.pareto_master_table is not None:
            tables["pareto_master.csv"] = bundle.pareto_master_table
        if bundle.pareto_summary is not None:
            tables["pareto_summary.csv"] = bundle.pareto_summary
        for name, df in tables.items():
            zf.writestr(name, df.to_csv(index=False))
        if checks is not None:
            zf.writestr("analysis_checks.json", json.dumps(checks, indent=2))
    return bio.getvalue()
