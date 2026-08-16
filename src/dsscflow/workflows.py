from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from dsscflow.io import read_transition_csv, read_sources
from dsscflow.optics.absorption import GaussianBroadening, OpticalGrid, OpticalPath, reconstruct_molar_absorptivity
from dsscflow.photon.panel import evaluate_photon_panel, summarize_source_robustness
from dsscflow.photon.factorial import source_specific_pcf_effects
from dsscflow.pareto.decision import core_pareto_front, epsilon_pareto_front, add_decision_scores


def reconstruct_spectra(transitions: pd.DataFrame, sigma_ev: float = 0.30) -> pd.DataFrame:
    rows = []
    grid = OpticalGrid()
    broadening = GaussianBroadening(sigma_ev=sigma_ev)
    for system, td in transitions.groupby("system", sort=True):
        spec = reconstruct_molar_absorptivity(td, grid=grid, broadening=broadening)
        meta = td.iloc[0][["system", "ID", "Donor", "Aux", "Bridge", "Anchor"]].to_dict()
        for k, v in meta.items():
            spec[k] = v
        spec["sigma_ev"] = sigma_ev
        rows.append(spec)
    return pd.concat(rows, ignore_index=True)


def evaluate_widths(
    transitions: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    widths=(0.20, 0.30, 0.40),
    *,
    optical_path: OpticalPath = OpticalPath(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panels=[]; effects=[]
    for sigma in widths:
        p=evaluate_photon_panel(transitions, sources, sigma_ev=float(sigma), optical_path=optical_path)
        panels.append(p)
        effects.append(source_specific_pcf_effects(p, max_order=2))
    return pd.concat(panels, ignore_index=True), pd.concat(effects, ignore_index=True)


def pareto_master(panel: pd.DataFrame, descriptors: pd.DataFrame, ifct: pd.DataFrame, tdm: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    ifct_s1 = ifct[(ifct["state"]==1)&(ifct["status"]=="OK")][["system","CT_percent","Anchor_electron_percent","Donor_to_Anchor_net_e"]].copy()
    tdm_s1 = tdm[(tdm["state"]==1)&(tdm["status"]=="OK")][["system","off_diagonal_fraction"]].copy()
    base=(descriptors.merge(ifct_s1,on="system",validate="one_to_one").merge(tdm_s1,on="system",validate="one_to_one"))
    runs=[]
    for (sigma,source), optical in panel.groupby(["sigma_ev","light_source"], sort=True):
        x=base.merge(optical[["system","PCF","PCC_nm","PCB_nm"]],on="system",validate="one_to_one").rename(columns={"PCF":"optical_value"})
        x=core_pareto_front(x)
        x=epsilon_pareto_front(x)
        x=add_decision_scores(x)
        x["sigma_ev"]=sigma; x["source"]=source; x["optical_metric"]=f"PCF_sigma_{sigma:.2f}"
        runs.append(x)
    master=pd.concat(runs,ignore_index=True)
    summary=(master.groupby(["system","ID"]).agg(
        exact_membership=("pareto_front","sum"), epsilon_membership=("epsilon_pareto_front","sum"),
        mean_core_score=("core_compromise_score","mean"), mean_ideal_distance=("distance_to_ideal","mean"),
        mean_mechanistic_support=("mechanistic_support_score","mean"), mean_PCF=("optical_value","mean"),
        min_PCF=("optical_value","min"), max_PCF=("optical_value","max")).reset_index())
    summary["possible_runs"]=master[["sigma_ev","source"]].drop_duplicates().shape[0]
    return master, summary.sort_values(["epsilon_membership","exact_membership","mean_ideal_distance"],ascending=[False,False,True])


def reproduce_dssc16(example_dir: str | Path, output_dir: str | Path) -> dict:
    example_dir=Path(example_dir); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    transitions=read_transition_csv(example_dir/"data/DSSC16_STAGE04B_CAMB3LYP_EXCITED_STATES_LONG.csv")
    sources=read_sources(example_dir/"light_sources")
    spectra=reconstruct_spectra(transitions,0.30)
    spectra.to_csv(output_dir/"DSSC16_SPECTRA_SIGMA030.csv",index=False)
    panel,effects=evaluate_widths(transitions,sources)
    panel.to_csv(output_dir/"DSSC16_PHOTON_BROADENING_PANEL.csv",index=False)
    effects.to_csv(output_dir/"DSSC16_PHOTON_BROADENING_FACTORIAL.csv",index=False)
    production=panel[np.isclose(panel["sigma_ev"],0.30)].copy()
    production.to_csv(output_dir/"DSSC16_PHOTON_PANEL_SIGMA030.csv",index=False)
    robustness=summarize_source_robustness(production)
    robustness.to_csv(output_dir/"DSSC16_PHOTON_ROBUSTNESS_SIGMA030.csv",index=False)
    descriptors=pd.read_csv(example_dir/"data/DSSC16_GLOBAL_DESCRIPTORS.csv")
    ifct=pd.read_csv(example_dir/"data/DSSC16_IFCT_BATCH24_RESULTS.csv")
    tdm=pd.read_csv(example_dir/"data/DSSC16_FRAGMENT_TDM_BATCH24_SUMMARY.csv")
    master,summary=pareto_master(panel,descriptors,ifct,tdm)
    master.to_csv(output_dir/"DSSC16_PHOTON_PARETO_15RUN_MASTER.csv",index=False)
    summary.to_csv(output_dir/"DSSC16_PHOTON_PARETO_ROBUST_SUMMARY.csv",index=False)
    # Publication regression checks
    top = production.sort_values(["light_source","PCF"],ascending=[True,False]).groupby("light_source").first().reset_index()
    top_ids=dict(zip(top["light_source"],top["ID"]))
    d08=production[production["ID"]=="D08"].set_index("light_source")["PCF"].to_dict()
    aux=source_specific_pcf_effects(production,max_order=1)
    aux_effect=aux[(aux["order"]==1)&(aux["term"]=="Aux")]
    fully_robust=set(summary.loc[(summary["exact_membership"]==15)&(summary["epsilon_membership"]==15),"ID"])
    report={
        "top_dye_by_source": top_ids,
        "D08_PCF_sigma030": {k:float(v) for k,v in d08.items()},
        "BTD_effect_positive_all_sources": bool((aux_effect["effect"]>0).all()),
        "robust_pareto_15_of_15": sorted(fully_robust),
        "checks": {
            "D08_top_all_sources": bool(set(top_ids.values())=={"D08"}),
            "robust_pareto_matches_publication": fully_robust=={"D08","D14","D16"},
        },
    }
    (output_dir/"DSSC16_REPRODUCIBILITY_REPORT.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
