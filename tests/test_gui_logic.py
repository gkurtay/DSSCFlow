from __future__ import annotations

import numpy as np

from dsscflow.gui.logic import (
    analyse,
    bundle_summary,
    export_zip_bytes,
    load_demo_data,
    publication_checks,
)
from dsscflow.gui.server import choose_port


def test_demo_gui_analysis_matches_publication_regression():
    demo = load_demo_data()
    bundle = analyse(
        demo["transitions"],
        demo["sources"],
        widths=(0.20, 0.30, 0.40),
        active_sigma=0.30,
        active_source="AM15G",
        descriptors=demo["descriptors"],
        ifct=demo["ifct"],
        tdm=demo["tdm"],
    )
    checks = publication_checks(bundle)
    assert checks["D08_top_all_sources"] is True
    assert checks["BTD_effect_positive_all_sources"] is True
    assert checks["robust_pareto_matches_D08_D14_D16"] is True

    summary = bundle_summary(bundle, ["D08", "D14", "D16"])
    assert summary["meta"]["dyes"] == 16
    assert summary["meta"]["transitions"] == 480
    assert set(summary["spectra"]) == {"D08", "D14", "D16"}
    assert summary["ranking"][0]["ID"] == "D08"
    assert summary["pareto"] is not None


def test_gui_optical_path_and_export_bundle():
    demo = load_demo_data()
    bundle = analyse(
        demo["transitions"],
        {"AM15G": demo["sources"]["AM15G"]},
        widths=(0.30,),
        active_sigma=0.30,
        concentration_molar=1.0e-5,
        path_length_cm=1.0,
    )
    assert len(bundle.panel) == 16
    assert np.isfinite(bundle.panel["PCF"]).all()
    payload = export_zip_bytes(bundle, {"available": False})
    assert payload[:2] == b"PK"
    assert len(payload) > 1000


def test_choose_port_returns_bindable_local_port():
    port = choose_port("127.0.0.1", 18765)
    assert 18765 <= port < 18815
