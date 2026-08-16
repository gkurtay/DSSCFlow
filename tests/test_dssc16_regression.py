from pathlib import Path
import math

from dsscflow.workflows import reproduce_dssc16

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "DSSC16"

EXPECTED_D08 = {
    "AM15G": 0.418517,
    "CIEFL10": 0.581556,
    "LEDB2": 0.470247,
    "LEDB3": 0.529705,
    "LEDB4": 0.557309,
}


def test_publication_regression(tmp_path):
    report = reproduce_dssc16(EX, tmp_path)
    assert report["checks"]["D08_top_all_sources"]
    assert report["checks"]["robust_pareto_matches_publication"]
    assert report["BTD_effect_positive_all_sources"]
    assert report["robust_pareto_15_of_15"] == ["D08", "D14", "D16"]
    for source, expected in EXPECTED_D08.items():
        assert math.isclose(report["D08_PCF_sigma030"][source], expected, rel_tol=0, abs_tol=5e-7)
