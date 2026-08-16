from importlib.resources import files


def test_gui_contains_author_attribution():
    html = files("dsscflow.gui").joinpath("static/index.html").read_text(encoding="utf-8")
    assert "Gülbin Kurtay" in html
    assert "Hacettepe University, Department of Chemistry" in html
    assert "gulbinkurtay@hacettepe.edu.tr" in html
    assert "0000-0003-0920-8409" in html


def test_gui_version_badge():
    html = files("dsscflow.gui").joinpath("static/index.html").read_text(encoding="utf-8")
    assert "v1.0.0" in html
