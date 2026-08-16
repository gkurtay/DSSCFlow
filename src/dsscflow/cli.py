from __future__ import annotations

from pathlib import Path
import json
import typer

from dsscflow import __version__
from dsscflow.io import read_transition_csv, read_sources
from dsscflow.workflows import reconstruct_spectra, evaluate_widths, reproduce_dssc16

app = typer.Typer(no_args_is_help=True, help="DSSCFlow: source-conditioned optical screening for molecular sensitizers.")


@app.command()
def version():
    """Print DSSCFlow version."""
    typer.echo(__version__)


@app.command()
def spectra(
    transitions: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(Path("spectra.csv"), "--out", "-o"),
    sigma: float = typer.Option(0.30, "--sigma"),
):
    """Reconstruct Gaussian-broadened molecular absorption spectra."""
    df = read_transition_csv(transitions)
    result = reconstruct_spectra(df, sigma_ev=sigma)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    typer.echo(f"Wrote {out} ({len(result)} rows)")


@app.command()
def photon(
    transitions: Path = typer.Argument(..., exists=True, readable=True),
    sources: Path = typer.Option(..., "--sources", exists=True, file_okay=False),
    out: Path = typer.Option(Path("photon_panel.csv"), "--out", "-o"),
    sigma: list[float] = typer.Option([0.30], "--sigma"),
):
    """Evaluate photon-accessibility descriptors for one or more broadenings."""
    df = read_transition_csv(transitions)
    src = read_sources(sources)
    panel, _ = evaluate_widths(df, src, widths=tuple(sigma))
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)
    typer.echo(f"Wrote {out} ({len(panel)} rows)")


@app.command()
def reproduce(
    example: Path = typer.Argument(Path("examples/DSSC16"), exists=True, file_okay=False),
    out: Path = typer.Option(Path("results/DSSC16"), "--out", "-o"),
):
    """Reproduce the DSSC16 reference analysis and run publication regression checks."""
    report = reproduce_dssc16(example, out)
    typer.echo(json.dumps(report, indent=2))
    if not all(report["checks"].values()):
        raise typer.Exit(code=2)


@app.command()
def gui(
    address: str = typer.Option("127.0.0.1", "--address", help="Local interface address."),
    port: int = typer.Option(8765, "--port", min=1, max=65535, help="Preferred local port."),
    browser: bool = typer.Option(True, "--browser/--no-browser", help="Open the default web browser."),
):
    """Launch the local browser-based DSSCFlow interface."""
    from dsscflow.gui.server import serve
    serve(address=address, port=port, open_browser=browser)


if __name__ == "__main__":
    app()
