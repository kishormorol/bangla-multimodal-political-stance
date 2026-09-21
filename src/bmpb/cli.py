"""`bmpb` — one entry point for the whole pipeline.

bmpb data                      mirror the Drive folder into data/raw/
bmpb ingest                    raw sheets -> data/processed/corpus.csv
bmpb splits                    grouped, leakage-free train/val/test
bmpb audit                     data health report (labels, images, leakage)
bmpb train --config ...        train one model into experiments/
bmpb evaluate --run ...        rescore a finished run
bmpb leaderboard               results table across runs and published CSVs
bmpb models                    list registered model families
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from bmpb.paths import EXPERIMENTS, TABLES

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


@app.command()
def data(
    no_images: bool = typer.Option(False, "--no-images", help="skip the image folders")
) -> None:
    """Mirror the shared Google Drive folder into data/raw/."""
    from bmpb.config import DataConfig
    from bmpb.data.download import download

    download(DataConfig.load(), images=not no_images)


@app.command()
def ingest() -> None:
    """Normalize the raw sheets into data/processed/corpus.csv."""
    from bmpb.data.ingest import ingest as run_ingest

    frame = run_ingest()
    console.print(f"[green]corpus:[/] {len(frame)} items")


@app.command()
def splits() -> None:
    """Rebuild the grouped, stratified train/val/test splits."""
    from bmpb.data.splits import make_splits

    made = make_splits()
    for name, frame in made.items():
        console.print(f"[green]{name}[/]: {len(frame)}")


@app.command()
def audit() -> None:
    """Report label balance, image coverage, and split leakage."""
    from bmpb.audit import run_audit

    report = run_audit()
    console.print_json(json.dumps(report, indent=2, default=str))


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="experiment config"),
    out: Path = typer.Option(EXPERIMENTS, "--out", "-o", help="where to write the run directory"),
) -> None:
    """Train one model and write a run directory."""
    from bmpb.train import train as run_train

    directory = run_train(config, out)
    console.print(f"[green]run written to[/] {directory}")


@app.command()
def evaluate(run: Path = typer.Option(..., "--run", "-r", exists=True)) -> None:
    """Rescore a finished run from its predictions.csv."""
    from bmpb.evaluate import evaluate_run

    scores = evaluate_run(run)
    console.print(scores.summary())


@app.command()
def leaderboard(
    runs: Path = typer.Option(EXPERIMENTS, "--runs"),
    out: Path = typer.Option(TABLES, "--out"),
) -> None:
    """Rebuild reports/tables/leaderboard.md."""
    from bmpb.evaluate import leaderboard as build

    console.print(f"[green]wrote[/] {build(runs, out)}")


@app.command()
def models() -> None:
    """List the registered model families and the configs that use them."""
    from bmpb.models.registry import available
    from bmpb.paths import CONFIGS

    table = Table("family", "configs")
    configs_by_family: dict[str, list[str]] = {}
    for path in sorted(CONFIGS.rglob("*.yaml")):
        if path.name == "data.yaml":
            continue
        import yaml

        payload = yaml.safe_load(path.read_text()) or {}
        configs_by_family.setdefault(payload.get("family", "?"), []).append(
            str(path.relative_to(CONFIGS))
        )
    for family in available():
        table.add_row(family, "\n".join(configs_by_family.get(family, [])) or "—")
    console.print(table)


if __name__ == "__main__":
    app()
