"""``nnmlx`` — command-line interface for the MLX nnU-Net / TotalSegmentator toolkit.

A thin shell over the toolkit API (``TaskCatalog`` / ``ModelStore`` / ``segment``)
so real-weights runs are one command instead of a script — the fast path for
exercising tasks during development.

    nnmlx segment total_fast ct.nii.gz seg.nii.gz
    nnmlx tasks list --modality CT
    nnmlx tasks show total
    nnmlx models list

Run via uv: ``uv run nnmlx ...`` (or ``uv run python -m nnunet_inference_mlx ...``).

No hidden state: every command builds an explicit, request-scoped ``ModelStore``
and ``TaskCatalog`` from the shared options on the top-level callback. Heavy
imports (MLX, SITK) are deferred into the commands that need them, so ``--help``
and catalog inspection stay instant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="MLX nnU-Net / TotalSegmentator inference toolkit.",
)
tasks_app = typer.Typer(no_args_is_help=True, help="Inspect the task catalog (name → recipe).")
models_app = typer.Typer(no_args_is_help=True, help="Inspect the model store (downloaded / loaded models).")
app.add_typer(tasks_app, name="tasks")
app.add_typer(models_app, name="models")


@dataclass
class Config:
    """Request-scoped store/catalog settings from the top-level options."""

    ecosystem: str
    model_root: Optional[Path]
    max_memory_mb: int


@app.callback()
def _main(
    ctx: typer.Context,
    ecosystem: str = typer.Option(
        "totalsegmentator", "--ecosystem", "-e",
        help="Model ecosystem: totalsegmentator | nnunet | moose.",
    ),
    model_root: Optional[Path] = typer.Option(
        None, "--model-root",
        help="Model root dir (overrides env / built-in default for the ecosystem).",
    ),
    max_memory_mb: int = typer.Option(
        4000, "--max-memory-mb",
        help="Loaded-model memory budget; the store evicts (LRU) beyond it.",
    ),
) -> None:
    ctx.obj = Config(ecosystem=ecosystem, model_root=model_root, max_memory_mb=max_memory_mb)


def _store(cfg: Config):
    from .store import ModelStore
    return ModelStore(cfg.ecosystem, model_root_dir=cfg.model_root,
                      max_memory_mb=cfg.max_memory_mb)


def _catalog(cfg: Config):
    from .catalog import TaskCatalog
    return TaskCatalog(cfg.ecosystem)


def _read_volume(path: Path):
    """NIfTI file → NiftiReader; directory → DicomReader."""
    from .imageio import DicomReader, NiftiReader
    reader = DicomReader() if path.is_dir() else NiftiReader()
    return reader.read(path)


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------


@app.command()
def segment(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="Task name, e.g. 'total' or 'total_fast' (qualify as 'ts:total' if ambiguous)."),
    input: Path = typer.Argument(..., exists=True, help="Input NIfTI file or DICOM series directory."),
    output: Path = typer.Argument(..., help="Output NIfTI path for the segmentation."),
    reorient_to: str = typer.Option("RAS", "--reorient-to", help="Canonical orientation for inference (RAS = nnU-Net/TS convention; do not change unless you know the model's training orientation)."),
    reorient: bool = typer.Option(True, "--reorient/--no-reorient", help="Reorient to canonical before inference (and back after)."),
    peak_working_memory_mb: Optional[int] = typer.Option(None, "--peak-working-memory-mb", help="Inverse-resample slab budget; auto-tiers from RAM if unset."),
    output_scaling: Optional[float] = typer.Option(None, "--output-scaling", help="Output resolution multiplier (2 = finer/half-spacing, 0.5 = coarser). Renders from logits; header fixed to same extent."),
    output_spacing: Optional[float] = typer.Option(None, "--output-spacing", help="Output spacing in mm (isotropic). Alternative to --output-scaling."),
    at_model_spacing: bool = typer.Option(False, "--at-model-spacing", help="Write at the model's native training spacing (no upsample back to the input grid)."),
    resample: str = typer.Option("linear", "--resample", help="Inverse resample: 'linear' (logit interp, path B, higher fidelity), 'nearest' (label NN, fastest, stair-stepped), or 'onehot' (one-hot label interp via the same kernel as 'linear' - smooth but resampled after argmax; for direct path-A-vs-B comparison)."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Sliding-window patches per forward pass. Higher better utilizes large GPUs (try 2-4 on M-series Max/Ultra); None = auto from RAM."),
    tile_step_size: float = typer.Option(0.5, "--tile-step-size", help="Sliding-window step as a fraction of patch size (0.5 = 50%% overlap). Higher = fewer patches = faster, slightly coarser at tile seams."),
    download: bool = typer.Option(True, "--download/--no-download", help="Auto-download missing weights (default on)."),
) -> None:
    """Run a named task on a volume and write the segmentation.

    Output resolution: by default the segmentation is written on the input's
    grid. --output-scaling / --output-spacing / --at-model-spacing (mutually
    exclusive) change only the output *sampling* — the labels still occupy the
    input's physical space. Note this is distinct from the model's own
    resolution (e.g. 'total_fast' is a 3 mm model): a finer output grid resamples
    the logits, it does not add detail the model never resolved. (Single-model
    tasks only for now.)
    """
    import numpy as np

    from .catalog import AmbiguousTaskError
    from .imageio import NiftiWriter
    from .segment import segment as run_segment

    if sum(x is not None and x is not False
           for x in (output_scaling, output_spacing, at_model_spacing or None)) > 1:
        typer.secho("pass at most one of --output-scaling / --output-spacing / --at-model-spacing",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    cfg: Config = ctx.obj
    store = _store(cfg)
    catalog = _catalog(cfg)
    try:
        spec = catalog.get(task)
    except AmbiguousTaskError as e:
        typer.secho(f"ambiguous task {task!r}: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    except KeyError as e:
        typer.secho(f"unknown task {task!r}. Try `nnmlx tasks list`. ({e})",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    typer.echo(f"task   : {spec.qualified_name}  (shape={spec.shape})")

    if download:
        from .segment import required_weights_ids
        need = [i for i in required_weights_ids(spec, store=store, catalog=catalog)
                if i not in set(store.downloaded())]
        if need:
            typer.echo(f"download: fetching weights {need} ...")
            try:
                store.download(need)
            except FileNotFoundError as e:
                typer.secho(str(e), fg=typer.colors.RED, err=True)
                raise typer.Exit(2)

    typer.echo(f"input  : {input}")
    image = _read_volume(input)
    typer.echo(f"         {image.geometry.shape_zyx} @ "
               f"{tuple(round(s, 3) for s in image.geometry.spacing_zyx)} mm")

    seg = run_segment(
        spec, image, store=store, catalog=catalog,
        reorient_to=reorient_to if reorient else None,
        peak_working_memory_mb=peak_working_memory_mb,
        output_spacing=output_spacing,
        output_scaling=output_scaling,
        at_model_spacing=at_model_spacing,
        output_interpolation=resample,
        step_size=tile_step_size,
        batch_size=batch_size,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    NiftiWriter().write(output, seg)
    typer.secho(f"wrote  : {output}", fg=typer.colors.GREEN)
    typer.echo(f"         {seg.geometry.shape_zyx} @ "
               f"{tuple(round(s, 3) for s in seg.geometry.spacing_zyx)} mm")
    # Label/voxel summary is cosmetic — never fail the (already-written) output
    # over it (e.g. a broken numpy install where np.unique can't import numpy.ma).
    try:
        data = np.asarray(seg.data)
        labels = sorted(int(v) for v in np.unique(data) if v != 0)
        typer.echo(f"         {len(labels)} foreground labels (max {max(labels) if labels else 0}), "
                   f"{int((data > 0).sum())} fg voxels")
    except Exception as e:
        typer.echo(f"         (label summary unavailable: {type(e).__name__})")


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


@tasks_app.command("list")
def tasks_list(
    ctx: typer.Context,
    modality: Optional[str] = typer.Option(None, "--modality", "-m", help="Filter by modality, e.g. CT."),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter by source, e.g. ts."),
) -> None:
    """List task names in the catalog."""
    catalog = _catalog(ctx.obj)
    names = catalog.by_modality(modality) if modality else catalog.names(source=source)
    if not names:
        typer.echo("(no tasks)")
        return
    for n in names:
        typer.echo(n)


@tasks_app.command("show")
def tasks_show(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="Task name (bare or 'source:name')."),
) -> None:
    """Show a task's recipe (shape, weights, label map)."""
    from .catalog import AmbiguousTaskError

    catalog = _catalog(ctx.obj)
    try:
        spec = catalog.get(task)
    except (KeyError, AmbiguousTaskError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    typer.echo(f"name     : {spec.qualified_name}")
    typer.echo(f"modality : {spec.modality}")
    typer.echo(f"shape    : {spec.shape}")
    if spec.shape == "single":
        typer.echo(f"weights  : {spec.single}")
    elif spec.shape == "cascade":
        for i, step in enumerate(spec.cascade):
            ref = step.weights_id if step.weights_id is not None else f"<task:{step.crop_from_task}>"
            typer.echo(f"  stage {i}: {ref}  crop={step.crop_to_classes} dilation={step.dilation_mm}mm")
    elif spec.shape == "label_union":
        for part in spec.union:
            typer.echo(f"  part {part.name}: weights={part.weights_id} remap={part.label_remap}")
    if spec.label_map:
        typer.echo(f"labels   : {dict(spec.label_map)}")


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@models_app.command("list")
def models_list(ctx: typer.Context) -> None:
    """List downloaded model ids for the ecosystem."""
    store = _store(ctx.obj)
    ids = store.downloaded()
    if not ids:
        typer.echo("(none downloaded)")
        return
    for i in ids:
        typer.echo(str(i))


@models_app.command("download")
def models_download(
    ctx: typer.Context,
    ids: list[str] = typer.Argument(..., help="Model ids to ensure present (e.g. 297 291)."),
    force: bool = typer.Option(False, "--force", help="Re-fetch even if already present."),
) -> None:
    """Ensure model ids are downloaded (idempotent; --force re-fetches).

    Fetches only what's missing. Remote fetch must be configured for the
    ecosystem; otherwise this reports what to do (e.g. run the upstream
    downloader) for any missing id.
    """
    store = _store(ctx.obj)
    parsed = [int(i) if i.isdigit() else i for i in ids]
    try:
        fetched = store.download(parsed, force=force)
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    typer.echo(f"fetched: {fetched}" if fetched else "all present (nothing to fetch)")


@models_app.command("loaded")
def models_loaded(ctx: typer.Context) -> None:
    """Show currently resident (loaded) models and total memory."""
    store = _store(ctx.obj)
    loaded = store.loaded()
    if not loaded:
        typer.echo("(none loaded)")
        return
    for entry in loaded:
        typer.echo(str(entry))
    typer.echo(f"total: {store.loaded_mb:.0f} MB")


if __name__ == "__main__":  # pragma: no cover
    app()
