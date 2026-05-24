# Examples

Runnable scripts demonstrating the main usage patterns. Each example is
self-contained — read the docstring at the top for prerequisites.

| Script | What it shows | Extras needed |
|---|---|---|
| [`01_single_volume.py`](01_single_volume.py) | NIfTI in, NIfTI out — the minimum viable inference call | none |
| [`02_batch_with_cache.py`](02_batch_with_cache.py) | Process a folder of CTs reusing one cached engine across all of them | none |
| [`03_path_b_resampling.py`](03_path_b_resampling.py) | Full path-B pipeline via `predict_with_resampling` (SITK in, SITK out, slab-streamed inverse, optional cc3d cleanup) | `[preprocessing]` `[postprocessing]` |
| [`04_two_stage_cascade.py`](04_two_stage_cascade.py) | MOOSE-style cascade: body detector → high-res organ-specific model, FOV-limited | `[preprocessing]` |
| [`05_layered_engine.py`](05_layered_engine.py) | Manual composition of `Predictor` / `SlidingWindowEngine` / `FoldEnsemble` for callers that need fine control (nnInteractive-style single-patch use, custom ensembling) | none |

## Running them

```bash
# From the repo root, with extras installed:
pip install '.[preprocessing,postprocessing]'

python examples/01_single_volume.py /path/to/scan.nii.gz /path/to/seg.nii.gz
```

## Where do the weights come from?

All examples assume you've already downloaded the model you want to use. The
`WeightsLayout` registry auto-discovers weights from:

- `$nnUNet_results` (vanilla nnU-Net convention)
- `$TOTALSEG_WEIGHTS_PATH` (TotalSegmentator env var)
- `~/.totalsegmentator/nnunet/results` (TotalSegmentator default install path)

To download TotalSegmentator weights without invoking the full TS pipeline,
run the TS CLI once with any small input — it downloads on first use:

```bash
TotalSegmentator -i scan.nii.gz -o /tmp/out --fast    # downloads task 297
TotalSegmentator -i scan.nii.gz -o /tmp/out           # downloads tasks 291-295
```

Then the examples can load by task ID:

```python
engine = cached_engine_from_task(297, folds=0)
```
