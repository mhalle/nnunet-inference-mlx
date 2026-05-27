#!/usr/bin/env python
"""Generate src/nnunet_inference_mlx/data/ts_tasks.json from an installed TotalSegmentator.

Strategy
--------
TS's task dispatch lives in ``totalsegmentator.python_api.totalsegmentator``.
The first ~500 lines of that function are a flat ``if task == ... / elif`` chain
where each branch is pure variable assignment::

    elif task == "lung_vessels":
        task_id = 117
        resample = [0.703125, 0.703125, 1.0]
        trainer = "nnUNetTrainerSkeletonRecall"
        crop = ["lung_upper_lobe_left", "lung_lower_lobe_left", ...]
        robust_crop = True

We extract that chain via AST (just to find its line span), then ``exec()`` it
in a controlled namespace with ``task`` / ``fast`` / ``fastest`` pre-set to each
combination we care about. Python's own execution handles every syntactic edge
case — nested if/elif (the ``total`` branch's fast/fastest split), conditional
expressions, complex right-hand sides. We just read the resulting locals.

Compared to a pure AST walk, this is:

  * **More robust** to TS internal refactors (a switch to match/case, helper
    extraction, etc. — as long as the branches still assign the same locals)
  * **Less code** (~150 lines vs ~300)
  * **Closer to ground truth** — whatever TS does to set ``task_id``, that's
    what we capture

The audit confirming this is safe lives at the top of ``CHANGELOG.md`` for
0.9.1.x; see also ``docs/post-0.8.2-roadmap.md``.

Usage
-----
Run inside an environment that has TotalSegmentator installed::

    uv pip install --python /tmp/ts_audit/bin/python totalsegmentator
    /tmp/ts_audit/bin/python scripts/refresh_ts_registry.py > \\
        src/nnunet_inference_mlx/data/ts_tasks.json
    git diff src/nnunet_inference_mlx/data/ts_tasks.json
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import textwrap
from datetime import date
from importlib.metadata import version
from typing import Any, Iterable


# --------------------------------------------------------------------------
# Stage 1: Extract the dispatch block as source
# --------------------------------------------------------------------------


def _is_task_eq_test(node: ast.AST) -> bool:
    """True if ``node`` is the test of an ``if task == "...":`` statement."""
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "task"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
    )


def extract_dispatch_block(fn_source: str) -> str:
    """Locate and return the source for the if/elif chain on ``task``.

    Uses AST only to find the line span of the chain — the returned string
    is dedented so it can be exec'd as top-level code.
    """
    tree = ast.parse(fn_source)
    fn = tree.body[0]
    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)), (
        f"expected def, got {type(fn).__name__}"
    )

    # Walk the function body top-down; find the first `if` whose test is
    # `task == "..."`. That if-statement's full extent (including all
    # chained elif/else) is the dispatch block we want.
    dispatch: ast.If | None = None
    for stmt in fn.body:
        if isinstance(stmt, ast.If) and _is_task_eq_test(stmt.test):
            dispatch = stmt
            break
    if dispatch is None:
        raise RuntimeError(
            "Could not find `if task == \"...\":` block in totalsegmentator() "
            "— TS source layout has likely changed; the generator needs an update."
        )

    # dispatch.end_lineno includes the final elif branch's body. The line
    # numbers are 1-indexed relative to the function source.
    lines = fn_source.split("\n")
    block = "\n".join(lines[dispatch.lineno - 1 : dispatch.end_lineno])
    return textwrap.dedent(block)


# --------------------------------------------------------------------------
# Stage 2: Probe each (task_name, fast, fastest) combination
# --------------------------------------------------------------------------


# Output vars we read from the namespace after exec. Set in branches; we
# pre-populate with None so AST-static branches that omit a field still
# yield a defined value.
_OUTPUT_FIELDS = (
    "task_id", "resample", "trainer", "crop", "crop_addon",
    "model", "folds", "robust_crop", "crop_model",
)

# Function-parameter defaults the dispatch block may reference but won't
# set. Pre-populated so exec doesn't NameError.
_INPUT_DEFAULTS = {
    "quiet": True,        # silences `if not quiet: print(...)` calls
    "verbose": False,
    "roi_subset": None,
    "roi_subset_robust": None,
    "remove_outside": None,
    "ml": False,
    "cascade": False,
    "crop_model": None,
    "body_seg": False,
    "force_split": False,
    "nora_tag": "None",
    "preview": False,
    "statistics": False,
    "radiomics": False,
    "test": 0,
}


def _noop(*args, **kwargs):
    """No-op callable substituted for TS module-level side-effect helpers."""
    return None


# Module-level callables that may appear inside the dispatch chain. We stub
# them out so the dispatch's variable assignments still run but side effects
# (license check, telemetry, download) are skipped. If TS adds a new
# callable to a branch, the resulting NameError surfaces with a clear
# instruction to extend this list.
_TS_CALLABLE_STUBS = {
    name: _noop for name in (
        "show_license_info",
        "has_valid_license_offline",
        "download_pretrained_weights",
        "send_usage_stats",
        "set_license_number",
        "set_config_key",
        "get_config_key",
        "increase_prediction_counter",
        "setup_nnunet",
        "setup_totalseg",
        "convert_device_to_cuda",
        "convert_device_to_string",
        "select_device",
        "validate_device_type_api",
    )
}


def probe_branch(
    dispatch_src: str,
    task: str,
    fast: bool,
    fastest: bool,
    modality: str,
) -> dict[str, Any] | None:
    """Execute the dispatch block with given inputs; return the resulting locals.

    Returns ``None`` if no branch matched (no ``task_id`` was assigned) or if
    the branch raised ``ValueError`` (TS does this for invalid flag combos
    on legacy tasks). Other exceptions propagate — they indicate a real bug
    in this generator.
    """
    ns: dict[str, Any] = {
        # Inputs
        "task": task, "fast": fast, "fastest": fastest, "modality": modality,
        # Output slots (None until a branch sets them)
        **{field: None for field in _OUTPUT_FIELDS},
        # Defaults for function-params the branches reference
        **_INPUT_DEFAULTS,
        # Sentinel default for `model` (most branches don't override)
        "model": "3d_fullres",
        # Stubs for TS module-level helpers called inside dispatch
        **_TS_CALLABLE_STUBS,
    }
    try:
        exec(dispatch_src, ns)  # noqa: S102 — controlled source
    except ValueError:
        # Branch deliberately rejected this combination (e.g. legacy
        # task that doesn't support fast).
        return None
    except NameError as e:
        # An unstubbed TS reference. Surface with a clear fix instruction.
        raise RuntimeError(
            f"probe_branch({task!r}, fast={fast}, fastest={fastest}): "
            f"branch references undefined name: {e}. "
            f"Add it to _TS_CALLABLE_STUBS or _INPUT_DEFAULTS in {__file__}."
        ) from e

    if ns["task_id"] is None:
        # No branch matched (`task` isn't in the dispatch chain).
        return None

    return {f: ns[f] for f in _OUTPUT_FIELDS}


def probe_task_variants(
    dispatch_src: str, task: str, modality: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Probe ``task`` with all (fast, fastest) combinations.

    De-duplicates: if the branch ignores ``fast``/``fastest`` (most do),
    the same params come out and we keep just the default-flag variant.
    If the branch responds to flags (the ``total`` branch), we emit
    distinct ``_fast``/``_fastest`` task names.

    Returns a list of ``(variant_name, params)`` pairs.
    """
    base = probe_branch(dispatch_src, task, False, False, modality)
    if base is None:
        return []

    variants = [(task, base)]

    fast_params = probe_branch(dispatch_src, task, True, False, modality)
    if fast_params is not None and fast_params != base:
        variants.append((f"{task}_fast", fast_params))

    fastest_params = probe_branch(dispatch_src, task, False, True, modality)
    if fastest_params is not None and fastest_params != base:
        variants.append((f"{task}_fastest", fastest_params))

    return variants


# --------------------------------------------------------------------------
# Stage 3: Convert probed params into TaskSpec-shaped dicts
# --------------------------------------------------------------------------


def _modality_from_task(task: str) -> str:
    """Match TS's convention: tasks ending ``_mr`` are MR, everything else CT.

    TS doesn't currently have PET models — would be a separate convention
    if/when it adds them.
    """
    return "MR" if task.endswith("_mr") else "CT"


def _rough_cropper_dataset_id(crop: list[str], robust_crop: bool, modality: str) -> int:
    """Pick the rough-segmentation dataset ID TS would use for this crop.

    Mirrors the ``crop_model_task = ...`` logic in
    ``python_api.totalsegmentator`` lines ~625-648 of v2.13.0.
    """
    crop_names = set(crop or ())
    # Body model (300) is used when the crop list includes body trunk/extremities
    if "body_trunc" in crop_names or "body_extremities" in crop_names:
        return 300
    if modality == "MR":
        return 852          # MR total, 3mm (no 6mm option for MR)
    if robust_crop:
        return 297          # CT total, 3mm (robust)
    return 298              # CT total, 6mm (default speed)


def _resolve_crop_class_ids(
    crop_names: list[str],
    rough_dataset_id: int,
    class_map: dict[str, dict[int, str]],
) -> tuple[int, ...]:
    """Map ``crop_names`` (anatomical names) to integer class IDs in the
    rough model's label space."""
    # The rough model uses the "total" (CT) or "total_mr" (MR) or "body"
    # label space, depending on which dataset.
    rough_task_name = {
        297: "total",
        298: "total",
        852: "total_mr",
        300: "body",
    }[rough_dataset_id]
    inv = {name: cid for cid, name in class_map[rough_task_name].items()}
    ids: list[int] = []
    for name in crop_names:
        if name not in inv:
            raise KeyError(
                f"Crop class name {name!r} not in class_map[{rough_task_name!r}]. "
                f"Available: {sorted(inv)[:10]}..."
            )
        ids.append(inv[name])
    return tuple(ids)


def _label_remap_for_union_part(
    part_dataset_id: int,
    class_map: dict[str, dict[int, str]],
    class_map_5_parts: dict[str, dict[int, str]],
    taskid_to_partname: dict[int, str],
    unified_task_name: str,
) -> dict[int, int]:
    """Build the {part_local_id: unified_id} remap for one part of a union.

    Mirrors ``nnunet.py:246-247``:
        for jdx, class_name in class_map_parts[map_taskid_to_partname[tid]].items():
            seg_combined[img_part][seg == jdx] = class_map_inv[class_name]
    """
    part_name = taskid_to_partname[part_dataset_id]
    part_classes = class_map_5_parts[part_name]   # {part_local_id: class_name}
    unified_inv = {name: uid for uid, name in class_map[unified_task_name].items()}
    remap: dict[int, int] = {}
    for part_local_id, class_name in part_classes.items():
        if class_name not in unified_inv:
            # Some part class names aren't in the unified total — skip silently
            # (matches TS's `class_map_inv.get(...)` defensive pattern).
            continue
        remap[part_local_id] = unified_inv[class_name]
    return remap


def _build_label_map(
    task_name: str, class_map: dict[str, dict[int, str]],
) -> dict[int, str]:
    """Return ``class_map[task_name]`` if present, else empty dict.

    For label_union tasks named ``total`` / ``total_mr`` etc., this returns
    the unified 117-class (CT) or 50-class (MR) namespace — exactly what
    the union output IDs land in.
    """
    return dict(class_map.get(task_name, {}))


def params_to_taskspec(
    variant_name: str,
    params: dict[str, Any],
    class_map: dict[str, dict[int, str]],
    class_map_5_parts: dict[str, dict[int, str]],
    map_taskid_to_partname_ct: dict[int, str],
    map_taskid_to_partname_mr: dict[int, str],
) -> dict[str, Any] | None:
    """Convert probed dispatch params into a TaskSpec-shaped dict (JSON-ready).

    Returns ``None`` if the task can't be modeled (e.g. its task_id list
    spans unknown parts). Errors that indicate a generator bug raise.
    """
    task_id = params["task_id"]
    crop = params["crop"]
    # Strip our synthesized fast/fastest suffix before detecting modality
    # or looking up the label_map (TS's class_map uses the base name).
    base_task_name = (
        variant_name
        .removesuffix("_fastest")
        .removesuffix("_fast")
    )
    modality = _modality_from_task(base_task_name)

    # ----- label_union -----
    if isinstance(task_id, list):
        # Determine which partname map applies based on what task IDs are in
        # the list. We assume a homogeneous list (TS doesn't mix modalities).
        if all(tid in map_taskid_to_partname_ct for tid in task_id):
            partname_map = map_taskid_to_partname_ct
            unified_task = base_task_name        # e.g. "total"
        elif all(tid in map_taskid_to_partname_mr for tid in task_id):
            partname_map = map_taskid_to_partname_mr
            unified_task = base_task_name        # e.g. "total_mr"
        else:
            # Unknown union — skip (e.g. the headneck_muscles 778/779 union
            # which isn't a TS public task name).
            return None

        union = []
        for tid in task_id:
            remap = _label_remap_for_union_part(
                tid, class_map, class_map_5_parts, partname_map, unified_task,
            )
            if not remap:
                continue
            union.append({
                "weights_id": tid,
                "label_remap": {str(k): v for k, v in remap.items()},
                "name": partname_map[tid].replace("class_map_part_", ""),
            })
        if not union:
            return None

        return {
            "name": variant_name,
            "source": "ts",
            "modality": modality,
            "shape": "label_union",
            "union": union,
            "label_map": {str(k): v for k, v in _build_label_map(unified_task, class_map).items()},
        }

    # ----- single -----
    if crop is None:
        spec: dict[str, Any] = {
            "name": variant_name,
            "source": "ts",
            "modality": modality,
            "shape": "single",
            "single": int(task_id),
        }
        lm = _build_label_map(base_task_name, class_map)
        if lm:
            spec["label_map"] = {str(k): v for k, v in lm.items()}
        return spec

    # ----- cascade -----
    if params.get("crop_model") is not None:
        # TS uses a custom rough-segmentation task (recursive totalsegmentator
        # call). Our flat CascadeStep can't express a task reference yet —
        # future TaskSpec extension. Skip for now with a clear note.
        print(
            f"  [skip] {variant_name}: uses crop_model={params['crop_model']!r} "
            "(nested-task cascade not yet supported in TaskSpec)",
            file=sys.stderr,
        )
        return None

    rough_id = _rough_cropper_dataset_id(crop, params.get("robust_crop", False), modality)
    try:
        crop_ids = _resolve_crop_class_ids(crop, rough_id, class_map)
    except KeyError as e:
        # A crop name didn't resolve — log and skip rather than fail the
        # whole generator. The maintainer can investigate from the warning.
        print(f"  [warn] {variant_name}: {e}", file=sys.stderr)
        return None

    dilation_mm = 10.0
    if params.get("crop_addon"):
        # TS uses [x, y, z] mm tuples; we apply a single isotropic margin.
        # Use the max — matches our `Stage.dilation_mm` semantics.
        dilation_mm = float(max(params["crop_addon"]))

    spec = {
        "name": variant_name,
        "source": "ts",
        "modality": modality,
        "shape": "cascade",
        "cascade": [
            {
                "weights_id": rough_id,
                "crop_to_classes": list(crop_ids),
                "dilation_mm": dilation_mm,
            },
            {
                "weights_id": int(task_id),
                "crop_to_classes": None,
                "dilation_mm": 10.0,
            },
        ],
    }
    lm = _build_label_map(base_task_name, class_map)
    if lm:
        spec["label_map"] = {str(k): v for k, v in lm.items()}
    return spec


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(out_stream=sys.stdout) -> None:
    try:
        import totalsegmentator.python_api as ts_api
        from totalsegmentator.map_to_binary import (
            class_map, class_map_5_parts,
            map_taskid_to_partname_ct, map_taskid_to_partname_mr,
        )
    except ImportError as e:
        sys.stderr.write(
            "ERROR: totalsegmentator is not installed in this Python.\n"
            f"  ({e})\n\n"
            "Install with: uv pip install --python <python> totalsegmentator\n"
        )
        sys.exit(1)

    ts_version = version("totalsegmentator")
    print(f"  TS version: {ts_version}", file=sys.stderr)
    print(f"  class_map: {len(class_map)} task names", file=sys.stderr)

    fn_src = inspect.getsource(ts_api.totalsegmentator)
    dispatch_src = extract_dispatch_block(fn_src)
    print(f"  dispatch block: {dispatch_src.count(chr(10))} lines", file=sys.stderr)

    tasks_json: list[dict] = []
    skipped: list[str] = []

    for task_name in sorted(class_map):
        modality = _modality_from_task(task_name)
        variants = probe_task_variants(dispatch_src, task_name, modality)
        if not variants:
            skipped.append(task_name)
            continue
        for variant_name, params in variants:
            spec = params_to_taskspec(
                variant_name, params,
                class_map, class_map_5_parts,
                map_taskid_to_partname_ct, map_taskid_to_partname_mr,
            )
            if spec is not None:
                tasks_json.append(spec)

    print(
        f"  emitted: {len(tasks_json)} task specs "
        f"({sum(1 for t in tasks_json if t['shape'] == 'single')} single, "
        f"{sum(1 for t in tasks_json if t['shape'] == 'cascade')} cascade, "
        f"{sum(1 for t in tasks_json if t['shape'] == 'label_union')} label_union)",
        file=sys.stderr,
    )
    if skipped:
        print(
            f"  skipped (no dispatch branch): {len(skipped)}: "
            f"{', '.join(skipped[:8])}{'...' if len(skipped) > 8 else ''}",
            file=sys.stderr,
        )

    output = {
        "_meta": {
            "schema_version": 1,
            "ts_version": ts_version,
            "generated": str(date.today()),
            "generator": "scripts/refresh_ts_registry.py",
            "generator_version": 1,
            "task_count": len(tasks_json),
        },
        "tasks": tasks_json,
    }
    json.dump(output, out_stream, indent=2)
    out_stream.write("\n")


if __name__ == "__main__":
    main()
