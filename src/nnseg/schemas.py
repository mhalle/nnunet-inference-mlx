"""What a task takes, and what a client may send: the wire's typed vocabulary.

Three things live here, all of them about the *boundary* and none about the
numerical core:

* **Parameter schemas** - one pydantic model per engine. Each generates the JSON
  Schema that ``describe()`` publishes *and* performs the validation at submit,
  so the advertised contract and the enforced one cannot drift. A hand-written
  schema sitting beside a hand-written validator is the same class of bug as a
  hand-maintained label list, and this project has been bitten by that shape
  before.
* **The role vocabulary** - a small alias table mapping the spellings models
  actually use (``T1c``, ``t1ce``, ``T1Gd``) onto one canonical name.
* **Input specs** - the per-task declaration of which images a task consumes.

**Why an alias table is safe here.** "The checkpoint is the spec" is a rule about
*authority*, not presentation: what must never be hand-maintained is anything
that can silently disagree with the model - a label list, a channel count, a
binding key. An alias is different. Binding always accepts the model's own
spelling, and a name this table does not know simply has no alias. The failure
mode is silence, not a wrong answer. Without it, every client would carry its own
copy of this mapping, forever.

**Parameters have two owners.** An algorithm's own knobs (VoxTell's ``prompts``)
and our processing knobs (``grid``, ``interp``) share one flat namespace on the
wire - clients send a single dict - but they are *published* as separate groups,
because they are documented by different people and wrong for different reasons.
Where a behavior is not ours to offer at all, it is published as a **fact rather
than a knob**: a MONAI bundle's postprocessing decides its own restore, and
advertising ``interp`` on it would be advertising a knob we do not turn.

Pydantic stops here and at :mod:`nnseg.serve`. It must not reach the value types
(:class:`~nnseg.result.Segmentation`, :class:`~nnseg.grid.Grid`) or the kernel
layer: those are frozen dataclasses on a hot path with no untrusted input, where
validation would be a cost with no reader.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .errors import RequestError

# --------------------------------------------------------------------------
# the role vocabulary
# --------------------------------------------------------------------------

#: Model-declared channel spellings -> our canonical name. Keys are *normalized*
#: (lowercased, punctuation stripped), so one entry covers ``T1-CE``, ``t1_ce``
#: and ``T1 CE``. ADVISORY - see the module docstring.
ROLE_ALIASES: dict[str, str] = {
    "t1": "T1w", "t1w": "T1w", "t1n": "T1w", "t1native": "T1w",
    "t1c": "T1w-ce", "t1ce": "T1w-ce", "t1gd": "T1w-ce", "t1wce": "T1w-ce",
    "t1post": "T1w-ce", "t1contrast": "T1w-ce",
    "t2": "T2w", "t2w": "T2w",
    "flair": "FLAIR", "t2f": "FLAIR", "t2flair": "FLAIR",
    "dwi": "DWI", "adc": "ADC", "swi": "SWI",
    "ct": "CT", "pet": "PET",
}

#: Names that carry no information about *which* image is wanted. A single-input
#: task calls its input "image"; aliasing that to anything would be inventing a
#: meaning the model never declared.
GENERIC_ROLES = frozenset({"image", "input", "data", "channel", "img", "volume"})


def _normalize(name) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def canonical_role(declared) -> str | None:
    """Our canonical name for a model-declared channel spelling, or None.

    None means "no confident mapping" - which is a normal answer, not a failure.
    Callers publish the alias when there is one and stay quiet otherwise; the
    model's own spelling is always the binding key.
    """
    n = _normalize(declared)
    if not n or n in GENERIC_ROLES:
        return None
    return ROLE_ALIASES.get(n)


def roles_match(a, b) -> bool:
    """Whether two role spellings refer to the same input.

    Matches on the normalized declared name first, then on canonical aliases, so
    a client may send ``T1c``, ``t1-ce`` or ``T1w-ce`` for a model that declared
    ``T1c``. Deliberately does NOT fall back to position.
    """
    if _normalize(a) == _normalize(b):
        return True
    ca, cb = canonical_role(a), canonical_role(b)
    return ca is not None and ca == cb


def input_specs(names, *, modality=None, kind: str = "image") -> list[dict]:
    """The ``inputs`` block: one entry per channel the task consumes, in the
    model's own channel order.

    ``channel`` is informational - it tells a client which position the model
    puts this input in, which is exactly the thing that differs between
    otherwise-identical conventions (MONAI's BraTS bundle declares T1c first;
    nnU-Net's BraTS convention puts FLAIR there). Binding is by name precisely
    so that difference cannot silently mis-serve a request.
    """
    out = []
    for i, name in enumerate(names):
        entry = {"name": str(name), "kind": kind, "required": True, "channel": i}
        alias = canonical_role(name)
        if alias:
            entry["alias"] = alias
        if modality:
            entry["modality"] = str(modality)
        out.append(entry)
    return out


def single_input(modality=None, name: str = "image") -> list[dict]:
    """The ``inputs`` block for a task that takes one image - the common case."""
    return input_specs([name], modality=modality)


def channel_names(chan) -> list:
    """A model's input channel names in channel order.

    ``dataset.json`` spells this as a dict keyed by stringified index
    (``{"0": "CT"}``); an ecosystem that already ordered them hands over a list.
    Sorted numerically, so a ten-channel model does not put ``"10"`` between
    ``"1"`` and ``"2"``.
    """
    if isinstance(chan, dict):
        try:
            return [str(v) for _, v in sorted(chan.items(), key=lambda kv: int(kv[0]))]
        except (TypeError, ValueError):
            return [str(v) for v in chan.values()]
    return [str(v) for v in chan] if chan else []


def declared_inputs(desc: dict):
    """The inputs a ``describe()`` payload declares.

    ONE reading of a describe payload, used both by the code that builds it and
    by the code that validates a request against it - so the two can never
    disagree about what a task takes. Three distinct answers, and the difference
    between the last two matters:

    * a list - these are the inputs, bind by name;
    * ``None`` - the model declared its channels and did so incompletely, so
      nothing can be bound (see ``inputs_incomplete``);
    * *absent* - the task never declared anything, which is today's contract:
      one image, or whatever ``channel_names`` says.
    """
    if "inputs" in desc:
        return desc["inputs"]
    names = channel_names(desc.get("channel_names"))
    modality = desc.get("modality")
    return input_specs(names, modality=modality) if names else single_input(modality)


# --------------------------------------------------------------------------
# parameter schemas
# --------------------------------------------------------------------------

class Params(BaseModel):
    """Base for every parameter model.

    ``extra="forbid"`` is the point of the exercise: an option we do not know is
    a typo or a stale client, and today both are accepted in silence and then
    ignored by a worker. Silence is the worst possible answer - the caller
    believes a knob was turned.
    """

    model_config = ConfigDict(extra="forbid")


class NoParams(Params):
    """An engine that takes no parameters of its own. Not the same as "unknown":
    this is a positive statement that the only knobs are the processing ones."""


class VoxTellParams(Params):
    """Free-text prompts naming the structures to segment."""

    # Prompts are *input*, not policy: they hash into the result-cache key the
    # way a source identity does. This docstring is published as the schema's
    # `description`, so it is written for a client, not for us.
    prompts: list[str] = Field(
        ..., min_length=1,
        description="Free-text names of structures to segment; one label per "
                    "prompt, numbered in the order given.")


class ProcessingParams(Params):
    """How nnseg resamples and frames the result around the network."""

    # Deliberately a subset of nnseg.segmenter.POLICY. `device`, `dtype`,
    # `weights`, `accumulate`, `batch_size` and `allow_transpose` are *deployment*
    # policy: they describe the machine the service runs on, not the result the
    # caller asked for, and letting a request pick its own device or weights root
    # on a shared server is a capability, not a convenience.
    grid: Annotated[float, Field(gt=0)] | Literal["input", "model"] | None = Field(
        None, description="Output grid: 'input' (default), 'model' for the "
                          "network's own spacing, or an isotropic size in mm.")
    folds: list[int] | None = Field(
        None, description="Which trained folds to ensemble.")
    configuration: str | None = Field(
        None, description="nnU-Net configuration name, when the model ships more "
                          "than one.")
    envelope_mm: float | None = Field(
        None, description="Crop the network's field of view to this margin around "
                          "the body, in mm.")
    resampling_order: int | None = Field(
        None, ge=0, le=5, description="Spline order for the forward resample.")
    interp: Literal["linear", "nearest"] | None = Field(
        None, description="Interpolation used to restore the result to the "
                          "output grid.")
    convention: Literal["auto", "corner", "center"] | None = Field(
        None, description="Grid-alignment convention; 'auto' follows the model's "
                          "lineage.")


#: Restore facts, keyed by what actually happens rather than by who asked. A
#: *fact*, not a knob: published so a client can see - before choosing a model -
#: whether boundaries came back graded or blocky. The MONAI engine has no entry
#: here because the answer is per bundle and read from the bundle's own config.
GRADED_RESTORE = {
    "restore": {
        "mode": "graded", "owner": "nnseg",
        # deliberately says "discretized", not "argmaxed": one guarantee has to
        # cover nnU-Net's argmax over logits and SynthStrip's threshold over a
        # signed distance transform
        "note": "the network's continuous output is resampled to the output grid "
                "and discretized after, so boundaries are not snapped to the "
                "model's voxel grid",
    }
}


def parameter_groups(engine_params, *, processing: bool = True) -> dict:
    """The ``parameters`` block: JSON Schema per owner.

    ``algorithm`` is the model's own knobs, ``processing`` is ours. Both are real
    JSON Schema, so a client can validate against them with any generic
    validator - and the Slicer client can render a form from them without knowing
    that MONAI or nnU-Net exist.
    """
    out = {"algorithm": (engine_params or NoParams).model_json_schema()}
    out["processing"] = (ProcessingParams.model_json_schema() if processing
                         else NoParams.model_json_schema())
    return out


@lru_cache(maxsize=None)
def wire_params(algorithm: type, processing: bool) -> type:
    """The single flat model a request's ``options`` are validated against.

    The two groups are published separately - they have different owners and
    different people to ask when they are wrong - but the wire stays flat,
    because a client sending one dict is simpler than a client learning where
    each key belongs. Merging them by inheritance also makes a name collision
    between an engine's knob and one of ours a loud error here rather than a
    silent shadowing (see the collision test).
    """
    if not processing:
        return algorithm
    if algorithm is NoParams:
        return ProcessingParams
    return create_model(f"{algorithm.__name__}Wire",
                        __base__=(algorithm, ProcessingParams))


def bind_sources(sources: list, inputs, *, multi_input: bool, task: str = "") -> list:
    """Bind a request's sources to a task's declared inputs, **by name**.

    Returns ``[(role, source), ...]`` in the model's own channel order, so a
    caller may send its sources in any order. Position is never a fallback: the
    same four BraTS files are ordered T1c-first by MONAI's bundle and FLAIR-first
    by nnU-Net's own convention, so a positional wire is guaranteed to
    mis-serve one of them, and mis-serving here looks like a plausible
    segmentation rather than an error.
    """
    what = f"{task!r} " if task else ""
    if inputs is None:
        raise RequestError(
            "not_bindable",
            f"{what}does not declare which images it takes, so nnseg will not "
            "bind them by position; see this task's `inputs_incomplete`")
    declared = [i["name"] for i in inputs]
    if len(declared) == 1:
        if len(sources) != 1:
            raise RequestError("input_count",
                               f"{what}takes 1 input, got {len(sources)}",
                               declared=1, got=len(sources))
        given = sources[0].get("role")
        if given and not roles_match(given, declared[0]):
            raise RequestError("unknown_role", f"unknown input role {given!r}",
                               role=given, declared=declared)
        return [(declared[0], sources[0])]
    if not multi_input:
        raise RequestError(
            "multi_input_unsupported",
            f"{what}takes {len(declared)} inputs ({', '.join(declared)}), which "
            "this engine cannot yet be handed over the wire",
            declared=declared)
    bound: dict[str, dict] = {}
    for entry in sources:
        given = entry.get("role")
        if not given:
            raise RequestError(
                "role_required",
                f"{what}takes {len(declared)} named inputs, so every source must "
                "carry a `role`", declared=declared)
        match = next((d for d in declared if roles_match(given, d)), None)
        if match is None:
            raise RequestError("unknown_role", f"unknown input role {given!r}",
                               role=given, declared=declared)
        if match in bound:
            raise RequestError("duplicate_role",
                               f"input role {match!r} is bound twice", role=match)
        bound[match] = entry
    missing = [d for d in declared if d not in bound]
    if missing:
        raise RequestError("missing_role",
                           f"{what}needs {len(declared)} named inputs; missing: "
                           + ", ".join(missing), missing=missing, declared=declared)
    return [(d, bound[d]) for d in declared]


def validate_options(model, options: dict) -> dict:
    """Validate a request's options against a parameter model.

    Returns the options unchanged on success (the caller keeps sending exactly
    what it sent - this validates, it does not rewrite, because the options dict
    is hashed into the result-cache key and a normalization here would move every
    key). Raises :class:`~nnseg.errors.RequestError` with our own vocabulary.
    """
    try:
        model.model_validate(options)
    except ValidationError as e:
        raise _as_request_error(e, model) from e
    return options


_CODES = {"extra_forbidden": "unknown_parameter", "missing": "missing_parameter"}
#: Headline order. An unknown key almost always IS the root cause - a typo'd
#: `promts` makes the real `prompts` "missing" too, and reporting the missing one
#: sends the caller looking for a field they thought they sent.
_RANK = {"extra_forbidden": 0, "missing": 1}


def _field_of(err) -> str:
    """The parameter an error is about.

    Only the first path element: pydantic tags union branches into ``loc``, so
    the raw path for a bad ``grid`` reads ``grid.constrained-float`` - an
    implementation detail of how we spelled the type, and not a name any client
    can act on.
    """
    loc = err.get("loc") or ()
    return str(loc[0]) if loc else "(root)"


def _as_request_error(exc: ValidationError, model) -> RequestError:
    """Translate pydantic's report into the API's one error vocabulary.

    Every failure is reported, not just the headline one: a caller fixing a
    request one round trip per mistake is the experience this whole introspection
    pass exists to avoid.
    """
    import difflib

    errs = sorted(exc.errors(), key=lambda e: _RANK.get(e.get("type"), 2))
    known = sorted(model.model_fields)
    every = [{"parameter": _field_of(e), "message": e.get("msg")} for e in errs]
    first = errs[0]
    field = _field_of(first)
    code = _CODES.get(first.get("type"), "invalid_parameter")
    if code == "unknown_parameter":
        near = difflib.get_close_matches(field, known, n=1)
        msg = f"unknown parameter {field!r}"
        if near:
            msg += f" (did you mean {near[0]!r}?)"
        return RequestError(code, msg, parameter=field, known=known, errors=every)
    if code == "missing_parameter":
        missing = [_field_of(e) for e in errs if e.get("type") == "missing"]
        return RequestError(code, f"{field!r} is required", missing=missing,
                            known=known, errors=every)
    return RequestError(code, f"{field!r}: {first.get('msg')}", parameter=field,
                        errors=every)
