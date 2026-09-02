"""Engines: the runtimes that actually turn an image into a labelmap.

**Ecosystem vs engine** - the two words are not synonyms, and keeping them apart
is what lets a family be added without touching the scheduler:

- An **ecosystem** (:mod:`nnseg.ecosystems`) is a *catalog*: the thing a user
  selects, with the ``eco:task@version`` grammar. It owns task names, weight
  installation, and describe.
- An **engine** (:mod:`nnseg.engines.registry`) is a *runtime*: a container
  image, a compute entry point, and a weights identity. It owns nothing the user
  names.

**Many ecosystems map to one engine.** ``ts``, ``moose``, ``mrsegmentator`` and ``custom`` are four
catalogs of nnU-Net models, all run by the ``nnunetv2`` engine. FastSurfer and
SynthStrip each bring a catalog *and* an engine, because their networks are not
nnU-Net at all.

The scheduler (``modal_app._execute_job``) is engine-agnostic: queue, cache,
prefetch, artifacts, cancellation and publication are the same for every family.
An engine supplies only ``_compute`` (plus ``_prepare``/``_ensure`` when it has
weights to install), and declares its image on its worker class.

**Adding an engine** is a row in :mod:`~nnseg.engines.registry` (name, enable
flag, weights identity), a module here with the compute, an ecosystem that names
the engine, and a worker class with the image. Dispatch, cache keys, describe,
env gating and knob forwarding are all derived from the registry - none of them
need a new branch.

Every engine's heavy dependency (FastSurfer's torch stack, synthstrip-torch) is
imported lazily inside the compute path, so importing nnseg never requires it -
see ``docs/dependency-discipline.md``.
"""
