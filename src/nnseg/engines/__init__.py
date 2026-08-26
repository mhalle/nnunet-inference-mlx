"""Non-nnU-Net inference engines behind the same job protocol.

An *engine* turns an input image into a labelmap; the queue, cache, artifacts,
and client are engine-agnostic (see docs/slicer-modal-design.md, "FastSurfer
engine"). nnU-Net models are the default engine (nnseg.segmenter); this package
holds the others. Each engine's heavy dependency (e.g. FastSurfer's own torch
stack) is lazy-imported so importing nnseg never requires it.
"""
