"""The errors nnseg raises deliberately.

A consumer needs to tell "these weights are not downloaded yet" (retry after fetching) from
"nnseg cannot run this model" (never retry, report it) from "this image is unusable" (the user's
fault) - today that means matching on message text. Each class therefore also inherits the
builtin it replaces, so ``except FileNotFoundError`` keeps working and ``except ModelNotFound``
becomes possible; callers can catch at whichever level suits them.
"""
from __future__ import annotations


class NnsegError(Exception):
    """Base for every error nnseg raises on purpose. Catch this to catch all of them."""


class InputError(NnsegError, ValueError):
    """The image or the arguments are unusable: not 3D, unreadable, contradictory options."""


class ModelNotFound(NnsegError, FileNotFoundError):
    """A model, configuration, fold or weights file is not where it should be.

    Recoverable: fetch the weights (``nnseg.weights_fetch``) or point ``model_root`` elsewhere
    and try again.
    """


class UnsupportedModel(NnsegError, NotImplementedError):
    """A valid nnU-Net model that nnseg cannot run *yet*.

    Not the caller's mistake and not recoverable by retrying - e.g. a non-identity
    ``transpose_forward``, region-based (sigmoid) labels, multi-channel input, or a
    ``3d_cascade_fullres`` configuration. Distinct from :class:`ModelNotFound` so a UI can say
    "this model is not supported" rather than "file missing".
    """


class ResourceError(NnsegError, RuntimeError):
    """Out of device or host memory, after nnseg's own fallbacks were exhausted."""


class Cancelled(NnsegError):
    """The caller's cancel signal fired; the run stopped between patches."""
