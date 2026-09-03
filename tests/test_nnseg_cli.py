"""The CLI hands pipeline.segment the kwargs it actually accepts.

Regression for 2026-08-24: the WeightsStore work renamed segment()'s weights-location
parameter to ``weights=``, but the CLI kept passing ``model_root=`` - unconditionally,
so every ``nnseg segment`` invocation raised TypeError. Nothing exercised the handler,
which is how it survived. This pins the wiring with a captured fake, no real model.
"""
import inspect
import subprocess
import sys

from nnseg import cli, pipeline


class FakeResult:
    def save(self, path):
        return path


def test_segment_cli_kwargs_are_accepted_by_pipeline(monkeypatch, tmp_path):
    real_params = set(inspect.signature(pipeline.segment).parameters)
    captured = {}

    def fake_segment(image, task, **kw):
        captured.update(kw, image=image, task=task)
        return FakeResult()

    monkeypatch.setattr(pipeline, "segment", fake_segment)
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "total_fast",
                   "-o", str(tmp_path / "out.nii.gz"),
                   "--model-root", str(tmp_path / "weights"), "--quiet"])
    assert rc == 0
    assert captured["task"] == "total_fast"
    assert captured["weights"] == str(tmp_path / "weights")
    assert "model_root" not in captured

    # every kwarg the CLI passes must exist in the real signature, so a future
    # rename cannot silently break the handler again
    unknown = set(captured) - real_params - {"image", "task"}
    assert not unknown, f"CLI passes kwargs segment() does not accept: {sorted(unknown)}"


def test_nnseg_errors_are_one_line_not_a_traceback(monkeypatch, tmp_path, capsys):
    """An outsider running `nnseg serve` without the serve extra got a raw ModuleNotFoundError
    traceback (2026-09-02) although main_serve raises a worded InputError - nothing caught it.
    Every NnsegError now ends as `nnseg: <message>` on stderr with status 2."""
    from nnseg.errors import InputError

    def fake_segment(image, task, **kw):
        raise InputError("the server needs the serve extra")

    monkeypatch.setattr(pipeline, "segment", fake_segment)
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "total_fast",
                   "-o", str(tmp_path / "out.nii.gz"), "--quiet"])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.strip() == "nnseg: the server needs the serve extra"
    assert "Traceback" not in err


def test_tasks_lists_the_catalog_without_torch(tmp_path):
    """`nnseg tasks` is the local answer to `nnseg remote tasks`: every ecosystem's tasks,
    from the catalog alone - no weights, no torch (the describe-only front end stays light)."""
    code = (
        "import sys, nnseg.cli as c\n"
        f"rc = c.main(['tasks', '--model-root', {str(tmp_path)!r}])\n"
        "assert rc == 0, rc\n"
        "assert 'torch' not in sys.modules, 'nnseg tasks imported torch'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    names = [line.split()[0] for line in r.stdout.splitlines() if line.strip()]
    assert "ts:total_fast" in names and "mrsegmentator:base" in names and "moose:clin_ct_body" in names
    # an empty weights root: no nnU-Net task is installed, whatever "materialized" says about
    # its spec (TS specs ship in the catalog, so materialized is always True there)
    assert not [line for line in r.stdout.splitlines()
                if line.startswith(("ts:", "moose:", "mrsegmentator:")) and line.endswith("installed")]
    r2 = subprocess.run([sys.executable, "-c", code.replace("'tasks', ", "'tasks', '--installed', ")],
                        capture_output=True, text=True, timeout=120)
    assert r2.returncode == 0, r2.stderr
    assert not [line for line in r2.stdout.splitlines() if line.startswith("ts:")]
