"""The CLI hands pipeline.segment the kwargs it actually accepts.

Regression for 2026-08-24: the WeightsStore work renamed segment()'s weights-location
parameter to ``weights=``, but the CLI kept passing ``model_root=`` - unconditionally,
so every ``nnseg segment`` invocation raised TypeError. Nothing exercised the handler,
which is how it survived. This pins the wiring with a captured fake, no real model.
"""
import inspect

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
