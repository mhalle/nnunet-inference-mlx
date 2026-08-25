"""The kernel layer must stay a leaf.

``nnseg`` holds two layers in one package. The kernel modules know nothing about tasks, plans,
weights or files - torch and numpy only - so they could be lifted into their own package the
day something outside wants them (nnU-Net's export path, TotalSegmentator, MOOSE, a CUDA user
who only needs the fused restore). That property is easy to lose by accident and cheap to
check, so it is checked here rather than trusted.
"""
import ast
import pathlib
import unittest

def _package_dir() -> pathlib.Path:
    """Locate the package by import, not by repo layout - the tests also run against a copy
    shipped into a container (cuda/), where there is no src/ directory."""
    try:
        import nnseg
        return pathlib.Path(nnseg.__file__).resolve().parent
    except Exception:
        return pathlib.Path(__file__).resolve().parent.parent / "src" / "nnseg"


SRC = _package_dir()

KERNEL = {"grid", "mapping", "tables", "restore", "resample", "reference", "shuffleup",
          "backends", "backends.metal", "backends.torch_gather", "backends.triton_gpu"}
PIPELINE = {"io", "preprocess", "frame", "network", "pipeline", "cli", "tasks", "values", "envelope",
            "weights_fetch", "trainers", "result", "cache", "segmenter", "weights", "progress", "job",
            "serve", "client", "modal_app", "sources", "ecosystems"}
# errors.py is deliberately dependency-free (stdlib only) so either layer may raise from it.
SHARED = {"errors"}
FORBIDDEN_FOR_KERNEL = {"nnunetv2", "SimpleITK", "nibabel", "scipy", "mlx", "totalsegmentator",
                        "nnunet_inference_mlx", "acvl_utils", "batchgenerators"}
# nnseg must import on a machine with no mlx - that is the whole point of the torch path, and
# depending on the MLX toolkit's value types once made it unimportable on Linux.
FORBIDDEN_EVERYWHERE = {"mlx", "nnunet_inference_mlx"}
# scipy is allowed at call time inside resample (the identity-probe operators) but must not be a
# module-level import, so the kernel layer stays importable without it.
SCIPY_OK_AT_CALL_TIME = {"resample"}


def _module_path(name: str) -> pathlib.Path:
    return SRC / (name.replace(".", "/") + ".py") if name != "backends" else SRC / "backends" / "__init__.py"


def _imports(path: pathlib.Path, top_level_only: bool):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if top_level_only and getattr(node, "col_offset", 0) != 0:
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative import within nnseg
                yield "." * node.level + (node.module or ""), node.lineno
            elif node.module:
                yield node.module.split(".")[0], node.lineno


class TestLayering(unittest.TestCase):
    def test_every_module_is_classified(self):
        found = {p.stem for p in SRC.glob("*.py") if p.stem not in ("__init__", "__main__")}
        found |= {f"backends.{p.stem}" for p in (SRC / "backends").glob("*.py") if p.stem != "__init__"}
        found |= {"backends"} if (SRC / "backends").is_dir() else set()
        self.assertEqual(found, KERNEL | PIPELINE | SHARED,
                         "a module was added without deciding which layer it belongs to")

    def test_shared_modules_import_nothing(self):
        """errors.py is classified SHARED because either layer may raise from it - which only
        holds while it stays dependency-free. Enforce that rather than trusting the comment."""
        for name in sorted(SHARED):
            for mod, line in _imports(_module_path(name), top_level_only=False):
                self.assertTrue(mod.startswith("__future__"),
                                f"{name}.py:{line} imports {mod!r}; SHARED modules must stay stdlib-only")

    def test_kernel_modules_do_not_import_the_pipeline_layer(self):
        for name in sorted(KERNEL):
            path = _module_path(name)
            for mod, line in _imports(path, top_level_only=False):
                if mod.startswith("."):
                    target = mod.lstrip(".")
                    if target in PIPELINE:
                        self.fail(f"{name}.py:{line} imports the pipeline module {target!r}")

    def test_kernel_modules_depend_only_on_torch_and_numpy(self):
        for name in sorted(KERNEL):
            path = _module_path(name)
            for mod, line in _imports(path, top_level_only=True):
                if mod in FORBIDDEN_FOR_KERNEL:
                    self.assertIn(name, SCIPY_OK_AT_CALL_TIME if mod == "scipy" else set(),
                                  f"{name}.py:{line} imports {mod!r} at module level; the kernel "
                                  f"layer must stay torch + numpy so it can be extracted")

    def test_no_module_depends_on_the_mlx_toolkit(self):
        for path in sorted(SRC.rglob("*.py")):
            for mod, line in _imports(path, top_level_only=False):
                self.assertNotIn(mod, FORBIDDEN_EVERYWHERE,
                                 f"{path.name}:{line} imports {mod!r}; nnseg must run where mlx does not")

    def test_the_nnseg_tests_do_not_depend_on_the_mlx_toolkit_either(self):
        """The rules above cover src/nnseg. The tests need the same property or CI cannot run
        them on Linux - test_nnseg_frame once imported nnunet_inference_mlx.values.Geometry and
        broke the build."""
        here = pathlib.Path(__file__).resolve().parent
        for path in sorted(list(here.glob("test_nnseg_*.py")) + list(here.glob("kernel_test_*.py"))):
            for mod, line in _imports(path, top_level_only=False):
                self.assertNotIn(mod, FORBIDDEN_EVERYWHERE,
                                 f"{path.name}:{line} imports {mod!r}; the nnseg tests must run "
                                 f"where mlx does not")

    def test_scipy_is_only_a_call_time_dependency(self):
        """resample builds its operators with scipy, but importing nnseg must not need it."""
        for mod, line in _imports(_module_path("resample"), top_level_only=True):
            self.assertNotEqual(mod, "scipy", f"resample.py:{line} imports scipy at module level")


if __name__ == "__main__":
    unittest.main()
