import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from moto.src import mma as _mma


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "examples" / "1_validation_alt.ipynb"
CHECKPOINT_PATH = REPO_ROOT / "examples" / "checkpoints" / "1_validation_alt_step50.npz"


def _load_notebook_checkpoint_helpers():
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    sources = []
    required = (
        "def save_step50_checkpoint",
        "def load_step50_checkpoint",
        "def thickness_violation_values",
        "def assert_nonincreasing_thickness_violation",
    )
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if any(name in source for name in required):
            sources.append(source)
    namespace = {"np": np, "Path": Path, "_mma": _mma}
    exec("\n\n".join(sources), namespace)
    for name in required:
        func_name = name.removeprefix("def ")
        if func_name not in namespace:
            raise AssertionError(f"{func_name} is missing from 1_validation_alt")
    return namespace


class Step50ThicknessRestartCheckpointTest(unittest.TestCase):
    def test_checkpoint_helpers_round_trip_mma_state(self):
        helpers = _load_notebook_checkpoint_helpers()
        design = np.full((4, 1), 0.5)
        params = _mma.MMAParams(
            max_iter=50,
            kkt_tol=1.0e-7,
            step_tol=1.0e-7,
            move_limit=1.0e-2,
            num_design_var=4,
            num_cons=2,
            lower_bound=np.zeros((4, 1)),
            upper_bound=np.ones((4, 1)),
        )
        state = _mma.init_mma(design, params)
        state.epoch = 50

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            helpers["save_step50_checkpoint"](
                path,
                state,
                max_vol_frac=0.5,
                constraint_mode="volume_and_thickness",
                thickness_start_epoch=50,
                convg_history={"volfrac_cons": [0.1], "thickness_cons": [0.2]},
            )
            loaded = helpers["load_step50_checkpoint"](path)

        restored = _mma.MMAState.from_array(
            loaded["mma_state_array"], int(loaded["num_design_var"])
        )
        self.assertEqual(restored.epoch, 50)
        np.testing.assert_allclose(restored.x, state.x)
        self.assertAlmostEqual(float(loaded["G_vol"]), 0.1)
        self.assertEqual(loaded["constraint_mode"].item(), "volume_and_thickness")

    def test_restart_from_step50_decreases_thickness_violation(self):
        if not CHECKPOINT_PATH.exists():
            self.skipTest(f"checkpoint fixture is missing: {CHECKPOINT_PATH}")
        helpers = _load_notebook_checkpoint_helpers()
        loaded = helpers["load_step50_checkpoint"](CHECKPOINT_PATH)
        if "restart_thickness_cons" not in loaded:
            self.skipTest("checkpoint does not contain restart_thickness_cons")

        violation = helpers["thickness_violation_values"](
            loaded["restart_thickness_cons"]
        )
        helpers["assert_nonincreasing_thickness_violation"](violation)
        self.assertTrue(np.all(np.isfinite(violation)))


if __name__ == "__main__":
    unittest.main()
