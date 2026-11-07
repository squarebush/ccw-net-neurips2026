import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def run_module():
    def run(module, *args):
        env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
        return subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=REPO, env=env,
                              capture_output=True, text=True)
    return run
