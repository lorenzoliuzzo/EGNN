import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent


@pytest.fixture(scope="session")
def legacy():
    """Root-level test.py (legacy standalone QCD pipeline), imported under a
    non-colliding module name so pytest does not treat it as a test module."""
    name = "legacy_standalone"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "test.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
