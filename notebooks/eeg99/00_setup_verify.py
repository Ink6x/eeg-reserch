"""Phase 0 environment check: verify eeg99 imports and core deps are available.

Run with:
    python notebooks/eeg99/00_setup_verify.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable

# Make `src/` importable when running the script directly from notebooks/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


REQUIRED_THIRD_PARTY: tuple[str, ...] = (
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "yaml",
    "matplotlib",
)

OPTIONAL_THIRD_PARTY: tuple[str, ...] = (
    "torch",
    "lightgbm",
    "pyriemann",
    "mne",
)


def _check(modules: Iterable[str], required: bool) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for name in modules:
        try:
            importlib.import_module(name)
            status[name] = True
        except ImportError:
            status[name] = False
            if required:
                print(f"  [MISSING-REQUIRED] {name}", file=sys.stderr)
            else:
                print(f"  [missing-optional] {name}")
    return status


def main() -> int:
    print("=" * 60)
    print(" eeg99 Phase 0 setup verification")
    print("=" * 60)

    print(f"\nPython: {sys.version.splitlines()[0]}")

    print("\n[1/3] eeg99 package import...")
    try:
        eeg99 = importlib.import_module("eeg99")
        print(f"  ok  eeg99 v{eeg99.__version__}")
    except ImportError as exc:
        print(f"  FAIL  {exc}", file=sys.stderr)
        return 1

    print("\n[2/3] Required third-party packages...")
    req = _check(REQUIRED_THIRD_PARTY, required=True)
    n_req_missing = sum(1 for ok in req.values() if not ok)

    print("\n[3/3] Optional packages (needed in later phases)...")
    _check(OPTIONAL_THIRD_PARTY, required=False)

    print("\n" + "-" * 60)
    if n_req_missing == 0:
        print(" Phase 0 setup OK.")
        return 0
    print(f" Phase 0 setup INCOMPLETE ({n_req_missing} required deps missing).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
