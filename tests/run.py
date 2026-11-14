"""Minimal test runner (pytest is not installed locally).

Usage: python tests/run.py
Imports every tests/test_*.py, calls each test_* function, prints PASS/FAIL,
and exits non-zero if any test fails.
"""
import glob
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (ROOT, os.path.join(ROOT, "experiments", "real")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    total = failed = 0
    for path in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for fn_name in sorted(dir(mod)):
            fn = getattr(mod, fn_name)
            if not fn_name.startswith("test_") or not callable(fn):
                continue
            total += 1
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {fn_name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            else:
                print(f"PASS {fn_name}")
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
