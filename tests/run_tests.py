"""Run project tests with concise results and captured diagnostics on failure."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import unittest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-output", action="store_true",
                        help="Show prints, RDKit diagnostics and progress bars during tests.")
    args = parser.parse_args()
    app_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(app_root))
    os.chdir(app_root)
    if args.show_output:
        os.environ["FREDIS_TEST_SHOW_OUTPUT"] = "1"
    from tests._output import configure_test_output
    configure_test_output()

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(app_root / "tests"), pattern="Test*.py", top_level_dir=str(app_root))
    show_output = os.environ.get("FREDIS_TEST_SHOW_OUTPUT") == "1"
    runner = unittest.TextTestRunner(verbosity=2, descriptions=False, buffer=not show_output,
                                     warnings=None if show_output else "ignore")
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
