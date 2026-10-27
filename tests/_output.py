"""Shared quiet defaults for project tests, including unittest discovery."""
from __future__ import annotations

import os
from rdkit import rdBase


def configure_test_output() -> None:
    """Keep native RDKit diagnostics opt-in within test processes.

    Native RDKit logs bypass unittest's Python stream buffering. Disabling
    them leaves Python exceptions intact, so invalid chemistry still fails
    with its normal traceback. Set FREDIS_TEST_SHOW_OUTPUT=1 for diagnostics.
    """
    show_output = os.environ.get("FREDIS_TEST_SHOW_OUTPUT") == "1"
    if show_output:
        rdBase.EnableLog("rdApp.*")
    else:
        rdBase.DisableLog("rdApp.*")
