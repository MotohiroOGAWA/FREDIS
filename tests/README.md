# Running tests

Run from any directory with:

```bash
bash /workspaces/FREDIS/mnt/app/tests/run_tests.sh
```

The runner shows one `test_name (...) ... ok` line per test and a final summary. Python stdout/stderr (including progress bars) are buffered and shown only on failures. Native RDKit diagnostics are disabled within test packages; Python exceptions and tracebacks are preserved. Progress bars still execute normally so their behavior can be tested. A failed run exits with status 1.

Enable diagnostics explicitly:

```bash
bash /workspaces/FREDIS/mnt/app/tests/run_tests.sh --show-output
```

For direct unittest discovery from `mnt/app`, use buffering:

```bash
python -m unittest discover -s tests -p 'Test*.py' -v -b
```

`FREDIS_TEST_SHOW_OUTPUT=1` enables native RDKit diagnostics for direct unittest or pytest runs. Omit unittest's `-b` (or use pytest's `-s`) when inspecting Python output.
