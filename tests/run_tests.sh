#!/bin/bash
# Run the shared verbose, buffered test runner from any working directory.
exec python "$(dirname -- "$0")/run_tests.py" "$@"
