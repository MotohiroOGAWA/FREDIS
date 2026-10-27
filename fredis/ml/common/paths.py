"""Persist file references relative to the document that owns them."""

import os
from pathlib import Path


PATH_KEYS = {
    "path", "source", "file", "input", "trainInput", "validationInput",
    "validation_input", "resume", "initialize_from", "initializeFrom",
    "params", "modelPath", "paramsPath", "sourcePath",
}


def is_path_key(key):
    return key in PATH_KEYS or key.endswith(("_dir", "_path", "Dir", "Path"))


def relative_paths(value, base):
    """Copy structured metadata, rewriting absolute file references only."""
    if isinstance(value, dict):
        return {
            key: (Path(os.path.relpath(item, base)).as_posix()
                  if is_path_key(key) and isinstance(item, (str, Path))
                  and str(item) and Path(item).is_absolute()
                  else relative_paths(item, base))
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [relative_paths(item, base) for item in value]
    return value


def resolve_paths(value, base):
    """Resolve persisted references for runtime use without changing the document."""
    if isinstance(value, dict):
        return {
            key: (str((Path(base) / item).resolve())
                  if is_path_key(key) and isinstance(item, str) and item
                  else resolve_paths(item, base))
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [resolve_paths(item, base) for item in value]
    return value
