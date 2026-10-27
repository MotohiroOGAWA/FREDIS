"""Chemistry options shared by data preparation and Reaction Pattern Analysis.

Chemistry is the Fragmenter plus the element symbols of the atom features. It
belongs to prepared data; model architecture is chosen only at training time.
"""

from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

DEFAULT_SYMBOLS = ("C", "N", "O", "P", "S", "F", "Cl", "Br", "I")
OPTIONS = {
    "max_reaction_count": ("fragment_ion_tree_builder", "max_reaction_count"),
    "precursor_max_reaction_count": ("precursor_candidate_max_reaction_count",),
    "mass_tolerance": ("mass_tolerance",),
}


def configure_chemistry_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--params",
        help="Fragmenter file (*.fragmenter.pft or JSON), or a chemistry configuration "
        "with fragmenter_params and symbols. Explicit options override file values.",
    )
    parser.add_argument(
        "--params-json",
        help="Inline Fragmenter or chemistry configuration JSON; overrides --params.",
    )
    parser.add_argument(
        "--max-reaction-count",
        type=int,
        help="Override fragment_ion_tree_builder.max_reaction_count.",
    )
    parser.add_argument(
        "--precursor-max-reaction-count",
        type=int,
        help="Override precursor_candidate_max_reaction_count.",
    )
    parser.add_argument(
        "--mass-tolerance", help="Override mass_tolerance, e.g. 0.01Da,10ppm."
    )
    parser.add_argument(
        "--set",
        dest="parameter_overrides",
        action="append",
        default=[],
        metavar="PATH=JSON",
        help="Override any Fragmenter parameter, including nested array entries. Repeat as needed; applied last.",
    )
    parser.add_argument(
        "--symbols-json",
        help="JSON array of element symbols used for atom features. "
        "Default: the configuration file's symbols, else "
        + ",".join(DEFAULT_SYMBOLS)
        + ".",
    )


def merge(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def unpack_chemistry(document: dict) -> dict:
    """``{'fragmenter_params', 'symbols'?}`` from a Fragmenter or chemistry configuration."""
    if not isinstance(document, dict):
        raise ValueError("Chemistry configuration must be a JSON object")
    if "fragmenter_params" in document:
        result = {"fragmenter_params": document["fragmenter_params"]}
    elif "fragment_ion_tree_builder" in document:
        result = {"fragmenter_params": document}
    else:
        raise ValueError(
            "Expected a Fragmenter configuration (fragment_ion_tree_builder) or fragmenter_params"
        )
    if document.get("symbols") is not None:
        result["symbols"] = document["symbols"]
    return copy.deepcopy(result)


def set_parameter(config: dict, path: list[str], value: object) -> None:
    if not path or any(
        key in ("__proto__", "prototype", "constructor") or not key for key in path
    ):
        raise ValueError("Invalid parameter path")
    parent = config
    for key in path[:-1]:
        if isinstance(parent, list):
            parent = parent[int(key)]
        else:
            if key not in parent:
                parent[key] = {}
            parent = parent[key]
    if isinstance(parent, list):
        parent[int(path[-1])] = value
    else:
        parent[path[-1]] = value


def has_chemistry_options(args: argparse.Namespace) -> bool:
    return bool(
        args.params
        or args.params_json
        or args.parameter_overrides
        or args.symbols_json
        or any(getattr(args, key, None) is not None for key in OPTIONS)
    )


def resolve_chemistry_options(
    args: argparse.Namespace, base: dict | None = None
) -> dict:
    """Return ``{'fragmenter_params', 'symbols'}``; ``base`` is used when no file is given."""
    config = copy.deepcopy(base) if base else {}
    if args.params:
        config = unpack_chemistry(json.loads(Path(args.params).read_text()))
    if args.params_json:
        update = unpack_chemistry(json.loads(args.params_json))
        config = merge(config, update) if config else update
    if "fragmenter_params" not in config:
        raise ValueError("Specify the Fragmenter with --params or --params-json")
    fragmenter = config["fragmenter_params"]
    for key, path in OPTIONS.items():
        value = getattr(args, key, None)
        if value is not None:
            set_parameter(fragmenter, list(path), value)
    for override in args.parameter_overrides:
        key, separator, raw = override.partition("=")
        if not separator:
            raise ValueError("--set requires PATH=JSON")
        set_parameter(fragmenter, key.split("."), json.loads(raw))
    if args.symbols_json:
        config["symbols"] = json.loads(args.symbols_json)
    config.setdefault("symbols", list(DEFAULT_SYMBOLS))
    return {"fragmenter_params": fragmenter, "symbols": list(config["symbols"])}
