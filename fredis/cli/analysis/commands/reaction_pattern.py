from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...base import CLICommand
from fredis.analysis.reaction_pattern import generate_reaction_pattern_analysis
from fredis.analysis.reaction_pattern.constants import RESULT_NAME
from fredis.analysis.reaction_pattern.tree import NodeSpec, TreeConfiguration, load_pattern_set_file, load_tree_configuration
from fredis.ml.specgen.config_options import configure_model_options, resolve_model_options


def parse_source_specs(values: list[str]) -> list[dict]:
    result = []
    for value in values:
        text = Path(value[1:]).read_text() if value.startswith("@") else value
        parsed = json.loads(text)
        rows = parsed if isinstance(parsed, list) else [parsed]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("--source-spec must be a JSON object, a list of objects, or @file.")
        result.extend(rows)
    return result


def parse_pattern_set_option(value: str):
    """``[ID=]FILE``: a pattern-set file, or a single pattern as a one-pattern set named after it."""
    set_id, separator, file_name = value.partition("=")
    if not separator or Path(value).exists():
        set_id, file_name = None, value
    return load_pattern_set_file(file_name, set_id=set_id or None, name=set_id or None)


def parse_node_option(value: str) -> NodeSpec:
    """``ID=SET[@PARENT]``."""
    node_id, separator, rest = value.partition("=")
    if not separator or not node_id or not rest:
        raise ValueError(f"--node must be ID=PATTERN_SET[@PARENT]: {value}")
    pattern_set, _, parent = rest.partition("@")
    return NodeSpec(node_id.strip(), pattern_set.strip(), parent.strip() or None)


def build_tree(base: TreeConfiguration | None, pattern_sets: list, nodes: list[NodeSpec]) -> TreeConfiguration | None:
    """Merge repeatable --pattern-set / --node options into a Tree Configuration."""
    if base is None and not pattern_sets and not nodes:
        return None
    sets = list(base.pattern_sets) if base is not None else []
    for item in pattern_sets:
        existing = next((index for index, current in enumerate(sets) if current.id == item.id), None)
        if existing is None:
            sets.append(item)
        elif sets[existing].patterns != item.patterns:
            raise ValueError(f"Pattern set id {item.id!r} is used for different patterns.")
    tree_nodes = list(base.nodes) if base is not None else []
    tree_nodes.extend(nodes)
    tree = TreeConfiguration(tuple(sets), tuple(tree_nodes))
    tree.validate()
    return tree


def _explicit_model_options(args: argparse.Namespace) -> bool:
    from fredis.ml.specgen.config_options import OPTIONS

    return bool(args.params or args.params_json or args.parameter_overrides
                or any(getattr(args, group + "_params_json", None) for group in ("fragmenter", "mol_encoder",
                                                                                   "reaction_model", "spectrum_model"))
                or any(getattr(args, key, None) is not None for key in OPTIONS))


RUN_DEFAULTS = {"minimum_relative_intensity": ("minimumRelativeIntensity", 0.0),
                "max_unique_fragment_smiles": ("maxUniqueFragmentSmiles", -1),
                "max_reaction_combinations": ("maxReactionCombinations", -1),
                "num_workers": ("numWorkers", 1), "chunk_size": ("chunkSize", 1)}


def run_settings(args: argparse.Namespace, previous_run: dict | None) -> dict:
    result = {}
    for name, (key, default) in RUN_DEFAULTS.items():
        value = getattr(args, name, None)
        if value is None:
            value = (previous_run or {}).get(key, default)
        result[name] = bool(value) if isinstance(default, bool) else type(default)(value)
    return result


class ReactionPatternAnalysisCommand(CLICommand):
    name = "reaction-pattern"
    help = "Measure peak assignment gained along a tree of reaction pattern sets."
    description = ("Analyse a PatternSet tree: every node's condition is the union of the pattern sets on its path, "
                   "and every edge compares parent condition with parent condition + child set. Writes "
                   "result.reactionpatternanalysis.pft (HDF5 summaries), compounds/*.peaks.parquet and "
                   "details/conditions/**/*.preft.pt.")
    order = 10

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--source", action="append", default=[], help="Dataset, prepared mapping, or Reaction Pattern Analysis result; auto-detected and repeatable.")
        parser.add_argument("--source-spec", action="append", default=[], help="Per-source JSON (or @JSON-file) with path, kind and columns; repeatable.")
        parser.add_argument("--output-dir", required=True)
        configure_model_options(parser)
        parser.add_argument("--tree-config", help="Tree Configuration JSON file (patternSets + nodes), or a run configuration containing one.")
        parser.add_argument("--tree-config-json", help="Inline Tree Configuration JSON; same content as --tree-config without a file.")
        parser.add_argument("--pattern-set", action="append", default=[], metavar="[ID=]FILE",
                            help="Add a reaction pattern-set file, or a single pattern as a one-pattern set, to the pattern set list; repeatable.")
        parser.add_argument("--node", action="append", default=[], metavar="ID=SET[@PARENT]",
                            help="Place a pattern set in the tree as a root, or below PARENT; repeatable.")
        for name, default in (("smiles", "SMILES"), ("adduct-type", "AdductType"), ("collision-energy", "CollisionEnergy"), ("precursor-mz", "PrecursorMZ")):
            parser.add_argument(f"--{name}-column", default=default)
        # None means "the existing result's value" with --extend, else the default in RUN_DEFAULTS.
        parser.add_argument("--minimum-relative-intensity", type=float, default=None)
        parser.add_argument("--max-unique-fragment-smiles", type=int, default=None)
        parser.add_argument("--max-reaction-combinations", type=int, default=None)
        parser.add_argument("--num-workers", type=int, default=None)
        parser.add_argument("--chunk-size", type=int, default=None)
        parser.add_argument("--overwrite", type=int, choices=(0, 1), default=0,
                            help="Delete and replace the complete output directory. If omitted in an interactive terminal, ask first.")
        parser.add_argument("--extend", type=int, choices=(0, 1), default=0,
                            help="Keep the result in --output-dir and calculate only conditions it does not have yet "
                                 "(e.g. added nodes). Sources, settings and tree default to the existing result's; "
                                 "--pattern-set/--node add to its tree.")

    def run(self, args: argparse.Namespace) -> None:
        extend = bool(args.extend)
        previous = None
        if extend:
            from fredis.analysis.reaction_pattern.previous import PreviousResult

            previous = PreviousResult(Path(args.output_dir) / RESULT_NAME)
        if extend and not _explicit_model_options(args):
            config = None
        else:
            if extend and not args.params and not args.params_json:
                args.params_json = json.dumps(previous.model_config)
            config = resolve_model_options(args)
        if args.tree_config and args.tree_config_json:
            raise ValueError("Use either --tree-config or --tree-config-json, not both.")
        if args.tree_config_json:
            base = TreeConfiguration.from_dict(json.loads(args.tree_config_json))
        elif args.tree_config:
            base = load_tree_configuration(args.tree_config)
        else:
            base = previous.tree if previous else None
        tree = build_tree(base, [parse_pattern_set_option(value) for value in args.pattern_set],
                          [parse_node_option(value) for value in args.node])
        if tree is not None and extend and base is previous.tree and not args.pattern_set and not args.node:
            tree = None
        sources = [{"path": value, "kind": "auto"} for value in args.source]
        sources.extend(parse_source_specs(args.source_spec))
        result = generate_reaction_pattern_analysis(
            sources=sources, output_dir=args.output_dir, model_config=config, tree=tree,
            smiles_column=args.smiles_column, adduct_type_column=args.adduct_type_column,
            collision_energy_column=args.collision_energy_column, precursor_mz_column=args.precursor_mz_column,
            overwrite=bool(args.overwrite), extend=extend, **run_settings(args, previous.run if previous else None),
        )
        print(json.dumps({"ok": True, "result": str(result)}, ensure_ascii=False))
