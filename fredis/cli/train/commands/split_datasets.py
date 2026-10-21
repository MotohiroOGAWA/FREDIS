from __future__ import annotations
import argparse
from ...base import CLICommand, namespace_argv
from fredis.ml.data_preparation.fragment_tree.split_datasets import build_arg_parser, main


class SplitDatasetsCommand(CLICommand):
    name = 'split-datasets'
    help = 'Check and export a dataset list as molecule-disjoint train / validation / test datasets.'
    order = 19

    def configure(self, parser: argparse.ArgumentParser) -> None:
        source = build_arg_parser()
        # Preserve the mutually exclusive input group on the top-level parser.
        group = parser.add_mutually_exclusive_group(required=True)
        input_names = {'datasets_json', 'config_stdin', 'checked_plan'}
        for action in source._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            (group if action.dest in input_names else parser)._add_action(action)

    def run(self, args: argparse.Namespace) -> None:
        main(namespace_argv(build_arg_parser(), args))
