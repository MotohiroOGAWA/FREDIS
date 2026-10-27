from __future__ import annotations

import argparse

from ...base import CLICommand, namespace_argv
from fredis.ml.training.fragment_tree_training.training import (
    build_arg_parser as build_training_arg_parser,
    main as run_training,
)


class FragmentTreeDetailedTrainCommand(CLICommand):
    name = "fragment-tree-detailed"
    aliases = ("tree-detailed",)
    help = (
        "Train with independent molecular, reaction and spectrum architecture options."
    )
    description = (
        "Train with detailed per-component and per-pattern architecture settings. "
        "Pass --dataset-dir to inherit chemistry and imported reaction models."
    )
    order = 31

    def configure(self, parser: argparse.ArgumentParser) -> None:
        source = build_training_arg_parser()
        parser.usage = "%(prog)s [options]"
        for action in source._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            parser._add_action(action)

    def run(self, args: argparse.Namespace) -> None:
        run_training(namespace_argv(build_training_arg_parser(), args))
