from __future__ import annotations

import argparse

from ...base import CLICommand, namespace_argv
from fredis.ml.training.fragment_tree_training.shared_training import (
    build_arg_parser as build_training_arg_parser,
    main as run_training,
)


class FragmentTreeTrainCommand(CLICommand):
    name = "fragment-tree"
    aliases = ("tree",)
    help = "Train with shared dimensions, heads, layers and FFN multipliers."
    description = (
        "Train with one shared hidden width, head count and base depth. "
        "Settings expand into fragment-tree-detailed arguments before training."
    )
    order = 30

    def configure(self, parser: argparse.ArgumentParser) -> None:
        source = build_training_arg_parser()
        parser.usage = "%(prog)s [options]"
        for action in source._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            parser._add_action(action)

    def run(self, args: argparse.Namespace) -> None:
        run_training(namespace_argv(build_training_arg_parser(), args))
