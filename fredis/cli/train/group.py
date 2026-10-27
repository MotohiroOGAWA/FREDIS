from __future__ import annotations

from ..base import CLIGroup


class TrainGroup(CLIGroup):
    name = "train"
    help = "Train FREDIS models."
    description = "Train FREDIS end-to-end machine-learning models."
    order = 30

    @property
    def commands_package(self) -> str:
        return "fredis.cli.train.commands"
