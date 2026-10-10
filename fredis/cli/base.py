from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import ClassVar, Sequence


class CLICommand(ABC):
    """Base class for one executable CLI command.

    A command corresponds to one subcommand such as:

        fredis analysis reaction-pattern
        fredis train fragment-tree

    Each command file under `commands/` should define a subclass of this class.
    """

    name: ClassVar[str]
    help: ClassVar[str] = ""
    description: ClassVar[str | None] = None
    aliases: ClassVar[Sequence[str]] = ()
    order: ClassVar[int] = 100

    def register(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> argparse.ArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.description or self.help,
            aliases=list(self.aliases),
        )

        self.configure(parser)
        parser.set_defaults(func=self.run)

        return parser

    def configure(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """Add command-specific arguments.

        Override this method in subclasses when the command needs arguments.
        """
        return None

    @abstractmethod
    def run(
        self,
        args: argparse.Namespace,
    ) -> None:
        """Run the command."""
        raise NotImplementedError


class CLIGroup(ABC):
    """Base class for a CLI command group.

    A group corresponds to one top-level command such as:

        fredis analysis
        fredis train

    Each feature directory should define one subclass of this class in group.py.
    """

    name: ClassVar[str]
    help: ClassVar[str] = ""
    description: ClassVar[str | None] = None
    aliases: ClassVar[Sequence[str]] = ()
    order: ClassVar[int] = 100

    @property
    @abstractmethod
    def commands_package(self) -> str:
        """Python package path that contains command modules."""
        raise NotImplementedError

    def register(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> argparse.ArgumentParser:
        from .discovery import register_commands_from_package

        parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.description or self.help,
            aliases=list(self.aliases),
        )

        command_subparsers = parser.add_subparsers(
            dest=f"{self.name}_command",
            required=True,
        )

        register_commands_from_package(
            command_subparsers,
            self.commands_package,
        )

        return parser