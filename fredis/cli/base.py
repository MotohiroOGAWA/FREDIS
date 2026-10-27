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


def namespace_argv(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[str]:
    """Re-serialize parsed arguments so a command can forward them to a module main()."""
    argv = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction) or not action.option_strings:
            continue
        value = getattr(args, action.dest, None)
        if value is None:
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(action.option_strings[0])
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                argv.append(action.option_strings[0])
        elif isinstance(action, argparse._AppendAction):
            for item in value:
                argv.extend([action.option_strings[0], str(item)])
        elif action.nargs in ("*", "+"):
            argv.extend([action.option_strings[0], *(str(item) for item in value)])
        else:
            argv.extend([action.option_strings[0], str(value)])
    return argv


def output_directory(value: str) -> str:
    """argparse type for --output-dir: a blank value would resolve to the working directory."""
    if not value.strip():
        raise argparse.ArgumentTypeError("an output directory is required")
    return value
