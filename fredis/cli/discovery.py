from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Iterable, TypeVar

from .base import CLICommand, CLIGroup


TCommand = TypeVar("TCommand", bound=CLICommand)
TGroup = TypeVar("TGroup", bound=CLIGroup)


def register_groups_from_package(
    subparsers: argparse._SubParsersAction,
    package_name: str,
) -> None:
    groups = discover_groups_from_package(package_name)

    for group in sorted(groups, key=lambda item: (item.order, item.name)):
        group.register(subparsers)


def register_commands_from_package(
    subparsers: argparse._SubParsersAction,
    package_name: str,
) -> None:
    commands = discover_commands_from_package(package_name)

    for command in sorted(commands, key=lambda item: (item.order, item.name)):
        command.register(subparsers)


def discover_groups_from_package(
    package_name: str,
) -> list[CLIGroup]:
    """Discover CLI groups from direct child packages.

    Example
    -------
    fredis.cli.analysis.group
    fredis.cli.train.group

    Each group.py must define a subclass of CLIGroup.
    """

    package = importlib.import_module(package_name)

    groups: list[CLIGroup] = []

    for module_info in pkgutil.iter_modules(
        package.__path__,
        package.__name__ + ".",
    ):
        if not module_info.ispkg:
            continue

        short_name = module_info.name.rsplit(".", maxsplit=1)[-1]

        if short_name.startswith("_"):
            continue

        group_module_name = module_info.name + ".group"

        try:
            group_module = importlib.import_module(group_module_name)
        except ModuleNotFoundError as exc:
            if exc.name == group_module_name:
                continue
            raise

        groups.extend(_instantiate_subclasses(group_module, CLIGroup))

    return groups


def discover_commands_from_package(
    package_name: str,
) -> list[CLICommand]:
    """Discover command classes from all modules in a commands package.

    Example
    -------
    fredis.cli.analysis.commands.reaction_pattern
    fredis.cli.train.commands.fragment_tree

    Each module can define one or more subclasses of CLICommand.
    """

    package = importlib.import_module(package_name)

    commands: list[CLICommand] = []

    for module_info in pkgutil.iter_modules(
        package.__path__,
        package.__name__ + ".",
    ):
        short_name = module_info.name.rsplit(".", maxsplit=1)[-1]

        if short_name.startswith("_"):
            continue

        module = importlib.import_module(module_info.name)
        commands.extend(_instantiate_subclasses(module, CLICommand))

    return commands


def _instantiate_subclasses(
    module: ModuleType,
    base_class: type[TCommand] | type[TGroup],
) -> list[TCommand] | list[TGroup]:
    instances = []

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is base_class:
            continue

        if not issubclass(obj, base_class):
            continue

        if obj.__module__ != module.__name__:
            continue

        if inspect.isabstract(obj):
            continue

        instances.append(obj())

    return instances