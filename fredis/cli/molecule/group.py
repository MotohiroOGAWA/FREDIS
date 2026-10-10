from __future__ import annotations

from ..base import CLIGroup


class MoleculeGroup(CLIGroup):
    name = "molecule"
    aliases = ("mol",)
    help = "Inspect molecules and calculate molecular properties."
    description = "Molecule-level chemistry utilities."
    order = 20

    @property
    def commands_package(self) -> str:
        return "fredis.cli.molecule.commands"
