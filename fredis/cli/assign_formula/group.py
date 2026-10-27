from __future__ import annotations

from ..base import CLIGroup


class AssignFormulaGroup(CLIGroup):
    name = "assign-formula"
    help = "Assign possible subformulas to MS/MS peaks."
    description = "Assign possible subformulas to MS/MS peaks from precursor formulas."
    order = 20

    @property
    def commands_package(self) -> str:
        return "fredis.cli.assign_formula.commands"
