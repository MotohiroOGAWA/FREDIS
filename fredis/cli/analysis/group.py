from __future__ import annotations

from ..base import CLIGroup


class AnalysisGroup(CLIGroup):
    name = "analysis"
    help = "Generate and inspect reusable dataset analysis results."
    order = 30

    @property
    def commands_package(self) -> str:
        return "fredis.cli.analysis.commands"
