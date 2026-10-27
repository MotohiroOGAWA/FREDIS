from __future__ import annotations
import argparse
import json
from pathlib import Path
from ...base import CLICommand


class TuneFragmentTreeCommand(CLICommand):
    name = "tune-fragment-tree"
    help = (
        "Run or resume fragment-tree trials and save a .tuning.pft comparison."
    )
    description = (
        "Run the ordinary fragment-tree CLI for each trial. Supply a JSON file with "
        "outputDir, baseArgs and parameters (CLI flag to candidate list), or an explicit trials plan."
    )
    order = 31

    def configure(self, parser: argparse.ArgumentParser) -> None:
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--plan", type=Path, help="Saved .tuning-plan.pft or legacy JSON search plan (same output resumes unfinished trials)")
        source.add_argument("--plan-json", help="Inline search specification JSON")

    def run(self, args: argparse.Namespace) -> None:
        from fredis.ml.tuning.fragment_tree import run

        plan = json.loads(args.plan.read_text() if args.plan else args.plan_json)
        raise SystemExit(run(plan))
