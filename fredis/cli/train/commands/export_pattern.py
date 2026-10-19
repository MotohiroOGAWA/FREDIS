import argparse
import json
from ...base import CLICommand


class ExportPatternCommand(CLICommand):
    name = "export-pattern"
    help = "Export a complete immutable reaction feature to .reaction-model.pt."
    order = 45

    def configure(self, parser: argparse.ArgumentParser):
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--pattern-id", type=int, required=True)
        parser.add_argument("--output", required=True)

    def run(self, args):
        from fredis.ml.training.fragment_tree_training.pattern_models import (
            export_pattern,
        )

        print(
            json.dumps(
                export_pattern(args.checkpoint, args.pattern_id, args.output), indent=2
            )
        )
