import argparse
import json
from ...base import CLICommand


class IntegratePatternsCommand(CLICommand):
    name = 'integrate-patterns'
    help = 'Combine immutable pattern feature encoders; choose one downstream predictor.'
    order = 40

    def configure(self, parser: argparse.ArgumentParser):
        parser.add_argument('--checkpoints', nargs='+', required=True)
        parser.add_argument('--head-checkpoint', required=True)
        parser.add_argument('--output', required=True)

    def run(self, args):
        from fredis.ml.training.fragment_tree_training.integration import integrate_checkpoints
        print(json.dumps(integrate_checkpoints(args.checkpoints, args.head_checkpoint, args.output), indent=2))
