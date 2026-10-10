"""Process a trusted parent-created preparation chunk in an isolated Python process."""
import argparse
import json
import pickle
from pathlib import Path
import torch
from .context import create_preparation_context
from .create_training_data import _prepare_group_safely
from fredis.ml.input.structure_builder import ReactionStructureBuilder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True)
    parser.add_argument('--result',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    with Path(args.task).open('rb') as stream:
        payload=pickle.load(stream)
    generator=create_preparation_context(payload['model_config'])
    options=payload['options']
    builder=ReactionStructureBuilder(generator,max_unique_fragment_smiles=options['max_unique_fragment_smiles'],
        max_reaction_combinations=options['max_reaction_combinations'])
    results=[]
    for task in payload['tasks']:
        result=_prepare_group_safely(task,builder,options)
        if 'path' in result: result['path']=str(result['path'])
        results.append(result)
    Path(args.result).write_text(json.dumps(results))


if __name__=='__main__':
    main()
