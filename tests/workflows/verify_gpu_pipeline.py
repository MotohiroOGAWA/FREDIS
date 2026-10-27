"""Exercise preparation, CUDA training, pattern fine-tuning and prediction on MoNA.

Run with PYTHONPATH=. python tests/workflows/verify_gpu_pipeline.py --output PATH.
Experimental peaks are retained; the small model verifies execution, not accuracy.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from fredis.domain.fragment import Fragmenter
from fredis.data.core.MSDataset import MSDataset
from fredis.ml.training.fragment_tree_training.integration import PATTERN_KEY
from fredis.ml.training.fragment_tree_training.pattern_models import inspect_model
from fredis.ml.training.fragment_tree_training.training import build_arg_parser, training_model_config

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', default='/workspaces/CLEFTS/raw_data/MoNA/flavonoids/mona_pos_cf_flavonoids.msds')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--training-command', choices=['fragment-tree', 'fragment-tree-detailed'],
                        default='fragment-tree-detailed')
    parser.add_argument('--continue-run', action='store_true')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=args.continue_run)
    torch.set_num_threads(1)
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    progress = out / 'progress.json'
    steps = json.loads(progress.read_text()) if args.continue_run and progress.exists() else []

    def run(name, argv):
        if any(step['name'] == name for step in steps):
            return
        started = time.monotonic()
        log = out / (name + '.log')
        with log.open('w') as handle:
            result = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env,
                                    stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name}: {log}\n{log.read_text()[-6000:]}')
        steps.append(dict(name=name, command=[sys.executable, *argv],
                          seconds=round(time.monotonic() - started, 3), log=str(log)))
        progress.write_text(json.dumps(steps, indent=2))
        print(name + ': passed', flush=True)

    if not (out / 'train.msds').exists():
        records = MSDataset.load(args.source, load_peak_metadata=False)
        for split, indices in [('train', [4305, 4358]), ('validation', [3765, 3762]),
                               ('prediction', [4305, 4358, 3765, 3762])]:
            raw = out / ('raw_' + split + '.msds')
            records[indices].save(str(raw))
            selected = MSDataset.load(str(raw), load_peak_metadata=False)
            metadata = selected.metadata.copy()
            metadata['SmokeCollisionEnergy'] = [float(str(value).split()[0]) for value in metadata.CollisionEnergy]
            metadata['OriginalRecordIndex'] = indices
            MSDataset(metadata, selected.peaks, columns=metadata.columns.tolist()).save(str(out / (split + '.msds')))
        del records
    train_data = MSDataset.load(str(out / 'train.msds'), load_peak_metadata=False)
    base = json.loads((ROOT / 'fredis/presets/fragment/reaction_pattern_sets/nonring_single_bond/nonring_single_bond.fragmenter.pft').read_text())
    base['fragment_ion_tree_builder']['max_reaction_count'] = 2
    (out / 'base.fragmenter.pft').write_text(json.dumps(base, indent=2))

    def prepare(name, chemistry, pretrained=None):
        argv = ['-m', 'fredis.cli', 'train', 'create-fragment-tree-data',
                '--input', str(out / 'train.msds'), '--validation-input', str(out / 'validation.msds'),
                '--output-dir', str(out / name), '--params', str(chemistry),
                '--collision-energy-column', 'SmokeCollisionEnergy', '--validation-ratio', '0.5',
                '--minimum-relative-intensity', '0.1', '--max-unique-fragment-smiles', '100',
                '--max-reaction-combinations', '1000', '--num-workers', '1', '--save-all-edges', '--overwrite', '1']
        if pretrained:
            argv += ['--pretrained-pattern-model', str(pretrained)]
        run('prepare_' + name, argv)
        files = list((out / name / 'train_structures').rglob('*.preft.pt'))
        validation = list((out / name / 'validation_structures').rglob('*.preft.pt'))
        assert files and validation, name
        for file in files + validation:
            torch.load(file, weights_only=False, map_location='cpu')['structure'].validate()
        return files

    prepare('base_data', out / 'base.fragmenter.pft')
    cli = build_arg_parser().parse_args(['--dataset-dir', 'unused', '--output-dir', 'unused',
        '--node-dim', '12', '--graph-dim', '17', '--num-heads', '3', '--num-layers', '1',
        '--reaction-hidden-dim', '12', '--reaction-num-heads', '3', '--reaction-main-adduct-dim', '12',
        '--fragment-fc-dims', '16', '--state-fc-dims', '16', '--state-set-layers', '1',
        '--spectrum-gat-layers', '1', '--dropout', '0.1'])
    config = training_model_config(cli)
    config['reaction_model_params']['reaction_transformer'] = dict(encoder_layers=1, decoder_layers=1)
    (out / 'base.architecture.json').write_text(json.dumps(dict(model_config=config), indent=2))

    def train(name, dataset, architecture=None, resume=None):
        argv = ['-m', 'fredis.cli', 'train', args.training_command, '--dataset-dir', str(out / dataset),
                '--output-dir', str(out / name), '--epochs', '2' if resume else '1',
                '--batch-size', '1', '--device', args.device, '--lr', '0.002', '--warmup-steps', '0',
                '--max-samples', '2', '--train-log-interval-steps', '1', '--dropout', '0.1',
                '--minimum-assignment-score', '0', '--minimum-assignment-score-without-precursor', '0',
                '--validate-before-training']
        if resume:
            argv += ['--resume', str(resume)]
        elif args.training_command == 'fragment-tree':
            argv += ['--hidden-dim', '12', '--num-heads', '3', '--num-layers', '1',
                     '--ffn-mult', '2', '--reaction-encoder-layers', '1',
                     '--reaction-decoder-layers', '1', '--fc-layers', '3', '--fc-mult', '2', '--overwrite']
        else:
            argv += ['--model-architecture', str(architecture), '--overwrite']
        run(name + ('_resume' if resume else ''), argv)
        checkpoint = out / name / 'last.pt'
        saved = torch.load(checkpoint, weights_only=False, map_location='cpu')
        assert all(torch.isfinite(value).all() for value in saved['model_state_dict'].values()
                   if value.is_floating_point())
        return checkpoint

    checkpoint = train('training', 'base_data', out / 'base.architecture.json')
    train('training', 'base_data', resume=checkpoint)
    fine = deepcopy(base)
    fine['fragment_ion_tree_builder']['reaction_pattern_set']['patterns'].append(dict(
        name='carbonyl_workflow_probe', reactant_smarts='[#6:1]-[#6:2](=[#8:3])-[#6:4]',
        products=[dict(name='carbonyl_fragment', smarts='[*:2]=[*:3]')]))
    (out / 'fine.fragmenter.pft').write_text(json.dumps(fine, indent=2))
    fine_files = prepare('fine_data', out / 'fine.fragmenter.pft', checkpoint)
    patterns = Fragmenter.from_dict(fine).reaction_pattern_set.patterns
    new_pattern = next(p for p in patterns if p.name == 'carbonyl_workflow_probe')
    assert sum(int((torch.load(path, weights_only=False)['structure'].reactions.reaction_type[:, 0]
                    == new_pattern.pattern_id).sum()) for path in fine_files) > 0
    fine_config = deepcopy(config)
    fine_config['reaction_model_params']['pattern_model_params'] = {
        str(new_pattern.pattern_id): dict(hidden_dim=8, num_heads=2, num_layers=1)}
    (out / 'fine.architecture.json').write_text(json.dumps(dict(model_config=fine_config), indent=2))
    fine_checkpoint = train('fine_tuning', 'fine_data', out / 'fine.architecture.json')
    before = torch.load(checkpoint, weights_only=False, map_location='cpu')
    after = torch.load(fine_checkpoint, weights_only=False, map_location='cpu')
    assert before['epoch'] >= 2 and before['global_step'] >= 2
    assert after['epoch'] >= 1 and after['global_step'] >= 1
    assert before['optimizer_state_dict']['state'] and after['optimizer_state_dict']['state']
    frozen_ids = set(map(int, after['model_config']['fine_tuning']['frozen_pattern_ids']))
    assert frozen_ids and new_pattern.pattern_id not in frozen_ids
    checked = 0
    remap = {item['old_id']: item['new_id'] for item in after['model_config']['fine_tuning']['pattern_imports']}
    for name, tensor in before['model_state_dict'].items():
        match = PATTERN_KEY.search(name)
        if match and int(match.group(1)) in remap:
            target = name[:match.start(1)] + str(remap[int(match.group(1))]) + name[match.end(1):]
            torch.testing.assert_close(tensor, after['model_state_dict'][target], rtol=0, atol=0)
            checked += 1
    assert checked
    # Current fine-tuning trains MolEncoder independently of frozen raw-chemistry reactions.
    mol_updated = any(not torch.equal(tensor, after['model_state_dict'][name])
                      for name, tensor in before['model_state_dict'].items()
                      if name.startswith('feature_model.mol_encoder.'))
    assert mol_updated

    row = train_data.metadata.iloc[0]
    prediction = ['-m', 'fredis.ml.specgen.predict_spectrum', '--model', str(fine_checkpoint),
                  '--device', args.device, '--strict', '--max-samples', '2', '--max-fragment-nodes', '30',
                  '--peak-intensity-threshold', '0']
    run('single_prediction', [*prediction, '--smiles', str(row.SMILES), '--ce', str(row.SmokeCollisionEnergy),
                             '--adduct-type', str(row.AdductType), '--output-dir', str(out / 'single')])
    run('batch_prediction', [*prediction, '--input', str(out / 'prediction.msds'),
                            '--collision-energy-column', 'SmokeCollisionEnergy', '--output-dir', str(out / 'batch'),
                            '--num-workers', '2', '--chunk-size', '1'])
    single = MSDataset.load(str(out / 'single/predicted.msds'))
    batch = MSDataset.load(str(out / 'batch/predicted.msds'))
    assert len(single) == 1 and len(batch) == 4
    assert batch.metadata.OriginalRecordIndex.tolist() == [4305, 4358, 3765, 3762]
    assert all(len(peaks.data) and np.isfinite(peaks.data).all() for peaks in batch.peaks)
    np.testing.assert_allclose(single.peaks[0].data, batch.peaks[0].data, rtol=1e-4, atol=1e-5)
    report = dict(source=args.source, experimental_record_indices=[4305, 4358, 3765, 3762],
                  device=args.device, training_command=args.training_command,
                  commands=steps, prepared_splits_valid=True, resume_verified=True,
                  training_epochs=before['epoch'], training_steps=before['global_step'],
                  fine_tuning_epochs=after['epoch'], fine_tuning_steps=after['global_step'],
                  frozen_parameter_tensors_preserved=checked, molecule_encoder_updated=mol_updated,
                  added_pattern_id=new_pattern.pattern_id,
                  patterns=inspect_model(fine_checkpoint)['patterns'], single_rows=len(single), batch_rows=len(batch),
                  single_batch_predictions_match=True, scope='Execution and numerical validity; not predictive accuracy')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(out / 'report.json', flush=True)


if __name__ == '__main__':
    main()
