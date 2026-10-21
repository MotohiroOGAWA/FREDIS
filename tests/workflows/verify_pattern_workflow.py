"""Run actual CLI preparation/training/branching/integration on tiny MoNA data.

Usage: PYTHONPATH=. python tests/workflows/verify_pattern_workflow.py --output data/workflow_verification
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fredis.libs.msentity.msentity.core.MSDataset import MSDataset
from fredis.domain.fragment import Fragmenter
from fredis.ml.specgen.predict_spectrum import load_generator
from fredis.ml.training.fragment_tree_training.integration import PATTERN_KEY

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--continue-run',action='store_true')
parser.add_argument('--source', default='/workspaces/CLEFTS/raw_data/MoNA/flavonoids/mona_pos_cf_flavonoids.msds')
args = parser.parse_args()
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=args.continue_run)
torch.set_num_threads(1)
records = MSDataset.load(args.source, load_peak_metadata=False)
# Experimental peaks and original metadata are retained. Add a numeric CE
# column parsed from MoNA's unit-bearing text, as supported by column mapping.
for split, indexes in [('train', [4305, 4358]), ('validation', [3765, 3762])]:
    view = records[indexes]
    raw = out / f'raw_{split}.msds'
    view.save(str(raw))
    view = MSDataset.load(str(raw),load_peak_metadata=False)
    metadata = view.metadata.copy()
    metadata['SmokeCollisionEnergy'] = [float(str(value).split()[0]) for value in metadata.CollisionEnergy]
    metadata['OriginalRecordIndex'] = indexes
    dataset = MSDataset(metadata, view.peaks, columns=metadata.columns.tolist())
    dataset.save(str(out / f'{split}.msds'))

base = json.loads((ROOT / 'fredis/presets/fragment/reaction_pattern_sets/nonring_single_bond/nonring_single_bond.fragmenter.pft').read_text())
base['fragment_ion_tree_builder']['max_reaction_count'] = 1
# Small motif definitions exercise new pattern parameters on the actual
# flavanone spectra. They are workflow probes, not validated fragmentation rules.
carbonyl = dict(name='probe_carbonyl', reactant_smarts='[#6:1]-[#6:2](=[#8:3])-[#6:4]',
                products=[dict(name='carbonyl_fragment', smarts='[*:2]=[*:3]')])
ether = dict(name='probe_ether', reactant_smarts='[#6:1]-[#8:2]-[#6:3]',
             products=[dict(name='oxygen_fragment', smarts='[*:2]')])
stages = {'base': [], 'carbonyl': [carbonyl], 'ether': [ether], 'union': [carbonyl, ether]}
progress_path = out/'progress.json'
steps = json.loads(progress_path.read_text()) if args.continue_run and progress_path.exists() else []
env = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def run(name, argv):
    if any(step['step']==name for step in steps):
        print(f'{name}: already passed',flush=True)
        return
    log = out / f'{name}.log'
    start = time.monotonic()
    with log.open('w') as handle:
        result = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} failed: {log}\n{log.read_text()[-5000:]}')
    steps.append({'step': name, 'seconds': round(time.monotonic()-start, 2), 'log': str(log), 'command': [sys.executable, *argv]})
    progress_path.write_text(json.dumps(steps,indent=2))
    print(f'{name}: passed', flush=True)


for name, extra in stages.items():
    config = deepcopy(base)
    config['fragment_ion_tree_builder']['reaction_pattern_set']['patterns'].extend(extra)
    path = out / f'{name}.fragmenter.pft'
    path.write_text(json.dumps(config, indent=2))
    run(f'prepare_{name}', ['-m', 'fredis.cli.main', 'train', 'create-fragment-tree-data',
        '--input', str(out/'train.msds'), '--validation-input', str(out/'validation.msds'),
        '--output-dir', str(out/f'data_{name}'), '--params', str(path),
        '--collision-energy-column', 'SmokeCollisionEnergy', '--minimum-relative-intensity', '0.1',
        '--max-unique-fragment-smiles', '100', '--max-reaction-combinations', '1000'])

small = ['--node-dim','8','--graph-dim','11','--num-heads','2','--num-layers','1',
         '--reaction-hidden-dim','8','--reaction-num-heads','2','--reaction-main-adduct-dim','8',
         '--reaction-state-layers','1','--spectrum-hidden-dim','8','--spectrum-num-heads','2',
         '--spectrum-num-layers','1','--ion-state-hidden-dim','8','--main-adduct-embedding-dim','8',
         '--collision-energy-feature-dim','16','--dropout','0.2','--fragment-tree-dropout','0.2']


def train(name, dataset, source=None, head_only=False, resume=None):
    argv = ['-m','fredis.cli.main','train','fragment-tree',
            '--train-dir',str(out/f'data_{dataset}'/'train_structures'),
            '--val-dir',str(out/f'data_{dataset}'/'validation_structures'),
            '--output-dir',str(out/name),'--epochs','2' if resume else '1',
            '--batch-size','1','--device','cuda','--overwrite','--lr','0.002','--warmup-steps','0',
            '--max-samples','2','--train-log-interval-steps','1',*small]
    if source:
        argv += ['--fine-tune-checkpoint',str(source)]
    if head_only:
        argv += ['--head-only']
    if resume:
        argv += ['--resume',str(resume)]
    run(name + ('_resume' if resume else ''), argv)
    return out/name/'last.pt'


base_ckpt = train('pretrain','base')
carbonyl_ckpt = train('branch_carbonyl','carbonyl',base_ckpt)
ether_ckpt = train('branch_ether','ether',base_ckpt)
sequential_ckpt = train('sequential','union',carbonyl_ckpt)
train('sequential','union',resume=sequential_ckpt)
merged = out/'integrated.pt'
run('integrate', ['-m','fredis.cli.main','train','integrate-patterns','--checkpoints',str(carbonyl_ckpt),str(ether_ckpt),
                  '--head-checkpoint',str(carbonyl_ckpt),'--output',str(merged)])
heads_ckpt = train('integrated_heads','union',merged,head_only=True)


def feature_tensors(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    f = Fragmenter.from_dict(checkpoint['model_config']['fragmenter_params'])
    ids = {p.pattern_id:p.key for p in f.reaction_pattern_set.patterns}
    tensors = {}
    for key,value in checkpoint['model_state_dict'].items():
        match = PATTERN_KEY.search(key)
        if match:
            tensors[(ids[int(match.group(1))], key[:match.start(1)]+'{id}'+key[match.end(1):])] = value
        elif key.startswith(('feature_model.mol_encoder.', 'feature_model.reaction_instance_encoder.')):
            tensors[(None,key)] = value
    return tensors


for before,after in [(base_ckpt,carbonyl_ckpt),(base_ckpt,ether_ckpt),(carbonyl_ckpt,sequential_ckpt),
                     (carbonyl_ckpt,merged),(ether_ckpt,merged),(merged,heads_ckpt)]:
    old,new = feature_tensors(before),feature_tensors(after)
    assert all(key in new and torch.equal(value,new[key]) for key,value in old.items()), (before,after)

# Compare actual final reaction features for one prepared source, not just weights.
prepared = next((out/'data_union/train_structures').rglob('*.preft.pt'))
data = torch.load(prepared,map_location='cpu',weights_only=False)['structure']
reference = load_generator(model_path=str(merged),params_path=None,device=torch.device('cpu'),strict=True)
updated = load_generator(model_path=str(heads_ckpt),params_path=None,device=torch.device('cpu'),strict=True)
with torch.no_grad():
    expected = reference.feature_model.encode_static(data)[0]
    actual = updated.feature_model.encode_static(data)[0]
torch.testing.assert_close(expected,actual,rtol=0,atol=0)
assert expected.numel() > 0
assert len(set(data.reactions.reaction_type[:,0].tolist())) > 1
base_data = torch.load(next((out/'data_base/train_structures').rglob('*.preft.pt')),
                       map_location='cpu',weights_only=False)['structure']
base_model = load_generator(model_path=str(base_ckpt),params_path=None,device=torch.device('cpu'),strict=True)
with torch.no_grad():
    base_features = base_model.feature_model.encode_static(base_data)[0]
base_patterns = base_model.fragmenter.reaction_pattern_set.patterns
max_feature_roundoff = 0.0
for checkpoint_path in (carbonyl_ckpt, ether_ckpt, sequential_ckpt, merged, heads_ckpt):
    expanded = load_generator(model_path=str(checkpoint_path),params_path=None,device=torch.device('cpu'),strict=True)
    new_ids = {p.key:p.pattern_id for p in expanded.fragmenter.reaction_pattern_set.patterns}
    remap = {p.pattern_id:new_ids[p.key] for p in base_patterns}
    types = base_data.reactions.reaction_type.clone()
    types[:,0] = torch.tensor([remap[pid] for pid in types[:,0].tolist()])
    with torch.no_grad():
        features = expanded.feature_model.encode_static(replace(base_data,reactions=replace(base_data.reactions,reaction_type=types)))[0]
    max_feature_roundoff = max(max_feature_roundoff,float((base_features-features).abs().max()))
    torch.testing.assert_close(base_features,features,rtol=1e-5,atol=1e-6)

for name,path in [('base',base_ckpt),('fine_tuned',sequential_ckpt),('integrated',merged),('heads',heads_ckpt)]:
    run(f'predict_{name}', ['-m','fredis.ml.specgen.predict_spectrum','--model',str(path),
         '--smiles','O=C1CC(OC2=CC=CC=C12)C1=CC=CC=C1','--ce','30','--adduct-type','[M+H]+',
         '--output-dir',str(out/f'prediction_{name}'),'--overwrite','--device','cpu','--strict',
         '--max-fragment-nodes','30','--ion-prediction-threshold','0'])
    predicted = MSDataset.load(str(next((out/f'prediction_{name}').glob('*.msds'))))
    assert len(predicted) == 1 and predicted.n_peaks_total > 0
report = {'source':args.source,'train_record_indexes':[4305,4358],'validation_record_indexes':[3765,3762],
          'steps':steps,'immutable_feature_tensors':True,'identical_final_reaction_features_after_head_training':True,
          'base_feature_max_float32_difference':max_feature_roundoff,
          'probe_patterns': [carbonyl,ether], 'scope':'Workflow correctness on four experimental spectra; not predictive-accuracy validation.'}
(out/'report.json').write_text(json.dumps(report,indent=2))
print('All workflows and exact feature-preservation checks passed.',flush=True)
