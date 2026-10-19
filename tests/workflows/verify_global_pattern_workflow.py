"""Real MoNA smoke workflow; all data/checkpoints are written outside Git.

PYTHONPATH=. python tests/workflows/verify_global_pattern_workflow.py --output /workspaces/CLEFTS/global_pattern_verification
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from fredis.libs.msentity.msentity.core.MSDataset import MSDataset
from fredis.domain.fragment import Fragmenter
from fredis.ml.training.fragment_tree_training.integration import PATTERN_KEY
from fredis.ml.specgen.predict_spectrum import load_generator

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--continue-run',action='store_true');args=parser.parse_args()
repo=Path(__file__).resolve().parents[2]; out=Path(args.output).resolve()
if out.is_relative_to(repo):raise ValueError('Verification output must be outside the Git repository')
out.mkdir(parents=True,exist_ok=args.continue_run)
torch.set_num_threads(1)
commands=json.loads((out/'progress.json').read_text()) if args.continue_run and (out/'progress.json').exists() else []
def run(name,argv):
    if any(c['step']==name for c in commands): return
    print(name,flush=True)
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,*argv],cwd=repo,env={**os.environ,'PYTHONPATH':str(repo),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError((out/(name+'.log')).read_text()[-7000:])
    commands.append(dict(step=name,args=argv));(out/'progress.json').write_text(json.dumps(commands,indent=2))

flav=MSDataset.load('/workspaces/CLEFTS/raw_data/MoNA/flavonoids/mona_pos_cf_flavonoids.msds',load_peak_metadata=False)
lipid=MSDataset.load('/workspaces/CLEFTS/raw_data/MoNA/MoNA-export-LC-MS-MS_Positive_Mode_high_cf.msds',load_peak_metadata=False)
# Keep measured peaks; normalize the experimental collision energy into a numeric column.
# Separate molecules for training and validation.
for label,dataset,index in [('flav_train',flav,4305),('flav_val',flav,3765),('lipid_train',lipid,40812),('lipid_val',lipid,40826)]:
    selected=dataset[[index]]; raw=out/(label+'.msds');selected.save(str(raw));selected=MSDataset.load(str(raw),load_peak_metadata=False)
    metadata=selected.metadata.copy();metadata['SmokeCollisionEnergy']=[float(str(v).split()[0]) for v in metadata.CollisionEnergy];metadata['OriginalRecordIndex']=[index]
    MSDataset(metadata,selected.peaks,columns=metadata.columns.tolist()).save(str(raw))
    print(label,metadata[['Name','SMILES','CollisionEnergy']].to_dict('records'),flush=True)
for split in ['train','val']:
    MSDataset.concat([MSDataset.load(str(out/(f'{kind}_{split}.msds')),load_peak_metadata=False) for kind in ['flav','lipid']]).save(str(out/(split+'.msds')))
# Each CLI input is a single dataset containing both compound classes.
base=json.loads((repo/'fredis/presets/fragment/reaction_pattern_sets/nonring_single_bond/nonring_single_bond.fragmenter.pft').read_text());base['fragment_ion_tree_builder']['max_reaction_count']=1
(out/'base.fragmenter.pft').write_text(json.dumps(base,indent=2))
flav_pattern=dict(name='flavonoid_carbonyl_ring',reactant_smarts='[C;R:1](=[O:2])~[C;R:3]~[C;R:4]~[O;R:5]',products=[dict(name='carbonyl',smarts='[*:1]=[*:2]')])
lipid_pattern=dict(name='lipid_long_chain_single_bond',reactant_smarts='[C;$([C]~[C]~[C]~[C]~[C]~[C]~[C]~[C](=O)-O-C):1]-&!@[C:2]',products=[dict(name='chain_side',smarts='[*:1]'),dict(name='other_side',smarts='[*:2]')])

def prepare(label,patterns=(),model=None):
    argv=['-m','fredis.cli','train','create-fragment-tree-data','--input',str(out/'train.msds'),'--validation-input',str(out/'val.msds'),'--output-dir',str(out/('data_'+label)),'--params',str(out/'base.fragmenter.pft'),'--collision-energy-column','SmokeCollisionEnergy','--minimum-relative-intensity','0.1','--max-unique-fragment-smiles','100','--max-reaction-combinations','1000']
    if patterns:argv+=['--reaction-models-json',json.dumps([dict(pattern=p) for p in patterns])]
    if model:argv+=['--pretrained-pattern-model',str(model)]
    run('prepare_'+label,argv)
    invalid=json.loads((out/('data_'+label)/'invalid_records.json').read_text())
    assert not any(invalid.values()), invalid
    for split in ['train_structures','validation_structures']:
        assert len(list((out/('data_'+label)/split).rglob('*.preft.pt')))==2

small=['--node-dim','8','--graph-dim','11','--num-heads','2','--num-layers','1','--reaction-hidden-dim','8','--reaction-num-heads','2','--reaction-main-adduct-dim','8','--reaction-state-layers','1','--spectrum-hidden-dim','8','--spectrum-num-heads','2','--spectrum-num-layers','1','--ion-state-hidden-dim','8','--main-adduct-embedding-dim','8','--collision-energy-feature-dim','16','--dropout','0']
def train(label,architecture=None,resume=False):
    argv=['-m','fredis.cli','train','fragment-tree','--train-dir',str(out/('data_'+label)/'train_structures'),'--val-dir',str(out/('data_'+label)/'validation_structures'),'--output-dir',str(out/label),'--epochs','2' if resume else '1','--batch-size','1','--device','cuda','--lr','0.002','--warmup-steps','0','--max-samples','2',*small]
    if architecture:argv+=['--model-architecture-json',json.dumps({'model_config':architecture})]
    if resume:argv+=['--resume',str(out/label/'last.pt')]
    run('train_'+label+('_resume' if resume else ''),argv);return out/label/'last.pt'

prepare('base');pretrain=train('base')
saved=torch.load(pretrain,map_location='cpu',weights_only=False)
for label,pattern in [('flavonoid',flav_pattern),('lipid',lipid_pattern)]:
    prepare(label,[pattern],pretrain)
    meta=json.loads((out/('data_'+label)/'preparation_config.json').read_text())
    fragmenter=Fragmenter.from_dict(meta['chemistry_config']['fragmenter_params'])
    pid=next(p.pattern_id for p in fragmenter.reaction_pattern_set.patterns if p.name==pattern['name'])
    arch=deepcopy(saved['model_config']);arch.pop('fine_tuning',None)
    arch['reaction_model_params']['pattern_model_params']={str(pid):dict(hidden_dim=8,graph_dim=13,num_layers=2,num_heads=2,dropout=0,neighborhood_hops=3,neighborhood_hidden_dim=8,global_num_layers=3 if label=='lipid' else 0,global_hidden_dim=8,global_num_heads=2)}
    train(label,arch)
train('lipid',resume=True)
merged=out/'integrated.pt';run('integrate',['-m','fredis.cli','train','integrate-patterns','--checkpoints',str(out/'flavonoid/last.pt'),str(out/'lipid/last.pt'),'--head-checkpoint',str(out/'lipid/last.pt'),'--output',str(merged)])
prepare('union',model=merged)
# The union dataset includes the entire imported model, so training survives removal of its original.
archive=next((out/'data_union/pretrained_models').glob('*.pt'));merged.unlink(); heads=train('union')

def tensors(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);patterns=Fragmenter.from_dict(checkpoint['model_config']['fragmenter_params']).reaction_pattern_set.patterns;ids={p.pattern_id:p.key for p in patterns};result={}
    for key,value in checkpoint['model_state_dict'].items():
        match=PATTERN_KEY.search(key)
        if match:result[(ids[int(match.group(1))],key[:match.start(1)]+'{id}'+key[match.end(1):])]=value
        elif key.startswith(('feature_model.mol_encoder.','feature_model.reaction_instance_encoder.')):result[(None,key)]=value
    return result
for before,after in [(pretrain,out/'flavonoid/last.pt'),(pretrain,out/'lipid/last.pt'),(out/'flavonoid/last.pt',archive),(out/'lipid/last.pt',archive),(archive,heads)]:
    old,new=tensors(before),tensors(after);assert all(k in new and torch.equal(v,new[k]) for k,v in old.items()),(before,after)
reference=load_generator(model_path=str(archive),params_path=None,device=torch.device('cpu'),strict=True)
updated=load_generator(model_path=str(heads),params_path=None,device=torch.device('cpu'),strict=True)
pattern_ids={p.name:p.pattern_id for p in updated.fragmenter.reaction_pattern_set.patterns};routing=[];maximum=0
for path in sorted((out/'data_union/train_structures').rglob('*.preft.pt')):
    data=torch.load(path,map_location='cpu',weights_only=False)['structure']
    with torch.no_grad():before=reference.feature_model.encode_static(data)[0];after=updated.feature_model.encode_static(data)[0]
    torch.testing.assert_close(before,after,rtol=0,atol=0);maximum=max(maximum,float((before-after).abs().max()))
    present=set(data.reaction_type[:,0].tolist());routing.append(dict(file=str(path),pattern_ids=sorted(present)))
assert any(pattern_ids[flav_pattern['name']] in r['pattern_ids'] for r in routing)
assert any(pattern_ids[lipid_pattern['name']] in r['pattern_ids'] for r in routing)
assert all(not {pattern_ids[flav_pattern['name']],pattern_ids[lipid_pattern['name']]}.issubset(set(r['pattern_ids'])) for r in routing)
for label in ['flav_train','lipid_train']:
    smiles=MSDataset.load(str(out/(label+'.msds')),load_peak_metadata=False).metadata.SMILES.iloc[0]
    run('predict_'+label,['-m','fredis.ml.specgen.predict_spectrum','--model',str(heads),'--smiles',smiles,'--ce','30','--adduct-type','[M+H]+','--output-dir',str(out/('prediction_'+label)),'--strict','--device','cpu','--max-fragment-nodes','100','--ion-prediction-threshold','0'])
    assert MSDataset.load(str(next((out/('prediction_'+label)).glob('*.msds')))).n_peaks_total>0
(out/'report.json').write_text(json.dumps(dict(commands=commands,routing=routing,pattern_ids=pattern_ids,imported_weights_preserved=True,independent_branches=True,global_gnn_layers=3,head_training_feature_difference=maximum,original_integrated_file_removed=True,scope='Workflow smoke test on measured spectra; accuracy and cleavage chemistry are not validated'),indent=2))
print(out/'report.json',flush=True)
