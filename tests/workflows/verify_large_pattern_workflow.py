"""50-compound pretraining, two 10-compound branches, all associated spectra."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from fredis.domain.fragment import Fragmenter
from fredis.domain.mass.parse_ce import parse_ce_to_ev
from fredis.libs.mmkit.mmkit import Adduct
from fredis.libs.msentity.msentity.core.MSDataset import MSDataset
from fredis.ml.specgen.predict_spectrum import load_generator
from fredis.ml.training.fragment_tree_training.integration import PATTERN_KEY
from fredis.ml.training.fragment_tree_training.pattern_models import export_pattern

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
parser.add_argument('--source',default='/workspaces/CLEFTS/raw_data/MoNA/MoNA-export-LC-MS-MS_Positive_Mode_high_cf.msds')
parser.add_argument('--continue-run',action='store_true')
args=parser.parse_args()
repo=Path(os.environ.get('FREDIS_REPO','/workspaces/CLEFTS/mnt/app')).resolve()
out=Path(args.output).resolve()
if out.is_relative_to(repo):raise ValueError('Output must be outside the Git repository')
out.mkdir(parents=True,exist_ok=args.continue_run)
(out/'review').mkdir(exist_ok=True);(out/'configs').mkdir(exist_ok=True)
if Path(__file__).resolve() != (out/'review/workflow.py').resolve():
    shutil.copyfile(Path(__file__),out/'review/workflow.py')
torch.set_num_threads(1);RDLogger.DisableLog('rdApp.*')
progress=out/'progress.json'
commands=json.loads(progress.read_text()) if args.continue_run and progress.exists() else []
def run(name,argv):
    if any(item['step']==name for item in commands):return
    print(name,flush=True);start=time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,*argv],cwd=repo,env={**os.environ,'PYTHONPATH':str(repo),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name}: '+(out/(name+'.log')).read_text()[-8000:])
    commands.append(dict(step=name,args=argv,seconds=round(time.monotonic()-start,2)))
    progress.write_text(json.dumps(commands,indent=2))

base=json.loads((repo/'fredis/presets/fragment/reaction_pattern_sets/nonring_single_bond/nonring_single_bond.fragmenter.pft').read_text())
base['fragment_ion_tree_builder']['max_reaction_count']=1
base['fragment_ion_tree_builder']['fragment_ion_adduct_rule_set']['adduct_rules'].append(dict(name='pos_K',adduct_type='[M+K]+',fragment_ion_adducts=['[M+K]+','[M-H]+','[M+H]+']))
(out/'configs/base.fragmenter.pft').write_text(json.dumps(base,indent=2))
flav_pattern=dict(name='flavonoid_carbonyl_ring',reactant_smarts='[C;R:1](=[O:2])~[C;R:3]~[C;R:4]~[O;R:5]',products=[dict(name='carbonyl',smarts='[*:1]=[*:2]')])
lipid_pattern=dict(name='lipid_long_chain_single_bond',reactant_smarts='[C;$([C;!R]~[C;!R]~[C;!R]~[C;!R]~[C;!R]~[C;!R]~[C;!R]~[C;!R](=O)-O-C):1]-&!@[C:2]',products=[dict(name='chain_side',smarts='[*:1]'),dict(name='other_side',smarts='[*:2]')])
(out/'configs/flavonoid.pattern.json').write_text(json.dumps(flav_pattern,indent=2));(out/'configs/lipid.pattern.json').write_text(json.dumps(lipid_pattern,indent=2))

if not (out/'selection.json').exists():
    print('Loading complete source and selecting compounds',flush=True)
    source=MSDataset.load(args.source,load_peak_metadata=False);meta=source.metadata
    fragmenter=Fragmenter.from_dict(base)
    mols={};canonical={};adduct_valid={};energies={}
    for text in meta.SMILES.astype(str).unique():
        mol=Chem.MolFromSmiles(text)
        key=Chem.MolToSmiles(mol,isomericSmiles=True) if mol is not None else None
        canonical[text]=key
        if key is not None:mols[key]=mol
    groups={}
    for index,text in enumerate(meta.SMILES.astype(str)):
        key=canonical[text]
        if key is not None:groups.setdefault(key,[]).append(index)
    eligible={}
    for key,indices in groups.items():
        mol=mols[key]
        if not 8<=mol.GetNumHeavyAtoms()<=65 or Chem.GetFormalCharge(mol)!=0 or '.' in key:continue
        if any(a.GetSymbol() not in {'H','C','N','O','P','S','F','Cl','Br','I'} for a in mol.GetAtoms()):continue
        good=True
        for idx in indices:
            row=meta.iloc[idx];adduct=str(row.AdductType)
            if adduct not in adduct_valid:
                try:fragmenter.get_index_by_adduct_type(Adduct.parse(adduct));adduct_valid[adduct]=True
                except Exception:adduct_valid[adduct]=False
            try:
                mz=float(row.PrecursorMZ)
                cache=(str(row.CollisionEnergy),mz if '%' in str(row.CollisionEnergy) else None)
                if cache not in energies:energies[cache]=parse_ce_to_ev(row.CollisionEnergy,mz)
                ce=energies[cache]
                valid=adduct_valid[adduct] and math.isfinite(mz) and mz>0 and ce is not None and math.isfinite(float(ce)) and float(ce)>=0
            except Exception:valid=False
            if not valid:good=False;break
        if good:eligible[key]=indices
    fq=Chem.MolFromSmarts(flav_pattern['reactant_smarts']);lq=Chem.MolFromSmarts(lipid_pattern['reactant_smarts'])
    flav=[];lipid=[]
    for key,indices in eligible.items():
        classes=set(meta.iloc[indices].Class.astype(str));mol=mols[key]
        if 'Flavonoids' in classes and mol.HasSubstructMatch(fq):flav.append(key)
        if any(any(word in c for word in ['Fatty Acyl','Glycerolipid','Glycerophospholipid']) for c in classes) and mol.HasSubstructMatch(lq):lipid.append(key)
    rng=random.Random(42)
    # Prefer multi-spectrum compounds, then diversify deterministically.
    rng.shuffle(flav);flav.sort(key=lambda k:(len(eligible[k])<3,len(eligible[k])>30))
    rng.shuffle(lipid);lipid.sort(key=lambda k:(len(eligible[k])<2,len(eligible[k])>30))
    assert len(flav)>=10 and len(lipid)>=10,(len(flav),len(lipid))
    selected_flav=flav[:10];selected_lipid=lipid[:10]
    used=set(selected_flav+selected_lipid)
    candidates=[key for key in eligible if key not in used and len(eligible[key])>=2 and mols[key].GetNumHeavyAtoms()<=35 and any(mols[key].HasSubstructMatch(p.reactant_query) for p in fragmenter.reaction_pattern_set.patterns)]
    rng.shuffle(candidates)
    # Round robin across chemical classes rather than taking only one family.
    by_class={}
    for key in candidates:by_class.setdefault(str(meta.iloc[eligible[key][0]].Class),[]).append(key)
    selected_base=[]
    while len(selected_base)<50:
        for label in sorted(by_class):
            if by_class[label]:selected_base.append(by_class[label].pop())
            if len(selected_base)==50:break
    stage_keys={'base':selected_base,'flavonoid':selected_flav,'lipid':selected_lipid}
    selection=dict(source=str(Path(args.source).resolve()),identity='RDKit canonical isomeric SMILES',seed=42,source_spectra=len(meta),source_compounds=len(groups),eligible_compounds=len(eligible),flavonoid_candidates=len(flav),lipid_candidates=len(lipid),stages={})
    (out/'raw').mkdir(exist_ok=True)
    for stage,keys in stage_keys.items():
        train_count=40 if stage=='base' else 8
        rows=[]
        for n,key in enumerate(keys):
            indices=eligible[key];first=meta.iloc[indices[0]]
            rows.append(dict(compound_id=key,name=str(first.Name),chemical_class=str(first.Class),split='train' if n<train_count else 'validation',spectrum_count=len(indices),source_record_indices=indices,adducts=sorted(set(meta.iloc[indices].AdductType.astype(str)))))
        selection['stages'][stage]=rows
        for split in ['train','validation']:
            selected=[r for r in rows if r['split']==split];indices=[i for r in selected for i in r['source_record_indices']]
            view=source[indices];tmp=out/'raw'/f'{stage}_{split}.msds';view.save(str(tmp));view=MSDataset.load(str(tmp),load_peak_metadata=False)
            metadata=view.metadata.copy();metadata['OriginalRecordIndex']=indices;metadata['OriginalSMILES']=metadata.SMILES
            metadata['SMILES']=[canonical[str(meta.iloc[i].SMILES)] for i in indices]
            metadata['VerificationCollisionEnergy']=[float(energies[(str(meta.iloc[i].CollisionEnergy),float(meta.iloc[i].PrecursorMZ) if '%' in str(meta.iloc[i].CollisionEnergy) else None)]) for i in indices]
            MSDataset(metadata,view.peaks,columns=metadata.columns.tolist()).save(str(tmp))
    selection['stages']['union']=sum((selection['stages'][s] for s in ['base','flavonoid','lipid']),[])
    for split in ['train','validation']:
        MSDataset.concat([MSDataset.load(str(out/'raw'/f'{stage}_{split}.msds'),load_peak_metadata=False) for stage in ['base','flavonoid','lipid']]).save(str(out/'raw'/f'union_{split}.msds'))
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
selection=json.loads((out/'selection.json').read_text())
for stage,rows in selection['stages'].items():print(stage,len(rows),'compounds',sum(r['spectrum_count'] for r in rows),'spectra',flush=True)

preparation_checks=[]
def prepare(label,patterns=(),model=None):
    argv=['-m','fredis.cli','train','create-fragment-tree-data','--input',str(out/'raw'/f'{label}_train.msds'),'--validation-input',str(out/'raw'/f'{label}_validation.msds'),'--output-dir',str(out/('data_'+label)),'--params',str(out/'configs/base.fragmenter.pft'),'--collision-energy-column','VerificationCollisionEnergy','--minimum-relative-intensity','0.1','--max-unique-fragment-smiles','500','--max-reaction-combinations','5000']
    if patterns:argv+=['--reaction-models-json',json.dumps([dict(pattern=p) for p in patterns])]
    if model:argv+=['--pretrained-pattern-model',str(model)]
    run('prepare_'+label,argv)
    invalid=json.loads((out/('data_'+label)/'invalid_records.json').read_text());assert not any(invalid.values()),invalid
    for split in ['train','validation']:
        expected=[r for r in selection['stages'][label] if r['split']==split]
        manifest=pd.read_csv(out/('data_'+label)/(split+'_structures')/'manifest.tsv',sep='\t')
        assert len(manifest)==len(expected),(label,split,len(manifest),len(expected))
        assert (manifest.status=='completed').all(),manifest.loc[manifest.status!='completed'].to_dict('records')
        assert int(manifest.rejected_sample_count.sum())==0,manifest.loc[manifest.rejected_sample_count>0].to_dict('records')
        count=sum(r['spectrum_count'] for r in expected)
        assert int(manifest.num_valid_samples.sum())==count,(label,split,manifest.num_valid_samples.sum(),count)
        preparation_checks.append(dict(stage=label,split=split,compounds=len(expected),all_spectra=count,prepared_samples=count,rejected=0))

small=['--node-dim','8','--graph-dim','11','--num-heads','2','--num-layers','1','--reaction-hidden-dim','8','--reaction-num-heads','2','--reaction-main-adduct-dim','8','--reaction-state-layers','1','--spectrum-hidden-dim','8','--spectrum-num-heads','2','--spectrum-num-layers','1','--ion-state-hidden-dim','8','--main-adduct-embedding-dim','8','--collision-energy-feature-dim','16','--dropout','0','--intensity-descriptors','TPSA','MolLogP']
def train(label,architecture=None,resume=False):
    argv=['-m','fredis.cli','train','fragment-tree','--train-dir',str(out/('data_'+label)/'train_structures'),'--val-dir',str(out/('data_'+label)/'validation_structures'),'--output-dir',str(out/label),'--epochs','2' if resume else '1','--batch-size','1','--device','cuda','--lr','0.002','--warmup-steps','0','--max-samples','32',*small]
    if architecture:
        path=out/'configs'/f'{label}.input-architecture.json';path.write_text(json.dumps({'model_config':architecture},indent=2));argv+=['--model-architecture',str(path)]
    if resume:argv+=['--resume',str(out/label/'last.pt')]
    run('train_'+label+('_resume' if resume else ''),argv)
    summary=json.loads((out/label/'dataset_summary.json').read_text())
    for split in ['train','validation']:
        expected=sum(r['spectrum_count'] for r in selection['stages'][label] if r['split']==split)
        assert summary[split]['samples']==expected and summary[split]['filtered_samples']==expected,(label,split,summary[split],expected)
        history=json.loads((out/label/'training_report.json').read_text())['history']
        assert all(row[f'{split}/samples']==expected for row in history),(label,split,history)
    return out/label/'last.pt'
prepare('base');pretrain=train('base')
saved=torch.load(pretrain,map_location='cpu',weights_only=False)
for label,pattern in [('flavonoid',flav_pattern),('lipid',lipid_pattern)]:
    prepare(label,[pattern],pretrain)
    meta=json.loads((out/('data_'+label)/'preparation_config.json').read_text());fragmenter=Fragmenter.from_dict(meta['chemistry_config']['fragmenter_params'])
    pid=next(p.pattern_id for p in fragmenter.reaction_pattern_set.patterns if p.name==pattern['name'])
    arch=deepcopy(saved['model_config']);arch.pop('fine_tuning',None)
    arch['reaction_model_params']['pattern_model_params']={str(pid):dict(hidden_dim=8,graph_dim=13,num_layers=2,num_heads=2,dropout=0,neighborhood_hops=3,neighborhood_hidden_dim=8)}
    train(label,arch)
train('lipid',resume=True)
merged=out/'integrated.pt'
run('integrate',['-m','fredis.cli','train','integrate-patterns','--checkpoints',str(out/'flavonoid/last.pt'),str(out/'lipid/last.pt'),'--head-checkpoint',str(out/'lipid/last.pt'),'--output',str(merged)])
prepare('union',model=merged)
archive=next((out/'data_union/pretrained_models').glob('*.pt'))
if merged.exists():merged.unlink()
heads=train('union')

def tensors(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);patterns=Fragmenter.from_dict(ck['model_config']['fragmenter_params']).reaction_pattern_set.patterns;ids={p.pattern_id:p.key for p in patterns};result={}
    for key,value in ck['model_state_dict'].items():
        match=PATTERN_KEY.search(key)
        if match:result[(ids[int(match.group(1))],key[:match.start(1)]+'{id}'+key[match.end(1):])]=value
        elif key.startswith(('feature_model.mol_encoder.','feature_model.reaction_instance_encoder.')):result[(None,key)]=value
    return result
weight_checks=[]
for before,after in [(pretrain,out/'flavonoid/last.pt'),(pretrain,out/'lipid/last.pt'),(out/'flavonoid/last.pt',archive),(out/'lipid/last.pt',archive),(archive,heads)]:
    old,new=tensors(before),tensors(after);assert all(k in new and torch.equal(v,new[k]) for k,v in old.items()),(before,after)
    weight_checks.append(dict(before=str(before),after=str(after),compared_tensors=len(old),bitwise_equal=True))
reference=load_generator(model_path=str(archive),params_path=None,device=torch.device('cpu'),strict=True)
updated=load_generator(model_path=str(heads),params_path=None,device=torch.device('cpu'),strict=True)
pattern_ids={p.name:p.pattern_id for p in updated.fragmenter.reaction_pattern_set.patterns};routing=[];maximum=0
for split in ['train','validation']:
    for path in sorted((out/'data_union'/(split+'_structures')).rglob('*.preft.pt')):
        data=torch.load(path,map_location='cpu',weights_only=False)['structure']
        with torch.no_grad():before=reference.feature_model.encode_static(data)[0];after=updated.feature_model.encode_static(data)[0]
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        if before.numel():maximum=max(maximum,float((before-after).abs().max()))
        present=set(data.reaction_type[:,0].tolist());routing.append(dict(file=str(path),split=split,pattern_ids=sorted(present)))
assert sum(pattern_ids[flav_pattern['name']] in r['pattern_ids'] for r in routing)>=10
assert sum(pattern_ids[lipid_pattern['name']] in r['pattern_ids'] for r in routing)>=10
models=out/'models';models.mkdir(exist_ok=True)
shutil.copyfile(archive,models/'integrated_features.pt');shutil.copyfile(heads,models/'integrated_final.pt')
for name in [flav_pattern['name'],lipid_pattern['name']]:
    asset=models/(name+'.reaction-model.pt')
    if not asset.exists():export_pattern(heads,pattern_ids[name],asset)
for stage in ['base','flavonoid','lipid','union']:
    ck=torch.load(out/stage/'last.pt',map_location='cpu',weights_only=False)
    (out/'configs'/f'{stage}.model.json').write_text(json.dumps({'model_config':ck['model_config']},indent=2))
predictions=[]
for stage in ['flavonoid','lipid']:
    raw=MSDataset.load(str(out/'raw'/f'{stage}_train.msds'),load_peak_metadata=False)
    row=raw.metadata.iloc[0]
    run('predict_'+stage,['-m','fredis.ml.specgen.predict_spectrum','--model',str(models/'integrated_final.pt'),'--smiles',str(row.SMILES),'--ce',str(row.VerificationCollisionEnergy),'--adduct-type',str(row.AdductType),'--output-dir',str(out/('prediction_'+stage)),'--strict','--device','cpu','--max-fragment-nodes','100','--ion-prediction-threshold','0'])
    predicted=MSDataset.load(str(out/('prediction_'+stage)/'predicted.msds'))
    assert predicted.n_peaks_total>0
    predictions.append(dict(stage=stage,compound=str(row.Name),adduct=str(row.AdductType),collision_energy=float(row.VerificationCollisionEnergy),peaks=predicted.n_peaks_total))
report=dict(predictions=predictions,source=selection['source'],selection=str(out/'selection.json'),commands=commands,preparation_checks=preparation_checks,weight_checks=weight_checks,routing=routing,pattern_ids=pattern_ids,head_training_feature_difference=maximum,all_selected_spectra_used=True,original_integrated_file_removed=True,scope='Workflow verification; not an accuracy benchmark')
(out/'report.json').write_text(json.dumps(report,indent=2))
print(out/'report.json',flush=True)
