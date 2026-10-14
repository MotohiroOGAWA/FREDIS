"""Exercise preparation-owned pattern selections on four real MoNA spectra."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import sys
import torch
from fredis.domain.fragment import Fragmenter
from fredis.ml.training.fragment_tree_training.pattern_models import inspect_model
from fredis.ml.training.fragment_tree_training.integration import PATTERN_KEY

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--prepared',default='data/workflow_verification_final');args=parser.parse_args()
root=Path(args.prepared).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
originals=out/'source_models';originals.mkdir();base=originals/'base.pt'
shutil.copyfile(root/'pretrain/last.pt',base);saved=torch.load(base,map_location='cpu',weights_only=False)
records=[dict(pattern=item['pattern'],source=str(base),source_pattern_id=item['pattern']['pattern_id']) for item in inspect_model(base)['patterns']]
fragmenter=Fragmenter.from_dict(json.loads((root/'carbonyl.fragmenter.pft').read_text()))
base_keys={p.key for p in Fragmenter.from_dict(saved['model_config']['fragmenter_params']).reaction_pattern_set.patterns}
for p in fragmenter.reaction_pattern_set.patterns:
    if p.key not in base_keys:records.append(dict(pattern=p.to_dict()))
commands=[]
def run(name,argv):
    with (out/(name+'.log')).open('w') as log:result=subprocess.run([sys.executable,*argv],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError((out/(name+'.log')).read_text()[-6000:])
    commands.append(dict(name=name,command=argv))
run('prepare',['-m','fredis.cli','train','create-fragment-tree-data','--input',str(root/'train.msds'),'--validation-input',str(root/'validation.msds'),'--output-dir',str(out/'dataset'),'--params',str(root/'base.fragmenter.pft'),'--pretrained-pattern-model',str(base),'--reaction-models-json',json.dumps([record for record in records if not record.get('source')]),'--collision-energy-column','SmokeCollisionEnergy','--minimum-relative-intensity','0.1','--max-unique-fragment-smiles','100','--max-reaction-combinations','1000'])
# Remove the original pretrained input and relocate the prepared dataset.
shutil.rmtree(originals)
(out/'dataset').rename(out/'relocated_dataset')
# Data fixes definitions/imports. New pattern architecture is chosen only here.
architecture=json.loads(json.dumps(saved['model_config']))
architecture.pop('fine_tuning',None)
architecture['reaction_model_params']['pattern_input_version']=2
architecture['reaction_model_params']['pattern_model_params']={str(p.pattern_id):dict(hidden_dim=12,graph_dim=19,num_layers=3,num_heads=3,dropout=0,descriptors=['TPSA','MolLogP'],neighborhood_hops=3,neighborhood_hidden_dim=7) for p in fragmenter.reaction_pattern_set.patterns if p.key not in base_keys}
architecture_path=out/'architecture.json';architecture_path.write_text(json.dumps({'model_config':architecture},indent=2))
metadata=json.loads((out/'relocated_dataset/preparation_config.json').read_text())
assert all('architecture' not in record for record in metadata['chemistry_config']['reaction_models'])
training=['-m','fredis.cli','train','fragment-tree','--train-dir',str(out/'relocated_dataset/train_structures'),'--val-dir',str(out/'relocated_dataset/validation_structures'),'--output-dir',str(out/'model'),'--batch-size','1','--device','cuda','--lr','0.002','--warmup-steps','0','--max-samples','2','--reaction-main-adduct-dim','8','--reaction-state-layers','1','--spectrum-hidden-dim','8','--spectrum-num-heads','2','--spectrum-num-layers','1','--ion-state-hidden-dim','8','--main-adduct-embedding-dim','8']
run('train',[*training,'--epochs','1','--model-architecture',str(architecture_path)])
checkpoint=out/'model/last.pt'
run('resume',[*training,'--epochs','2','--resume',str(checkpoint)])
new=torch.load(checkpoint,map_location='cpu',weights_only=False)
old_patterns=Fragmenter.from_dict(saved['model_config']['fragmenter_params']).reaction_pattern_set.patterns
new_patterns=Fragmenter.from_dict(new['model_config']['fragmenter_params']).reaction_pattern_set.patterns
by_key={p.key:p.pattern_id for p in new_patterns};mapping={p.pattern_id:by_key[p.key] for p in old_patterns}
for name,tensor in saved['model_state_dict'].items():
    match=PATTERN_KEY.search(name)
    if match:target=name[:match.start(1)]+str(mapping[int(match.group(1))])+name[match.end(1):]
    elif name.startswith(('feature_model.mol_encoder.','feature_model.reaction_instance_encoder.')):target=name
    else:continue
    assert torch.equal(tensor,new['model_state_dict'][target]),name
assert set(new['model_config']['fine_tuning']['frozen_pattern_ids'])==set(mapping.values())
added=next(p for p in new_patterns if p.key not in base_keys)
assert new['model_config']['reaction_model_params']['pattern_model_params'][str(added.pattern_id)]['num_layers']==3
# Integrate the new raw-neighborhood model with a previously trained legacy branch.
from fredis.ml.training.fragment_tree_training.integration import integrate_checkpoints
integrated=out/'integrated.pt';integrate_checkpoints([checkpoint,root/'branch_ether/last.pt'],checkpoint,integrated)
run('prepare_union',['-m','fredis.cli','train','create-fragment-tree-data','--input',str(root/'train.msds'),'--validation-input',str(root/'validation.msds'),'--output-dir',str(out/'union'),'--params',str(root/'base.fragmenter.pft'),'--pretrained-pattern-model',str(integrated),'--collision-energy-column','SmokeCollisionEnergy','--minimum-relative-intensity','0.1','--max-unique-fragment-smiles','100','--max-reaction-combinations','1000'])
archived_integrated=next((out/'union/pretrained_models').glob('*.pt'))
integrated.unlink()
head_args=training.copy()
for flag,value in [('--train-dir',str(out/'union/train_structures')),('--val-dir',str(out/'union/validation_structures')),('--output-dir',str(out/'heads'))]:head_args[head_args.index(flag)+1]=value
run('train_heads',[*head_args,'--epochs','1'])
from fredis.ml.specgen.predict_spectrum import load_generator
prepared=torch.load(next((out/'union/train_structures').rglob('*.preft.pt')),map_location='cpu',weights_only=False)['structure']
reference=load_generator(model_path=str(archived_integrated),params_path=None,device=torch.device('cpu'),strict=True)
updated=load_generator(model_path=str(out/'heads/last.pt'),params_path=None,device=torch.device('cpu'),strict=True)
with torch.no_grad():
    before=reference.feature_model.encode_static(prepared)[0];after=updated.feature_model.encode_static(prepared)[0]
    torch.testing.assert_close(before,after,rtol=0,atol=0)
run('predict',['-m','fredis.ml.specgen.predict_spectrum','--model',str(checkpoint),'--smiles','O=C1CC(OC2=CC=CC=C12)C1=CC=CC=C1','--ce','30','--adduct-type','[M+H]+','--output-dir',str(out/'prediction'),'--strict','--device','cpu','--max-fragment-nodes','30','--ion-prediction-threshold','0'])
from fredis.libs.msentity.msentity.core.MSDataset import MSDataset
assert MSDataset.load(str(next((out/'prediction').glob('*.msds')))).n_peaks_total>0
(out/'report.json').write_text(json.dumps(dict(commands=commands,patterns_fixed_during_preparation=True,original_pretrained_files_removed=True,dataset_relocated=True,imported_weights_preserved=True,new_architecture_chosen_during_training=True,graphormer_hidden_dim=12,neighborhood_hops=3,neighborhood_hidden_dim=7,head_training_feature_difference=float((before-after).abs().max())),indent=2));print(out/'report.json')
