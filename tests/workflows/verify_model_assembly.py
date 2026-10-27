"""Verify New model assembly on the real spectra prepared by verify_pattern_workflow.py."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import torch
from fredis.domain.fragment import Fragmenter
from fredis.ml.training.fragment_tree_training.pattern_models import export_pattern, inspect_model
from fredis.ml.training.fragment_tree_training.integration import integrate_checkpoints, PATTERN_KEY
from fredis.ml.specgen.predict_spectrum import load_generator

parser=argparse.ArgumentParser()
parser.add_argument('--prepared',default='data/workflow_verification_final')
parser.add_argument('--output',required=True)
args=parser.parse_args()
data=Path(args.prepared).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
base=data/'pretrain/last.pt'
saved=torch.load(base,map_location='cpu',weights_only=False)
base_patterns=inspect_model(base)['patterns'];pid=base_patterns[0]['pattern']['pattern_id']
asset=out/'single.reaction-model.pt';export_pattern(base,pid,asset)
imports=[dict(path=str(asset)),dict(path=str(base),pattern_ids=[p['pattern']['pattern_id'] for p in base_patterns[1:]])]
commands=[]
def run(name,argv):
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,*argv],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError((out/(name+'.log')).read_text()[-6000:])
    commands.append(dict(name=name,command=argv))

def train(name,dataset,imports,*,resume=None):
    config=deepcopy(saved['model_config'])
    config.pop('fine_tuning',None)
    fragmenter=Fragmenter.from_dict(json.loads((data/(dataset+'.fragmenter.pft')).read_text()))
    base_keys={p.key for p in Fragmenter.from_dict(saved['model_config']['fragmenter_params']).reaction_pattern_set.patterns}
    config['reaction_model_params']['pattern_model_params']={str(p.pattern_id):dict(hidden_dim=12,num_layers=2,num_heads=3,dropout=0,ffn_mult=4,max_degree=4,max_spatial_dist=4,max_edge_dist=4) for p in fragmenter.reaction_pattern_set.patterns if p.key not in base_keys}
    arch=out/(name+'.architecture.json');arch.write_text(json.dumps(dict(model_config=config,pretrained_patterns=imports),indent=2))
    argv=['-m','fredis.cli','train','fragment-tree','--train-dir',str(data/('data_'+dataset)/'train_structures'),'--val-dir',str(data/('data_'+dataset)/'validation_structures'),'--output-dir',str(out/name),'--epochs','2' if resume else '1','--batch-size','1','--device','cuda','--lr','0.002','--warmup-steps','0','--max-samples','2','--train-log-interval-steps','1']
    argv+=['--resume',str(resume)] if resume else ['--model-architecture',str(arch)]
    run(name+('_resume' if resume else ''),argv)
    return out/name/'last.pt'
branch=train('assembled','carbonyl',imports)
train('assembled','carbonyl',imports,resume=branch)
sequential=train('sequential','union',[dict(path=str(branch))])
merged=out/'integrated.pt';integrate_checkpoints([branch,data/'branch_ether/last.pt'],branch,merged)
heads=train('heads','union',[dict(path=str(merged))])
# Every feature dependency survives import, training, resume and integration.
def features(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    keys={p.pattern_id:p.key for p in Fragmenter.from_dict(saved['model_config']['fragmenter_params']).reaction_pattern_set.patterns}
    result={}
    for name,tensor in saved['model_state_dict'].items():
        match=PATTERN_KEY.search(name)
        if match:result[(keys[int(match.group(1))],name[:match.start(1)]+'{id}'+name[match.end(1):])]=tensor
        elif name.startswith(('feature_model.mol_encoder.','feature_model.reaction_instance_encoder.')):result[(None,name)]=tensor
    return result
for before,after in [(base,branch),(branch,sequential),(branch,merged),(data/'branch_ether/last.pt',merged),(merged,heads)]:
    a,b=features(before),features(after)
    assert all(key in b and torch.equal(tensor,b[key]) for key,tensor in a.items()),(before,after)
prepared=torch.load(next((data/'data_union/train_structures').rglob('*.preft.pt')),map_location='cpu',weights_only=False)['structure']
generators=[load_generator(model_path=str(p),params_path=None,device=torch.device('cpu'),strict=True) for p in [merged,heads]]
with torch.no_grad():
    a,b=[g.feature_model.encode_static(prepared)[0] for g in generators]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
run('predict',['-m','fredis.ml.specgen.predict_spectrum','--model',str(heads),'--smiles','O=C1CC(OC2=CC=CC=C12)C1=CC=CC=C1','--ce','30','--adduct-type','[M+H]+','--output-dir',str(out/'predictions'),'--device','cpu','--strict','--max-fragment-nodes','30','--ion-prediction-threshold','0'])
from fredis.data.core.MSDataset import MSDataset
predicted=MSDataset.load(str(next((out/'predictions').glob('*.msds'))))
assert len(predicted)==1 and predicted.n_peaks_total>0

(out/'report.json').write_text(json.dumps(dict(commands=commands,complete_pattern_features_preserved=True,head_training_feature_difference=float((a-b).abs().max()),pattern_architectures=inspect_model(merged)['patterns'],scope='Workflow verification on four experimental MoNA spectra, not predictive accuracy'),indent=2))
print(out/'report.json')
