"""Reproduce the small real-data branching validation inputs without altering originals."""
import json
from pathlib import Path
from rdkit import Chem
from fredis.libs.msentity.msentity import MSDataset
from fredis.ml.data_preparation.fragment_tree.create_training_data import main

root=Path('data/train_preprocessing/test/fragment_tree_smoke')
root.mkdir(parents=True,exist_ok=True)
original=json.loads(Path('data/train_preprocessing/test/preparation_config.json').read_text())
# Only the chemistry is reused; model architecture is chosen by run_smoke.sh at training time.
chemistry={key:original['chemistry_config'][key] for key in ('fragmenter_params','symbols')}
used=set();provenance={}
for key,name,limit in [('input','train',8),('validation_input','validation',4)]:
    dataset=MSDataset.load(original[key],load_peak_metadata=False)
    selected=[];groups={}
    for i,(smiles,adduct) in enumerate(zip(dataset['SMILES'],dataset['AdductType'])):
        if adduct not in ('[M+H]+','[M+Na]+'):continue
        mol=Chem.MolFromSmiles(str(smiles))
        if mol is None or not 5<=mol.GetNumHeavyAtoms()<=12:continue
        canonical=Chem.MolToSmiles(mol)
        if name=='validation' and canonical in used:continue
        if canonical not in groups and len(groups)>=limit:continue
        if len(groups.get(canonical,[]))>=2:continue
        groups.setdefault(canonical,[]).append(i);selected.append(i)
        if len(groups)==limit and len(selected)>=limit*2:break
    used.update(groups)
    dataset[selected].save(str(root/f'{name}.msds'))
    provenance[name]=dict(source=original[key],record_indexes=selected,unique_sources=len(groups),samples=len(selected))
(root/'selection.json').write_text(json.dumps(provenance,indent=2))
(root/'chemistry_config.json').write_text(json.dumps(chemistry,indent=2))
main(['--input',str(root/'train.msds'),'--validation-input',str(root/'validation.msds'),
      '--output-dir',str(root/'prepared'),'--params-json',json.dumps(chemistry),
      '--validation-ratio','0.5','--num-workers','1',
      '--max-unique-fragment-smiles',str(original.get('max_unique_fragment_smiles',-1)),
      '--max-reaction-combinations',str(original.get('max_reaction_combinations',-1)),'--overwrite','1'])
