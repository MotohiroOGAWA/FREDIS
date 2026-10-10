"""Reload the produced checkpoint and compare genuine inference before/after training."""
import json
from pathlib import Path
import torch
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.specgen.predict_spectrum import load_generator
from fredis.ml.training.fragment_tree_training.spectrum_validation import validate_spectra
root=Path('data/train_preprocessing/test/fragment_tree_smoke')
checkpoint=torch.load(root/'training/last.pt',map_location='cpu',weights_only=False)
c=checkpoint['model_config']
torch.manual_seed(checkpoint['training_settings']['seed'])
initial=create_spectrum_generator(c).cuda().eval()
trained=load_generator(model_path=str(root/'training/last.pt'),params_path=None,device=torch.device('cuda'),strict=True)
summary={}
for split in ('train','validation'):
    pieces=[SourceReactionStructure.load(path) for path in sorted((root/f'prepared/{split}_structures').rglob('*.preft.pt'))]
    for name,model in [('initial',initial),('trained',trained)]:
        summary[f'{split}_{name}']=validate_spectra(model,pieces,root/'comparison',f'{split}_{name}')
(root/'comparison/summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
