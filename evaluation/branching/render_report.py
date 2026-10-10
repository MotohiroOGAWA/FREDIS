"""Export training curves and the actual inference similarity distribution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('data/train_preprocessing/test/fragment_tree_smoke')
metrics=json.loads((root/'training/metrics.json').read_text())
history=metrics['history']
results=json.loads((root/f"training/spectrum_validation/epoch_{history[-1]['epoch']}.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for key,label in [('train_loss','Train (16 spectra)'),('validation_loss','Validation (8 spectra)')]:
    axes[0].plot([r['epoch'] for r in history],[r[key] for r in history],marker='.',label=label)
axes[0].set(xlabel='Epoch',ylabel='Teacher-forced total loss',title='Small real-data training')
axes[0].legend();axes[0].grid(alpha=.2)
axes[1].hist([r['cosine_similarity'] for r in results['spectra']],bins=results['summary']['histogram_edges'],edgecolor='white')
axes[1].set(xlabel='Cosine to original spectrum',ylabel='Spectra',xlim=(0,1),title='Free-running validation (n=8)')
axes[1].axvline(results['summary']['cosine_mean'],color='darkred',label=f"Mean {results['summary']['cosine_mean']:.3f}")
axes[1].legend();axes[1].set_yticks(range(5))
fig.savefig(root/'verification.png',dpi=180)
