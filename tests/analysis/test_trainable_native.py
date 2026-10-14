import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fredis.libs.msentity.msentity.core.MSDataset import MSDataset
from fredis.libs.msentity.msentity.core.PeakSeries import PeakSeries
from fredis.analysis.training_eligibility.native import extract
from tests.ml.reaction.TestReactionPipeline import config,chemistry


def test_native_backward_metadata_failure_and_resume(tmp_path):
    cfg=config();cfg['reaction_model_params']['pattern_input_version']=3;cfg['mol_encoder_params']['descriptors']=[]
    meta=pd.DataFrame(dict(SpecID=['first','second','bad-adduct','bad-peaks'],SMILES=['CCO']*4,AdductType=['[M+H]+','[M+H]+','invalid','[M+H]+'],CollisionEnergy=[10.,40.,20.,20.],PrecursorMZ=[47.0494]*4))
    peaks=np.array([[47.0494,1.],[31.0184,.5]]*4);peaks[-1,1]=np.nan
    data=MSDataset(meta,PeakSeries(peaks,np.arange(0,9,2)));file=tmp_path/'data.msds';data.save(str(file))
    out=tmp_path/'fredis'
    stats=extract(file,out,chemistry(),model_config=cfg,workers=1,batch_size=2,max_reaction_combinations=1000)
    assert stats['trainable']==2 and stats['excluded']==2
    assert stats['funnel']['native_checked']==2
    assert (out/'trainable_spec_ids.txt').read_text()=='first\nsecond\n'
    records=pd.read_csv(out/'records.tsv',sep='\t')
    assert records.loc[:1,'Stage'].tolist()==['forward_loss_backward']*2
    assert records.loc[:1,'GradientsFinite'].tolist()==[True,True]
    assert records.loc[3,'ReasonCategory']=='InvalidPeaks'
    resumed=extract(file,out,chemistry(),model_config=cfg,workers=1,batch_size=2,max_reaction_combinations=1000,resume=True)
    assert resumed==stats
    with pytest.raises(ValueError,match='Resume rejected'):
        extract(file,out,chemistry(),model_config=cfg,workers=1,batch_size=1,max_reaction_combinations=1000,resume=True)

    (out/'trainable_spec_ids.txt').write_text('tampered\n')
    with pytest.raises(ValueError,match='was modified'):
        extract(file,out,chemistry(),model_config=cfg,workers=1,batch_size=2,max_reaction_combinations=1000,resume=True)


def test_empty_and_duplicate_ids_fail_before_output(tmp_path):
    meta=pd.DataFrame(dict(SpecID=['same','same'],SMILES=['CCO']*2,AdductType=['[M+H]+']*2,CollisionEnergy=[20.]*2,PrecursorMZ=[47.]*2))
    data=MSDataset(meta,PeakSeries(np.array([[47.,1.]]*2),np.arange(3)));file=tmp_path/'input.msds';data.save(str(file))
    with pytest.raises(ValueError,match='Duplicate'):extract(file,tmp_path/'out',chemistry())
    assert not (tmp_path/'out').exists()
