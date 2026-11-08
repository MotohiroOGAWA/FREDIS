import json
from pathlib import Path
import numpy as np
import pytest
import msentity as ms
import msentity_similarity as sim
from fredis.analysis.theoretical_coverage import calculate, enrich_file
from fredis.domain.fragment.fragmenter import Fragmenter
from fredis.libs.mmkit.mmkit import Compound, Adduct


@pytest.fixture
def chemistry():
    params=json.loads((Path(__file__).parents[1]/'fixtures/fragmenter.pft').read_text())
    params['fragment_ion_tree_builder']['max_reaction_count']=1
    fragmenter=Fragmenter.from_dict(params)
    tree=fragmenter.build_fragment_ion_tree(Compound.from_smiles('CCO'))
    masses=[float(f.exact_mass) for f in tree.get_formula_candidate_group(Adduct.parse('[M+H]+')).formulas]
    return params,np.array([[masses[0],4.],[masses[-1],3.],[999.123,2.]])


def annotate(dataset,ids,smiles=None):
    dataset.set_column('SpecID',ids)
    dataset.set_column('SMILES',smiles or ['CCO']*len(dataset))
    dataset.set_column('AdductType',['[M+H]+']*len(dataset))


def test_parallel_coverage_and_alpha(chemistry):
    params,peaks=chemistry
    with ms.from_peaks([peaks,peaks]) as observed:
        annotate(observed,['0001','0002'])
        values,errors=calculate(observed,params,alpha=.5,workers=2)
        assert values==pytest.approx([7/9,7/9])
        assert errors==['','']
        values,_=calculate(observed,params,alpha=1,workers=1)
        assert values==pytest.approx([25/29,25/29])


def test_enrich_preserves_rows_spectra_and_metadata(tmp_path,chemistry):
    params,peaks=chemistry
    path=tmp_path/'predicted.mssim'
    with ms.from_peaks([peaks,peaks]) as observed,ms.from_peaks([peaks[:1]]) as predicted:
        annotate(observed,['other','0001']);annotate(predicted,['0001'])
        with sim.similarity_by_key(predicted,observed,intensity_exponent=.5,include_matched_data=True) as result:
            frame=enrich_file(path,result,observed,params,alpha=.5,workers=1)
    assert frame.index2.tolist()==[1]
    assert frame.theoretical_fragment_peak_intensity_coverage.tolist()==pytest.approx([7/9])
    with sim.load(path) as result:
        assert result.to_pandas().theoretical_fragment_peak_intensity_coverage.tolist()==pytest.approx([7/9])
        assert result.to_pandas().intensity_coverage2.tolist()==pytest.approx([4/9])
        assert result.metadata['parameters']['intensity_exponent']==.5
        assert result.metadata['fredis_theoretical_coverage']['fragmenter_params']==params
        a,b=result.match_records(0)
        assert b['SpecID']=='0001'
        assert b.intensity.tolist()==[4,3,2]


def test_failure_is_missing_not_zero_and_lightweight_stays_lightweight(tmp_path,chemistry):
    params,peaks=chemistry
    with ms.from_peaks([peaks,peaks]) as obs,ms.from_peaks([peaks[:1],peaks[:1]]) as pred:
        annotate(obs,['one','two'],['CCO','invalid smiles']);annotate(pred,['one','two'])
        with sim.similarity_by_key(pred,obs,include_matched_data=False) as result:
            frame=enrich_file(tmp_path/'result.mssim',result,obs,params,alpha=1,workers=1)
    assert frame.theoretical_fragment_peak_intensity_coverage.iloc[0]==pytest.approx(25/29)
    assert np.isnan(frame.theoretical_fragment_peak_intensity_coverage.iloc[1])
    assert frame.theoretical_fragment_error.iloc[1]
    with sim.load(tmp_path/'result.mssim') as result:
        assert result.metadata['fredis_theoretical_coverage']['failure_count']==1
        assert not result.has_matched_data
