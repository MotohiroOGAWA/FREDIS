import json
from pathlib import Path
import pandas as pd
import pytest
from evaluation.filtering import evaluate


def test_retention_and_export(tmp_path):
    pd.DataFrame({'SpecID':['001','002','003','004'], 'Class':['A','A',None,'B']}).to_json(tmp_path/'metadata.json', orient='records')
    (tmp_path/'a.txt').write_text('001\n002\n002\nunknown\n')
    (tmp_path/'b.txt').write_text('002\n003\n')
    config={'metadata':str(tmp_path/'metadata.json'), 'groupColumn':'Class', 'tools':[
        {'name':'A','path':str(tmp_path/'a.txt'),'color':'#ff0000'},
        {'name':'B','path':str(tmp_path/'b.txt'),'color':'#0000ff'}]}
    result=evaluate(config,tmp_path/'out')
    row=next(r for r in result['rows'] if r['scope']=='overall' and r['tool']=='Common')
    assert (row['totalN'],row['retainedN'],row['excludedN'],row['retainedPercent'])==(4,1,3,25)
    assert next(r for r in result['rows'] if r['category']=='A' and r['scope']=='category' and r['tool']=='A')['retainedPercent']==100
    assert result['diagnostics'][0]['unknownN']==1
    assert result['diagnostics'][0]['duplicateN']==1
    assert any(r['category']=='(Missing)' for r in result['rows'])
    output=tmp_path/'out'
    assert (output/'retention.png').read_bytes().startswith(b'\x89PNG')
    assert len(pd.read_csv(output/'retention.csv'))==12
    saved=json.loads((output/'settings.evalfilter.pft').read_text())
    assert saved['config']==config
    assert evaluate(saved['config'])['rows']==result['rows']
    pd.DataFrame({'SpecID':['001','001'],'Class':['A','B']}).to_json(tmp_path/'metadata.json',orient='records')
    with pytest.raises(ValueError,match='unique'):
        evaluate(config)
