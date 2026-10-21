import json
from pathlib import Path
import pandas as pd
import pytest
from evaluation.filtering import evaluate, present


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
    assert not (output/'retention.png').exists()
    assert not (output/'retention.tsv').exists()
    saved=json.loads((output/'settings.evaltrainable.pft').read_text())
    assert saved['config']['resultPath']=='results.json'
    cached=json.loads((output/saved['config']['resultPath']).read_text())
    assert cached['rows']==result['rows']
    # A moved evaluation directory must render/export even without source files.
    moved=tmp_path/'moved'
    output.rename(moved)
    for filename in ['metadata.json','a.txt','b.txt']:
        (tmp_path/filename).unlink()
    cached=json.loads((moved/saved['config']['resultPath']).read_text())
    view={'categories':['A'],'tools':['Common','B'],'showOverall':False,'sort':'count'}
    display=present(cached,view,tmp_path/'exports',formats=['tsv'])
    assert len(display['displayRows'])==2
    assert {r['category'] for r in display['displayRows']}=={'A'}
    assert {r['retainedPercent'] for r in display['displayRows']}=={50}
    assert len(pd.read_csv(tmp_path/'exports'/'retention.tsv', sep='\t'))==2
    assert not (tmp_path/'exports'/'retention.png').exists()
    present(cached,view,tmp_path/'png-only',formats=['png'])
    assert (tmp_path/'png-only'/'retention.png').read_bytes().startswith(b'\x89PNG')
    assert not (tmp_path/'png-only'/'retention.tsv').exists()
    present(cached,view,tmp_path/'svg-only',formats=['svg'])
    import xml.etree.ElementTree as ET
    svg = ET.parse(tmp_path/'svg-only'/'retention.svg').getroot()
    assert svg.tag == '{http://www.w3.org/2000/svg}svg'
    assert not (tmp_path/'svg-only'/'retention.png').exists()
    assert not (tmp_path/'svg-only'/'retention.tsv').exists()
    assert not present(cached,{'tools':[]})['displayRows']
    assert not present(cached,{'minN':5})['displayRows']
    pd.DataFrame({'SpecID':['001','001'],'Class':['A','B']}).to_json(tmp_path/'metadata.json',orient='records')
    with pytest.raises(ValueError,match='unique'):
        evaluate(config)
