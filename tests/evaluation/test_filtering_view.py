import json
from pathlib import Path
import pandas as pd
import pytest
from evaluation.filtering import evaluate, present
from evaluation.filtering_view import select_view


@pytest.fixture
def result(tmp_path):
    pd.DataFrame({'SpecID':['1','2','3','4','5','6'],
                  'Class':['C','C','A','A','B','D'],
                  'Kingdom':['Z','Z','A','A','A','Z'],
                  'Superclass':['X','X','Y','Y','Y','A']}).to_json(tmp_path/'meta.json', orient='records')
    (tmp_path/'a.txt').write_text('1\n3\n4\n5\n')
    (tmp_path/'b.txt').write_text('2\n3\n6\n')
    return evaluate({'metadata':str(tmp_path/'meta.json'), 'groupColumn':'Class',
                     'classificationColumns':['Kingdom','Superclass'],
                     'tools':[{'name':'A','path':str(tmp_path/'a.txt'),'color':'red'},
                              {'name':'B','path':str(tmp_path/'b.txt'),'color':'blue'}]})


def test_priority_and_all_tool_statistics(result):
    view={'showOverall':False, 'sortRules':[
        {'field':'metadata:Kingdom','direction':'asc'},
        {'field':'metadata:Superclass','direction':'asc'},
        {'field':'totalN','direction':'desc'}]}
    rows,categories,_,fields=select_view(result,view)
    assert [g['category'] for g in categories]==['A','B','D','C']
    a=categories[0]
    assert a['metadata:Kingdom']==['A']
    assert (a['stats:min'],a['stats:max'],a['stats:mean'],a['stats:median'])==(50,100,75,75)
    assert {'metadata:Kingdom','tool:A:retainedN','stats:median'} <= {f['id'] for f in fields}
    # Hiding a tool must not change Common, summary statistics, or denominators.
    view['tools']=['Common']
    _,only_common,_,_=select_view(result,view)
    assert only_common[0]['stats:mean']==75
    assert {r['retainedPercent'] for r in rows if r['category']=='C' and r['tool']=='Common'}=={0}


def test_numeric_filters_and_sorting(result):
    view={'showOverall':False,'filters':[{'field':'tool:A:retainedPercent','op':'>=','value':50},
                                       {'field':'stats:mean','op':'>=','value':50}],
          'sortRules':[{'field':'stats:median','direction':'desc'}, {'field':'totalN','direction':'desc'}]}
    _,categories,_,_=select_view(result,view)
    assert [g['category'] for g in categories]==['A','C','B']
    view['filters'].append({'field':'stats:min','op':'>','value':0})
    assert [g['category'] for g in select_view(result,view)[1]]==['A','C']
    for metric in ['min','max','mean','median']:
        assert all(g['stats:'+metric]>=50 for g in select_view(result,{'filters':[{'field':'stats:'+metric,'op':'>=','value':50}]})[1])
    assert not select_view(result,{'filters':[{'field':'totalN','op':'>','value':100}]})[0]
    with pytest.raises(ValueError,match='finite'):
        select_view(result,{'filters':[{'field':'stats:mean','op':'>','value':'NaN'}]})


def test_common_color_layout_and_export_share_view(result,tmp_path):
    view={'showOverall':False,'categories':['A'], 'orientation':'vertical','colors':{'Common':'#123456'},
          'sortRules':[{'field':'tool:A:retainedPercent','direction':'asc'}]}
    shown=present(result,view,tmp_path/'out',['tsv','svg','png'])
    assert len(shown['categoryRows'])==1
    assert shown['categoryRows'][0]['category']=='A'
    svg=(tmp_path/'out'/'retention.svg').read_text()
    assert '#123456' in svg
    assert len(pd.read_csv(tmp_path/'out'/'categories.tsv',sep='\t'))==1
    assert len(pd.read_csv(tmp_path/'out'/'retention.tsv',sep='\t'))==3
    horizontal=present(result,{**view,'orientation':'horizontal'})
    assert shown['png']!=horizontal['png']


def test_list_filters_and_graph_checks_are_independent(result,tmp_path):
    view={'showOverall':False,'categories':['A','C'],
          'filters':[{'field':'totalN','op':'==','value':1}]}
    shown=present(result,view,tmp_path/'checked',['tsv'])
    assert {g['category'] for g in shown['listRows']}=={'B','D'}
    assert {g['category'] for g in shown['categoryRows']}=={'A','C'}
    assert set(pd.read_csv(tmp_path/'checked'/'retention.tsv',sep='\t').category)=={'A','C'}
    # Bulk-unselect visible B/D leaves previously selected A/C intact.
    selected=set(view['categories'])-{g['category'] for g in shown['listRows']}
    assert selected=={'A','C'}
    selected.update(g['category'] for g in shown['listRows'])
    all_checked=present(result,{**view,'categories':sorted(selected)})
    assert len(all_checked['categoryRows'])==4
    assert {g['category'] for g in all_checked['listRows']}=={'B','D'}


def test_extra_column_can_be_loaded_without_retention_recalculation(result,tmp_path):
    from evaluation.filtering import enrich_metadata
    frame=pd.read_json(tmp_path/'meta.json')
    frame['Origin']=['Z','Z','A','A','C','B']
    frame.to_json(tmp_path/'meta.json',orient='records')
    # Tool lists are not needed to load a different metadata column.
    (tmp_path/'a.txt').unlink()
    (tmp_path/'b.txt').unlink()
    view={'showOverall':False,'sortRules':[{'field':'metadata:Origin','direction':'asc'}]}
    enriched=enrich_metadata(result,view)
    assert enriched['rows']==result['rows']
    assert 'Origin' in enriched['metadataColumns']
    assert [g['category'] for g in select_view(enriched,view)[1]]==['A','D','B','C']
    assert enrich_metadata(enriched,view) is enriched
    # Swapping IDs between categories preserves counts but must still be rejected.
    frame.loc[[0,2],'Class']=frame.loc[[2,0],'Class'].to_numpy()
    frame.to_json(tmp_path/'meta.json',orient='records')
    with pytest.raises(ValueError,match='changed'):
        enrich_metadata(result,view)


def test_category_selection_keeps_image_size_and_cached_counts(result):
    import base64
    import struct
    many={**result, 'groups':[], 'metadataColumns':[], 'availableMetadataColumns':[],
          'rows':[{**row,'category':f'Category {i:02d}'} for i in range(30)
                  for row in result['rows'] if row['scope']=='category' and row['category']=='A']}
    full=present(many,{'showOverall':False})
    small=present(many,{'showOverall':False,'categories':['Category 00']})
    full_width,full_height=struct.unpack('>II',base64.b64decode(full['png'])[16:24])
    small_width,small_height=struct.unpack('>II',base64.b64decode(small['png'])[16:24])
    assert (full_width, full_height) == (small_width, small_height) == (1200, 700)
    assert len(full['categoryRows']) == 30
    assert len(small['categoryRows']) == 1
    assert small['displayRows'] == [row for row in full['displayRows'] if row['category'] == 'Category 00']


@pytest.mark.parametrize('orientation', ['vertical', 'horizontal'])
def test_group_pixel_width_and_text_visibility(result, tmp_path, orientation):
    import base64
    import struct
    view = dict(groupWidth=100, paddingWidth=20, height=500, orientation=orientation,
                title='Custom chart title', titleSize=23, categoryLabelSize=12,
                tickLabelSize=13, axisTitleSize=14, valueLabelSize=9, legendSize=15)
    shown = present(result, view, tmp_path/'shown', ['svg'])
    count = len(shown['categoryRows'])
    image_width, image_height = struct.unpack('>II', base64.b64decode(shown['png'])[16:24])
    assert image_width > 100*count+20*(count+1)
    assert image_height > 500
    # SVG axes rectangle preserves the exact requested plot dimensions.
    import xml.etree.ElementTree as ET
    svg_root = ET.parse(tmp_path/'shown'/'retention.svg').getroot()
    clip_rect = svg_root.find('.//{http://www.w3.org/2000/svg}clipPath/{http://www.w3.org/2000/svg}rect')
    assert float(clip_rect.attrib['width']) == pytest.approx((100*count+20*(count+1))*72/150)
    assert float(clip_rect.attrib['height']) == pytest.approx((500+(80 if orientation=='vertical' else 0))*72/150)
    assert 'Custom chart title' in (tmp_path/'shown'/'retention.svg').read_text()
    hidden = {**view, **{key: False for key in ['showTitle','showCategoryLabels',
              'showTickLabels','showAxisTitle','showValues','showLegend','showCounts']}}
    present(result, hidden, tmp_path/'hidden', ['svg'])
    svg = (tmp_path/'hidden'/'retention.svg').read_text()
    for text in ['Custom chart title', 'Retained spectra', 'Common', '(N=', '(100.0%)']:
        assert text not in svg
    with pytest.raises(ValueError, match='Calculated image width'):
        present(result, {**view, 'groupWidth':12000, 'paddingWidth':12000})
