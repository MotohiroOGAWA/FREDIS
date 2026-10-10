"""Direct reaction locations and products, tested against actual RDKit."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[1] / 'src/features/reaction-pattern-set'))
import reaction_preview as preview
from rdkit import Chem
from rdkit.Chem import rdChemReactions

def complete_preview(payload):
    result = preview.reaction_preview(payload)
    for pattern, definition in zip(result['patterns'], payload['patterns']):
        for match in pattern['matches']:
            products = preview.reaction_preview_products({'smiles':payload['smiles'], 'pattern':definition,
                                                          'atoms':match['atoms'], 'bonds':match['bonds']})
            match.update(products)
    return result

split = {'name': 'C–O cleavage', 'reactant_smarts': '[#6:4]-[#8:9]',
         'products': [{'name': 'Split', 'smarts': '[#6:4].[#8:9]'},
                      {'name': 'Carbon side', 'smarts': '[#6:4]'}]}
# Initial rendering must perform matching only: exactly one input drawing and no SMIRKS compilation.
original_drawing, original_compile = preview.drawing, preview.rdChemReactions.ReactionFromSmarts
calls = []
def input_drawing(*args, **kwargs):
    calls.append(1)
    return original_drawing(*args, **kwargs)
def forbidden_compile(*args, **kwargs):
    raise AssertionError('Initial matching must not compile reactions')
preview.drawing = input_drawing
preview.rdChemReactions.ReactionFromSmarts = forbidden_compile
try:
    matches_only = preview.reaction_preview({'smiles':'CCOCCO','patterns':[split]})
    assert calls == [1]
    assert len(matches_only['patterns'][0]['matches']) == 3
    assert not matches_only['patterns'][0]['errors']
    assert all(not m['loaded'] and not m['products'] for m in matches_only['patterns'][0]['matches'])
finally:
    preview.drawing, preview.rdChemReactions.ReactionFromSmarts = original_drawing, original_compile

x = complete_preview({'smiles': 'CCOCCO', 'patterns': [split]})
assert not x['truncated']
assert len(x['patterns'][0]['matches']) == 3
assert [m['atoms'] for m in x['patterns'][0]['matches']] == [[1,2],[2,3],[4,5]]
for match in x['patterns'][0]['matches']:
    assert len(match['products']) == 2 and not match['errors']
    split_product = next(p for p in match['products'] if p['name'] == 'Split')
    assert len(split_product['molecules']) == 2
    assert sorted(i for m in split_product['molecules'] for i in m['sourceAtoms']) == list(range(6))
    for m in split_product['molecules']:
        assert Chem.MolFromSmiles(m['smiles']) is not None
        assert m['formula'] and m['exactMass'] > 0 and '<svg' in m['drawing']['svg']
    carbon = next(p for p in match['products'] if p['name'] == 'Carbon side')
    assert len(carbon['molecules']) == 1
    oxygen_index = next(i for i in match['atoms'] if i in (2, 5))
    assert oxygen_index not in carbon['molecules'][0]['sourceAtoms']
    for product in match['products']:
        reaction = rdChemReactions.ReactionFromSmarts(product['smirks'])
        assert reaction is not None
        assert reaction.GetReactantTemplate(0).GetNumAtoms() == 6
        assert {atom.GetAtomMapNum() for atom in reaction.GetReactantTemplate(0).GetAtoms()} == set(range(1, 7))
assert len({next(p for p in match['products'] if p['name'] == 'Split')['smirks']
            for match in x['patterns'][0]['matches']}) == 3
assert '<svg' in x['drawing']['svg'] and len(x['drawing']['positions']) == 6
assert 'fill:#FFFFFF' not in x['drawing']['svg']
assert x['patterns'][0]['matches'][0]['bonds'] == [1]
split_reactions = preview.reaction_explore({
    'smiles':'CCO', 'patterns':[split], 'maxReactionCount':-1, 'mode':'stepwise'
})['reactions']
split_components = [reaction for reaction in split_reactions if reaction['name'] == 'Split']
assert split_components
assert {reaction['productMoleculeCount'] for reaction in split_components} == {2}
assert {reaction['productMoleculeId'] for reaction in split_components} == {0, 1}

symmetric = {'name':'Symmetric C–C', 'reactant_smarts':'[#6:1]-[#6:2]',
             'products':[{'name':'One side','smarts':'[#6:1]'}]}
x = complete_preview({'smiles':'CC(C)C', 'patterns':[symmetric]})
assert len(x['patterns'][0]['matches']) == 3
for match in x['patterns'][0]['matches']:
    assert match['assignments'] == 2
    assert {p['smiles'] for p in match['products']} == {'C','CCC'}

cyclize = {'name':'New bond', 'reactant_smarts':'[#6:1]-[#6:2]-[#6:3]',
           'products':[{'name':'Ring','smarts':'[#6:1]1-[#6:2]-[#6:3]-1'}]}
x = complete_preview({'smiles':'CCC', 'patterns':[cyclize]})
assert len(x['patterns'][0]['matches']) == 1
assert [p['smiles'] for p in x['patterns'][0]['matches'][0]['products']] == ['C1CC1']

unchanged = {'name':'Keep source types', 'reactant_smarts':'[!#1:1]~[!#1:2]',
             'products':[{'name':'Unchanged','smarts':'[!#1:1]~[!#1:2]'}]}
x = complete_preview({'smiles':'C=O', 'patterns':[unchanged]})
assert x['patterns'][0]['matches'][0]['products'][0]['smiles'] == 'C=O'
ring = {'name':'Ring bond', 'reactant_smarts':'[#6:1]-;@[#6:2]', 'products':[]}
x = complete_preview({'smiles':'CC1CCC1', 'patterns':[ring]})
assert len(x['patterns'][0]['matches']) == 4
source = Chem.MolFromSmiles('CC1CCC1')
assert all(source.GetBondWithIdx(m['bonds'][0]).IsInRing() for m in x['patterns'][0]['matches'])

bad = {'name':'Invalid', 'reactant_smarts':'[broken', 'products':[]}
x = complete_preview({'smiles':'CCO', 'patterns':[bad,split,symmetric]})
assert x['patterns'][0]['errors'] and len(x['patterns'][1]['matches']) == 1
assert len(x['patterns'][2]['matches']) == 1
assert not preview.reaction_preview({'smiles':'CC', 'patterns':[split]})['patterns'][0]['matches']

invalid_product = {'name':'Valence', 'reactant_smarts':'[#6:1]-[#8:2]',
                   'products':[{'name':'Invalid oxygen','smarts':'[#6:1]#[#8:2]'}]}
x = complete_preview({'smiles':'CO', 'patterns':[invalid_product]})
assert x['patterns'][0]['matches'][0]['errors']
assert not x['patterns'][0]['matches'][0]['products']

for smiles, patterns in [('not smiles',[split]), ('',[split]), ('CC',[])]:
    try:
        preview.reaction_preview({'smiles':smiles,'patterns':patterns})
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid request accepted')

preview.LIMIT = 2
assert preview.reaction_preview({'smiles':'CCCC','patterns':[symmetric]})['truncated']

# The Workbench reaction viewer uses the production Source-anchored reaction
# generator and the same unordered compatibility rules as FragmentTreeBuilder.
preview.LIMIT = 512
chain_cut = {'name':'Chain cut', 'reactant_smarts':'[#6:1]-[#6:2]',
             'products':[{'name':'Retain role 1','smarts':'[#6:1]'}]}
request = {'smiles':'CCCCC', 'patterns':[chain_cut], 'maxReactionCount':2, 'mode':'stepwise'}
exploration = preview.reaction_explore(request)
assert len(exploration['reactions']) == 8
assert exploration['availableReactionIds'] == list(range(8))
assert any(reaction['sourceAtomMaps'] == [2, 1] and reaction['atoms'] == [1, 0]
           for reaction in exploration['reactions']), 'query-role order must not be sorted'
assert all(reaction['reactionBondMaps'] and reaction['reactionBonds']
           for reaction in exploration['reactions']), 'reaction centers must be explicit'
assert len({tuple(map(tuple, reaction['reactionBondMaps']))
            for reaction in exploration['reactions']}) == 4
inspected = preview.reaction_explore({**request, 'inspectReactionId':0})
assert inspected['candidateProduct']['smirks']
assert rdChemReactions.ReactionFromSmarts(inspected['candidateProduct']['smirks']) is not None
selected_id = next(reaction_id for reaction_id in exploration['availableReactionIds']
                   if preview.reaction_explore({**request, 'selectedReactionIds':[reaction_id]})['availableReactionIds'])
selected = preview.reaction_explore({**request, 'selectedReactionIds':[selected_id]})
assert selected['selectedProduct'] and '<svg' in selected['selectedProduct']['drawing']['svg']
assert selected['selectedProduct']['smirks']
a = selected['reactions'][selected_id]
for candidate_id in selected['availableReactionIds']:
    b = selected['reactions'][candidate_id]
    assert set(b['sourceAtomMaps']) <= set(a['retainedAtomMaps'])
    assert set(a['sourceAtomMaps']) <= set(b['retainedAtomMaps'])
    assert not {tuple(edge) for edge in a['changedBondMaps']} & {tuple(edge) for edge in b['changedBondMaps']}
generated = preview.reaction_explore({**request, 'mode':'exhaustive'})
assert generated['results'] and all(1 <= item['reactionCount'] <= 2 for item in generated['results'])
assert all('<svg' in item['molecule']['drawing']['svg'] for item in generated['results'])
assert selected['selectedProduct']['smirks'] == next(
    item['smirks'] for item in generated['results'] if item['reactionIds'] == [selected_id]
)
assert inspected['candidateProduct']['smirks'] == next(
    item['smirks'] for item in generated['results'] if item['reactionIds'] == [0]
)
unlimited = preview.reaction_explore({**request, 'maxReactionCount':-1, 'mode':'exhaustive'})
assert unlimited['maxReactionCount'] == len(unlimited['reactions'])
assert {item['smirks'] for item in generated['results']} <= {
    item['smirks'] for item in unlimited['results']
}
for invalid_maximum in (0, -2):
    try:
        preview.reaction_explore({**request, 'maxReactionCount':invalid_maximum})
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid Max reactions accepted')

# C-S-P-O regression: after cutting S-P and retaining P-O, S-P is invalid for
# both reasons (changed-bond overlap and a Source match outside retained atoms).
both_sides = {'name':'Any single cut', 'reactant_smarts':'[!#1:1]-[!#1:2]',
              'products':[{'name':'Role 1','smarts':'[!#1:1]'},
                          {'name':'Role 2','smarts':'[!#1:2]'}]}
chain_request = {'smiles':'CSPO', 'patterns':[both_sides], 'maxReactionCount':3,
                 'mode':'stepwise'}
chain = preview.reaction_explore(chain_request)
sp = next(reaction for reaction in chain['reactions']
          if reaction['sourceAtomMaps'] == [3, 2] and reaction['retainedAtomMaps'] == [3, 4])
after_sp = preview.reaction_explore({**chain_request, 'selectedReactionIds':[sp['id']]})
remaining_sources = [after_sp['reactions'][index]['sourceAtomMaps']
                     for index in after_sp['availableReactionIds']]
assert remaining_sources == [[3, 4], [4, 3]]
assert all([2, 3] not in after_sp['reactions'][index]['changedBondMaps']
           for index in after_sp['availableReactionIds'])
print('Direct reaction preview: exact source locations, omitted maps, symmetric assignments, multiple rules, dot-separated products, new bonds, RDKit SVG, validation and limits passed.')
