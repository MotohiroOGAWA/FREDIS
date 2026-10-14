import pytest
import torch
from fredis.ml.reaction.global_attention import GlobalStructureAttention


def test_global_attention_uses_bonds_remote_atoms_and_preserves_match_shape():
    torch.manual_seed(7)
    model = GlobalStructureAttention(3, 2, 2, 8, num_layers=3, num_heads=2)
    atoms = torch.randn(7, 3, requires_grad=True)
    edges = torch.tensor([[0,1,1,2,2,3,3,4,4,5],[1,0,2,1,3,2,4,3,5,4]])
    bonds = torch.randn(10, 2, requires_grad=True)
    batch = torch.zeros(7, dtype=torch.long)
    sites = torch.tensor([[0,1],[2,3]])
    result = model(atoms, edges, bonds, batch, sites, torch.zeros(2,dtype=torch.long))
    assert result.shape == (2,2,8)
    result.square().mean().backward()
    assert atoms.grad[6].abs().sum() > 0  # Beyond 3 hops, even disconnected memory is visible.
    assert bonds.grad.abs().sum() > 0
    assert not torch.equal(result[0,0],result[0,1])


def test_source_molecules_are_isolated_and_atom_order_is_invariant():
    torch.manual_seed(9)
    model = GlobalStructureAttention(3,2,2,8,num_heads=2).eval()
    atoms = torch.randn(6,3)
    edges = torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]])
    bonds = torch.randn(8,2)
    batch = torch.tensor([0,0,0,1,1,1]); sites = torch.tensor([[0,1],[3,4]]); ids = torch.tensor([0,1])
    baseline = model(atoms,edges,bonds,batch,sites,ids)
    modified = atoms.clone(); modified[3:] += 100
    torch.testing.assert_close(baseline[0], model(modified,edges,bonds,batch,sites,ids)[0], rtol=0,atol=0)
    permutation = torch.tensor([2,0,1,5,3,4]); inverse = permutation.argsort()
    actual = model(atoms[permutation],inverse[edges],bonds,batch[permutation],inverse[sites],ids)
    torch.testing.assert_close(baseline,actual)
    with pytest.raises(ValueError, match='belong'):
        model(atoms,edges,bonds,batch,sites,ids.flip(0))


@pytest.mark.parametrize('kwargs',[dict(hidden_dim=7,num_heads=2),dict(hidden_dim=8,num_layers=0),dict(hidden_dim=8,num_heads=0)])
def test_invalid_architecture(kwargs):
    with pytest.raises(ValueError): GlobalStructureAttention(3,2,2,**kwargs)


def test_remote_double_bond_location_changes_attention_features():
    torch.manual_seed(21)
    model = GlobalStructureAttention(3,1,2,8,num_heads=2).eval()
    atoms = torch.ones(24,3)
    forward = torch.stack((torch.arange(23),torch.arange(1,24)))
    edges = torch.cat((forward,forward.flip(0)),dim=1)
    first=torch.ones(46,1);second=first.clone()
    first[[8,31]]=2;second[[13,36]]=2
    batch=torch.zeros(24,dtype=torch.long);sites=torch.tensor([[0,1]]);source=torch.tensor([0])
    # Local environments at the reactant are identical within three hops.
    a=model(atoms,edges,first,batch,sites,source)
    b=model(atoms,edges,second,batch,sites,source)
    assert (a-b).abs().max()>1e-6
