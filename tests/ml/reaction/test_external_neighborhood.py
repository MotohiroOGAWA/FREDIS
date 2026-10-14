import pytest
import torch
from fredis.ml.reaction.external_neighborhood import ExternalNeighborhoodEncoder
from fredis.ml.reaction.reactant_graphormer import ReactantGraphormerEncoder
from fredis.ml.specgen.components.reaction.reaction_instance_encoder import ReactionInstanceEncoder


def edges(pairs):
    return torch.tensor(pairs+[(b,a) for a,b in pairs],dtype=torch.long).T.contiguous()


def test_three_hop_propagation_boundary_and_zero_unconnected_role():
    torch.manual_seed(3)
    model=ExternalNeighborhoodEncoder(4,7,3)
    x=torch.randn(6,4,requires_grad=True)
    graph=edges([(0,1),(0,2),(2,3),(3,4),(4,5)])
    sites=torch.tensor([[0,1]])
    result=model(x,graph,sites)
    assert result.shape==(1,2,7)
    assert torch.count_nonzero(result[:,1])==0
    near=x.detach().clone();near[4]+=2
    assert not torch.equal(result[:,0],model(near,graph,sites)[:,0])
    far=x.detach().clone();far[5]+=100
    torch.testing.assert_close(result,model(far,graph,sites),rtol=0,atol=0)
    root=x.detach().clone();root[:2]+=100
    torch.testing.assert_close(result,model(root,graph,sites),rtol=0,atol=0)
    result.square().sum().backward()
    assert x.grad[4].abs().sum()>0
    assert torch.count_nonzero(x.grad[5])==0
    assert all(layer.weight.grad.abs().sum()>0 for layer in model.messages)


def test_no_hops_and_disconnected_matches_are_exact_zero():
    x=torch.randn(4,3);graph=edges([(0,1),(2,3)])
    model=ExternalNeighborhoodEncoder(3,5,3)
    result=model(x,graph,torch.tensor([[0,1],[2,3]]))
    assert torch.count_nonzero(result)==0
    disabled=ExternalNeighborhoodEncoder(3,5,0)
    assert torch.count_nonzero(disabled(x,graph,torch.tensor([[0]])))==0


def test_neighborhood_is_match_specific_and_batch_order_independent():
    model=ExternalNeighborhoodEncoder(3,5,2)
    x=torch.randn(4,3);graph=edges([(0,1),(0,2),(2,3)])
    sites=torch.tensor([[0,1],[0,2]])
    result=model(x,graph,sites)
    assert torch.count_nonzero(result[0,1])==0
    assert torch.count_nonzero(result[1,1])>0
    for i in range(2):
        torch.testing.assert_close(result[i:i+1],model(x,graph,sites[i:i+1]))


def test_one_hot_positions_are_fixed_input_columns():
    model=ReactantGraphormerEncoder('CCC',atom_dim=4,bond_dim=0,hidden_dim=8,graph_dim=11,num_layers=1,num_heads=2)
    captured=[]
    handle=model.node_proj.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    atoms=torch.randn(2,3,4)
    model.encode_matches(atoms);handle.remove()
    torch.testing.assert_close(captured[0][...,:4],atoms)
    torch.testing.assert_close(captured[0][...,4:],torch.eye(3).expand(2,-1,-1))
    assert model.graph_token_count==2
    assert not hasattr(model,'positions')


def test_pattern_input_is_raw_chemistry_plus_neighborhood_plus_position():
    model=ReactionInstanceEncoder(4,6,8,(2,1,1),num_heads=2,pattern_smarts={0:'CC',1:'CC'},
        raw_atom_dim=3,pattern_input_version=2,pattern_model_params={
            '0':dict(hidden_dim=8,graph_dim=11,num_heads=2,neighborhood_hops=3,neighborhood_hidden_dim=5),
            '1':dict(hidden_dim=12,graph_dim=19,num_heads=3,neighborhood_hops=0,neighborhood_hidden_dim=7)})
    chemical=torch.randn(6,3,requires_grad=True)
    args=dict(source_atom_h=torch.randn(6,4),source_mol_h=torch.randn(1,6),
        source_atom_features=chemical,source_edge_index=edges([(0,1),(0,2),(2,3),(3,4),(4,5)]),
        reaction_type=torch.tensor([[0,0,0],[1,0,0]]),reaction_tree_index=torch.tensor([0,0]),
        reaction_source_atom_ptr=torch.tensor([0,2,4]),reaction_source_atom_index=torch.tensor([0,1,0,1]),
        reaction_static_features=torch.zeros(2,6),reaction_source_atom_hops=[])
    captured=[];handle=model.reactant_encoders['0'].node_proj.register_forward_pre_hook(lambda module,values:captured.append(values[0]))
    output=model(**args);handle.remove()
    assert output.shape==(2,8)
    assert captured[0].shape==(1,2,3+5+2)
    torch.testing.assert_close(captured[0][...,:3],chemical[:2].unsqueeze(0))
    assert torch.count_nonzero(captured[0][0,1,3:8])==0
    assert '1' not in model.neighborhood_encoders
    # The global molecular atom encoder must not enter the SMARTS input.
    changed={**args,'source_atom_h':torch.randn(6,4)*100}
    torch.testing.assert_close(output,model(**changed),rtol=0,atol=0)
    output[:,0].sum().backward()
    assert chemical.grad[4].abs().sum()>0
    assert torch.count_nonzero(chemical.grad[5])==0


@pytest.mark.parametrize('hops,width',[(-1,4),(1,0),(1.5,4)])
def test_invalid_neighborhood_architecture(hops,width):
    with pytest.raises(ValueError):ExternalNeighborhoodEncoder(3,width,hops)
