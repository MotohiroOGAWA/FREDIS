"""Full-atom-feature queries, one-hop memory and pattern-owned structural cross attention."""

import pytest
import torch
from torch_geometric.data import Batch

from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.ml.mol.mol_encoder import MolEncoder
from fredis.ml.common.layers.graphormer import GraphormerEncoder
from fredis.ml.specgen.components.reaction.reactant_atom_attention import (
    ReactionAtomEncoder, OneHopAtomGNN, build_molecular_batch,
)
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.input.source_reaction_structure import SourceReactionStructure
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.training.fragment_tree_training.model import ReactionFragmentTreeTrainingModel
from .TestReactionPipeline import config


def raw_batch(device='cpu'):
    mol = MolEncoder(symbols=['C', 'N', 'O'], node_dim=8, graph_dim=11, num_layers=1, num_heads=2)
    graphs = [mol.graph_builder.build(Compound.from_smiles(s), device=device) for s in ('CCCCCO', 'CCN')]
    return Batch.from_data_list(graphs), mol.atom_dim, mol.bond_dim


def encoder(atom_dim, bond_dim, **kwargs):
    return ReactionAtomEncoder('CC', atom_dim=atom_dim, bond_dim=bond_dim,
                               hidden_dim=12, output_dim=7, num_heads=3, **kwargs).eval()


def test_query_uses_all_atom_features_and_position_one_hot_and_both_sides_get_centrality():
    graph, atom_dim, bond_dim = raw_batch()
    model = encoder(atom_dim, bond_dim, num_layers=2)
    assert not isinstance(model, GraphormerEncoder)
    sites = torch.tensor([[0, 1]])
    molecular = build_molecular_batch(graph.x, graph.edge_index, graph.edge_attr, graph.ptr)
    inputs, tokens = [], []
    model.reactant_projection.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach()))
    model.layers[0].register_forward_pre_hook(lambda module, args: tokens.append((args[0].detach(), args[1].detach())))
    output = model(molecular, sites)
    assert output.shape == (1, 7)
    expected = torch.cat((graph.x[sites], torch.eye(2)[None]), dim=-1)
    torch.testing.assert_close(inputs[0], expected)
    degree = model.centrality(torch.zeros_like(tokens[0][1]), molecular.edge_index, molecular.edge_mask, molecular.node_mask)
    torch.testing.assert_close(tokens[0][0], model.reactant_projection(expected) + degree[0, :2][None])
    hidden = model.atom_gnn(graph.x, graph.edge_index, graph.edge_attr)
    torch.testing.assert_close(tokens[0][1], hidden[molecular.node_ids.clamp_min(0)] + degree)
    assert model.readout.in_features == 2 * 12


def test_gnn_has_exactly_one_hop_and_uses_edge_features():
    torch.manual_seed(11)
    gnn = OneHopAtomGNN(3, 2, 8)
    atoms = torch.randn(3, 3, requires_grad=True)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    bonds = torch.randn(4, 2, requires_grad=True)
    output = gnn(atoms, edges, bonds)
    far = atoms.detach().clone(); far[2] += torch.arange(3.)
    torch.testing.assert_close(output[0], gnn(far, edges, bonds)[0])
    near = atoms.detach().clone(); near[1] += torch.arange(3.)
    assert not torch.allclose(output[0], gnn(near, edges, bonds)[0])
    changed = bonds.detach().clone(); changed[1] += 5
    assert not torch.allclose(output[0], gnn(atoms, edges, changed)[0])
    output[0, 0].backward()
    assert atoms.grad[1].abs().sum() > 0 and not atoms.grad[2].any()
    assert bonds.grad[1].abs().sum() > 0


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_all_atoms_padding_molecule_isolation_and_all_layer_gradients(device, monkeypatch):
    torch.manual_seed(41)
    graph, atom_dim, bond_dim = raw_batch(device)
    graph.x.requires_grad_(); graph.edge_attr.requires_grad_()
    model = encoder(atom_dim, bond_dim, num_layers=3, max_spatial_dist=3, max_edge_dist=3).to(device)
    sites = torch.tensor([[0, 1], [6, 7]], device=device)
    molecular = build_molecular_batch(graph.x, graph.edge_index, graph.edge_attr, graph.ptr)
    output = model(molecular, sites)
    other = graph.x.detach().clone(); other[6:] += torch.arange(atom_dim, device=device)
    changed = build_molecular_batch(other, graph.edge_index, graph.edge_attr, graph.ptr)
    torch.testing.assert_close(output[:1], model(changed, sites)[:1])
    far = graph.x.detach().clone(); far[5] += torch.arange(atom_dim, device=device)
    changed = build_molecular_batch(far, graph.edge_index, graph.edge_attr, graph.ptr)
    assert not torch.allclose(output[:1], model(changed, sites)[:1])
    # Bias limits must never mask real distant atoms; padding must always be masked.
    from fredis.ml.specgen.components.reaction import reactant_atom_attention as module
    original = module.F.scaled_dot_product_attention
    masks = []
    def capture(*args, **kwargs):
        masks.append(kwargs['attn_mask'].detach())
        return original(*args, **kwargs)
    monkeypatch.setattr(module.F, 'scaled_dot_product_attention', capture)
    model(molecular, sites)
    assert masks[0].shape == (2, 3, 2, 6)
    assert torch.isfinite(masks[0][0]).all()
    assert torch.isneginf(masks[0][1, ..., 3:]).all()
    for weights in (model.spatial.dist_emb.weight, model.edge_encoding.w_pos):
        before = weights.detach().clone()
        with torch.no_grad(): weights[1].add_(torch.arange(weights[1].numel(), device=device).reshape_as(weights[1]))
        assert not torch.allclose(output, model(molecular, sites))
        with torch.no_grad(): weights.copy_(before)
    output = model(molecular, sites)
    (output * torch.randn_like(output)).sum().backward()
    for parameter in (model.reactant_projection.weight, model.atom_gnn.message_projection.weight,
                      model.centrality.in_deg_emb.weight, model.spatial.dist_emb.weight,
                      model.edge_encoding.w_pos, model.readout.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    for layer in model.layers:
        assert layer.q_proj.weight.grad.abs().sum() > 0
        assert layer.k_proj.weight.grad.abs().sum() > 0
        assert layer.v_proj.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='same source molecule'):
        model(molecular, torch.tensor([[0, 6]], device=device))


def test_training_static_chunk_parity_independence_reload_and_prediction():
    torch.manual_seed(43)
    generator = create_spectrum_generator(config()).eval()
    builder = ReactionStructureBuilder(generator)
    items = []
    for smiles in ('CC(O)N', 'CCCCO'):
        item, _ = builder.build(Compound.from_smiles(smiles), [Adduct.parse('[M+H]+')], [20.],
                                [[18.033826, 44.049476]], [[.5, 1.]])
        items.append(item)
    data = SourceReactionStructure.from_structures(items)
    model = generator.feature_model
    expected = model(data)
    static = model.encode_static(data, max_graphs=1)
    actual = model(data, static_features=static)
    for field in ('reaction_h', 'local_reaction_h', 'memory', 'state_h', 'source_embeddings'):
        torch.testing.assert_close(getattr(expected, field), getattr(actual, field))
    training = ReactionFragmentTreeTrainingModel(model, spectrum_model=generator.spectrum_model)
    output = training(data)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(e.reactant_projection.weight.grad.abs().sum() > 0 for e in model.reaction_instance_encoder.reactant_encoders.values())
    restored = create_spectrum_generator(config()).eval()
    restored.load_state_dict(generator.state_dict(), strict=True)
    torch.testing.assert_close(expected.local_reaction_h, restored.feature_model(data).local_reaction_h)
    with torch.no_grad():
        for parameter in restored.mol_encoder.parameters(): parameter.add_(1)
    torch.testing.assert_close(expected.local_reaction_h, restored.feature_model(data).local_reaction_h)
    prediction = generator.predict([Compound.from_smiles('CCO')], [Adduct.parse('[M+H]+')], [20.])
    assert prediction.spectrum_batch.canonical_data.num_samples == 1


def test_export_import_freezes_pattern_and_keeps_molecular_encoder_independent(tmp_path):
    from fredis.ml.training.fragment_tree_training.pattern_models import export_pattern, assemble_config, initialize_imports, inspect_model
    params = config()
    generator = create_spectrum_generator(params)
    training = ReactionFragmentTreeTrainingModel(generator.feature_model, spectrum_model=generator.spectrum_model)
    checkpoint = tmp_path / 'base.pt'
    torch.save(dict(architecture=params['architecture'], model_config=params, model_state_dict=training.state_dict()), checkpoint)
    pid = inspect_model(checkpoint)['patterns'][0]['pattern']['pattern_id']
    asset = tmp_path / 'one.reaction-model.pt'
    export_pattern(checkpoint, pid, asset)
    assembled = assemble_config(params, [dict(path=str(asset))])
    restored = create_spectrum_generator(assembled)
    target = ReactionFragmentTreeTrainingModel(restored.feature_model, spectrum_model=restored.spectrum_model)
    initialize_imports(target, assembled)
    restored.train()
    inherited = restored.feature_model.reaction_instance_encoder.reactant_encoders[str(pid)]
    assert not inherited.training and all(not p.requires_grad for p in inherited.parameters())
    inherited_position = restored.feature_model.reaction_instance_encoder.endpoint_projections[str(pid)]
    assert not inherited_position.training and all(not p.requires_grad for p in inherited_position.parameters())
    assert all(p.requires_grad for p in restored.mol_encoder.parameters())
    assert restored.mol_encoder.training
    for name, tensor in torch.load(asset, weights_only=False)['model_state_dict'].items():
        torch.testing.assert_close(tensor, target.state_dict()[name])


@pytest.mark.parametrize('key,value', [('hidden_dim', 0), ('num_layers', 0), ('num_heads', 5),
                                      ('ffn_mult', 0), ('max_edge_dist', 0)])
def test_invalid_architecture_is_rejected(key, value):
    graph, atom_dim, bond_dim = raw_batch()
    kwargs = dict(atom_dim=atom_dim, bond_dim=bond_dim, hidden_dim=12, output_dim=7, num_heads=3)
    kwargs[key] = value
    with pytest.raises(ValueError): ReactionAtomEncoder('CC', **kwargs)


def test_every_atom_and_bond_channel_is_forwarded_without_slicing():
    graph, atom_dim, bond_dim = raw_batch()
    model = encoder(atom_dim, bond_dim, num_layers=1)
    captures = {}
    def record(name):
        def hook(_module, args):
            captures[name] = args[0].detach().clone()
        return hook
    model.reactant_projection.register_forward_pre_hook(record('query'))
    model.atom_gnn.self_projection.register_forward_pre_hook(record('self'))
    model.atom_gnn.message_projection.register_forward_pre_hook(record('messages'))
    molecular = build_molecular_batch(graph.x, graph.edge_index, graph.edge_attr, graph.ptr)
    model(molecular, torch.tensor([[0, 1]]))
    torch.testing.assert_close(captures['query'][0, :, :atom_dim], graph.x[:2])
    torch.testing.assert_close(captures['self'], graph.x)
    torch.testing.assert_close(captures['messages'][:, :atom_dim], graph.x[graph.edge_index[0]])
    torch.testing.assert_close(captures['messages'][:, atom_dim:], graph.edge_attr)
    assert bond_dim == 9 and model.edge_encoding.w_pos.shape[1] == bond_dim


@pytest.mark.parametrize('matches', [1, 7, 257])
def test_attention_processes_actual_batch_reaction_count_without_fixed_chunks(matches, monkeypatch):
    graph, atom_dim, bond_dim = raw_batch()
    model = encoder(atom_dim, bond_dim, num_layers=2)
    sites = torch.tensor([[0, 1]]).expand(matches, -1)
    molecular = build_molecular_batch(graph.x, graph.edge_index, graph.edge_attr, graph.ptr)
    from fredis.ml.specgen.components.reaction import reactant_atom_attention as module
    original = module.F.scaled_dot_product_attention
    sizes = []
    def record(query, *args, **kwargs):
        sizes.append(query.shape[0])
        return original(query, *args, **kwargs)
    monkeypatch.setattr(module.F, 'scaled_dot_product_attention', record)
    output = model(molecular, sites)
    assert output.shape == (matches, 7)
    assert sizes == [matches, matches]
    torch.testing.assert_close(output, output[:1].expand_as(output))
