import torch
from torch_geometric.data import Batch

from fredis.libs.mmkit.mmkit import Compound
from fredis.ml.mol.mol_encoder import MolEncoder


def test_repeated_encoding_preserves_raw_features_and_gradients():
    model = MolEncoder(
        symbols=('Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S'),
        node_dim=16, graph_dim=64, num_layers=1, num_heads=8, dropout=0,
    ).eval()
    batch = Batch.from_data_list([
        model.encode_components(Compound.from_smiles('CCO')),
        model.encode_components(Compound.from_smiles('c1ccccc1')),
    ])
    raw_features = batch.x.clone()
    assert raw_features.shape == (9, 35)
    outputs = []
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        encoded = model(batch)
        assert encoded is not batch
        assert encoded.shape == (2, 64)
        torch.testing.assert_close(batch.x, raw_features)
        assert 'embeddings' not in batch
        encoded.square().sum().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())
        outputs.append(encoded.detach())
    torch.testing.assert_close(*outputs)
