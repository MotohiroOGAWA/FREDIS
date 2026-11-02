"""Probability starts on observed precursor structures and follows their descendants."""

import torch
import pytest
from fredis.domain.fragment.fragmenter import Fragmenter
from fredis.libs.mmkit.mmkit import Compound, Adduct
from fredis.ml.input.structure_builder import ReactionStructureBuilder
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from fredis.ml.specgen.spectrum_topology import canonical_spectrum_topology
from .TestReactionPipeline import config


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_loss_precursor_initializes_structural_history_and_preserves_gradients(
    tmp_path, device
):
    torch.manual_seed(3)
    cfg = config()
    rules = cfg["fragmenter_params"]["fragment_ion_tree_builder"][
        "fragment_ion_adduct_rule_set"
    ]
    rules["adduct_rules"].append(
        {
            "name": "loss",
            "adduct_type": "[M+H-H2O]+",
            "fragment_ion_adducts": ["[M+H]+"],
        }
    )
    generator = create_spectrum_generator(cfg).eval()
    source = Compound.from_smiles("CCCO")
    adduct = Adduct.parse("[M+H-H2O]+")
    mz = adduct.apply_to_formula(source.formula).exact_mass
    data, _ = ReactionStructureBuilder(generator).build(
        source, [adduct], [20.0], [[mz]], [[1.0]], save_all_edges=True
    )
    generator = generator.to(device)
    data = data.to(device)
    batch = data.spectrum_batch
    topology = canonical_spectrum_topology(batch)
    assert topology.initial[topology.roots].sum() == 0
    assert topology.initial.sum() == 1
    core = batch.canonical_data
    starts = topology.execution_rows[topology.initial > 0]
    assert starts.numel()
    for cid in core.execution_graph.compound_index[starts].tolist():
        assert core.compounds.compounds[cid].formula.elements.get("O", 0) == 0
    path = tmp_path / "prepared.pft"
    data.save(path)
    loaded = type(data).load(path)
    loaded.validate()
    features = generator.feature_model(data)
    output = generator.spectrum_model(
        batch,
        reaction_h=features.local_reaction_h,
        collision_energy_h=features.collision_energy_h,
        source_embeddings=features.source_embeddings,
    )
    assert torch.isfinite(output.loss)
    assert output.metrics["probability_mass_error"] < 1e-5
    source_cid = int(data.source_compounds.compound_index[0])
    assert output.intensity[batch.state_node_index == source_cid].sum() == 0
    oxygen = data.formula_elements.index("O")
    assert output.intensity[batch.state_formula_tensor[:, oxygen] > 0].sum() == 0
    output.loss.backward()
    assert all(
        torch.isfinite(p.grad).all()
        for p in generator.parameters()
        if p.grad is not None
    )
    # Inference uses the same structural initialization without observed peaks.
    with torch.no_grad():
        prepared, *_ = generator.prepare([source], [adduct], [20.0])
    topology = canonical_spectrum_topology(prepared.spectrum_batch)
    assert topology.initial[topology.roots].sum() == 0
    assert topology.initial.sum() == 1
