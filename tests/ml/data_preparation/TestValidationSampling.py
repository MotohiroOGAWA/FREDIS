from fredis.ml.data_preparation.fragment_tree.validation_sampling import (
    sample_validation_smiles_by_max_tanimoto,
)


def test_samples_by_unique_training_smiles_count_and_is_deterministic():
    train = ["CC", "CC", "CCC", "c1ccccc1", "CCO", "CCN", "C=O", "C#N", "COC", "CCCl"]
    validation = ["CC", "CCCC", "c1ccccc1O", "N#N", "O", "CCF", "C1CC1"]

    first = sample_validation_smiles_by_max_tanimoto(train, validation, ratio=0.2, seed=4)
    second = sample_validation_smiles_by_max_tanimoto(train, validation, ratio=0.2, seed=4)

    assert len(first) == 2  # ceil(9 unique training SMILES * 0.2)
    assert first == second
    assert len({item.smiles for item in first}) == len(first)


def test_exact_training_structure_has_maximum_similarity_one():
    selected = sample_validation_smiles_by_max_tanimoto(
        ["CC"], ["CC"], ratio=1.0
    )
    assert selected[0].max_tanimoto_index == 1.0
    assert selected[0].bin_index == 9
