from __future__ import annotations

import pytest
import torch

from fredis.ml.specgen.predict_spectrum import direct_input_dataset
from fredis.ml.specgen.source_reaction_msdataset import build_prediction_cache, predict_source_msdataset
from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
from tests.ml.reaction.TestReactionPipeline import config


def _generator():
    value = config()
    return create_spectrum_generator(value).eval()


def test_predict_source_msdataset_skips_bad_records_and_reports_them():
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCCO', 'CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    # Row 1 (index 1) has the exact malformed adduct from the reported bug;
    # everything else in the dataset must still get predicted.
    dataset['AdductType'] = ['[M+H]+', '[Cat]+', '[M+H]+']

    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )

    assert len(predicted) == 2
    assert len(failures) == 1
    assert failures.iloc[0]['index'] == 1
    assert "'Cat'" in failures.iloc[0]['error']
    assert list(predicted.metadata['SMILES']) == ['CCCO', 'CCCO']


def test_predict_source_msdataset_survives_unexpected_exception_types():
    # Adduct.parse('[M+i]+') raises AttributeError (a real, separate bug in its
    # unfinished isotope-notation handling), not ValueError/TypeError. Any
    # exception type from a bad record must be caught, not just the ones this
    # function anticipated.
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    dataset['AdductType'] = ['[M+H]+', '[M+i]+']

    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )

    assert len(predicted) == 1
    assert len(failures) == 1
    assert failures.iloc[0]['index'] == 1
    assert failures.iloc[0]['error'].startswith('AttributeError:')


def test_predict_source_msdataset_empty_failures_when_everything_is_valid():
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )
    assert len(predicted) == 2
    assert failures.empty
    assert list(failures.columns) == ['index', 'smiles', 'error']


def test_predict_source_msdataset_adds_main_adduct_and_model_collision_energy_columns():
    # [M+H]+ is already one of the model's reference adduct types, so the
    # loss-stripped MainAdduct is unchanged; ModelCollisionEnergyEV is the
    # numeric eV value actually fed to the model, not the raw column text.
    dataset = direct_input_dataset(
        smiles_values=['CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )
    assert failures.empty
    assert predicted.metadata['MainAdduct'].tolist() == ['[M+H]+']
    assert predicted.metadata['ModelCollisionEnergyEV'].tolist() == [20.0]


def test_predict_source_msdataset_skips_adducts_unsupported_by_this_model():
    # [M+K]+ is syntactically valid but outside this model's trained adduct
    # vocabulary (unlike training's validation split, real input isn't
    # limited to it). Left unchecked this crashes deep inside
    # generator.prepare()'s conditions tensor, which is built once per
    # shared-compound group -- taking every other same-compound row down
    # with it. All three rows here share one SMILES to exercise exactly that
    # grouped-batching failure mode.
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCCO', 'CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    dataset['AdductType'] = ['[M+H]+', '[M+K]+', '[M+H]+']

    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )

    assert len(predicted) == 2
    assert len(failures) == 1
    assert failures.iloc[0]['index'] == 1
    assert 'supported adduct types' in failures.iloc[0]['error']


def test_predict_source_msdataset_skips_a_row_with_no_resolvable_precursor():
    # [M-2H2O+H]+ is in this model's adduct vocabulary and syntactically
    # valid, but propanol (CCCO) has no way to lose two waters -- resolving
    # it finds zero candidate reaction sequences. Left unchecked, that empty
    # tuple reaches prepare_source_reactions, which raises
    # UnresolvedPrecursorError deep inside generator.predict_batches(),
    # crashing every other sample sharing this compound (row 0 here).
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    dataset['AdductType'] = ['[M+H]+', '[M-2H2O+H]+']

    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )

    assert len(predicted) == 1
    assert len(failures) == 1
    assert failures.iloc[0]['index'] == 1
    assert 'No valid precursor reaction sequence' in failures.iloc[0]['error']


def test_predict_source_msdataset_reports_a_compound_rdkit_cannot_fragment():
    # Fragmentation now runs once per unique compound in a separate
    # precompute phase, not inline per row during parsing; a compound RDKit
    # rejects there must still fail only its own rows, exactly like a bad
    # adduct or an unparseable collision energy does.
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCCO', 'CCN'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    dataset['SMILES'] = ['CCCO', 'not-a-smiles', 'CCN']

    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )

    assert len(predicted) == 2
    assert len(failures) == 1
    assert failures.iloc[0]['index'] == 1
    assert 'not-a-smiles' in failures.iloc[0]['error']


def test_predict_source_msdataset_skips_atoms_unsupported_by_the_checkpoint():
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCN', 'c1ccccc1'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    # Sodium is valid chemistry/RDKit input, but it is absent from this
    # checkpoint's learned atom vocabulary.  It must be reported as one bad
    # record instead of reaching graph construction and aborting the batch.
    dataset['SMILES'] = ['CCCO', 'O=C(O)O[Na]', 'c1ccccc1']

    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )

    assert len(predicted) == 2
    assert predicted.metadata['SMILES'].tolist() == ['CCCO', 'c1ccccc1']
    assert failures['index'].tolist() == [1]
    assert "Unsupported atom symbol(s): Na" in failures.iloc[0]['error']


def test_predict_source_msdataset_isolates_unexpected_prediction_errors(monkeypatch):
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCN', 'c1ccccc1'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    generator = _generator()
    original_predict_batches = generator.predict_batches

    def predict_batches(sources, precursor_types, collision_energy, **kwargs):
        if any(source.smiles == 'CCN' for source in sources):
            raise RuntimeError('simulated record-specific inference failure')
        yield from original_predict_batches(
            sources, precursor_types, collision_energy, **kwargs
        )

    monkeypatch.setattr(generator, 'predict_batches', predict_batches)
    predicted, failures = predict_source_msdataset(
        dataset, generator,
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None,
    )

    assert len(predicted) == 2
    assert predicted.metadata['SMILES'].tolist() == ['CCCO', 'c1ccccc1']
    assert failures['index'].tolist() == [1]
    assert failures.iloc[0]['error'] == 'RuntimeError: simulated record-specific inference failure'


@pytest.mark.parametrize('num_workers,chunk_size', [(1, 1), (2, 1), (2, 4)])
def test_predict_source_msdataset_parallel_prepare_matches_serial(num_workers, chunk_size):
    # num_workers>1 fans the RDKit-heavy precompute out to real subprocesses
    # (fredis.ml.specgen.prepare_worker); results must match the in-process
    # (num_workers=1) path exactly, both for compounds that fragment fine and
    # one that fails outright.
    dataset = direct_input_dataset(
        smiles_values=['CCCO', 'CCN', 'c1ccccc1', 'CCCO'],
        collision_energy='20 eV',
        adduct_type='[M+H]+',
    )
    dataset['SMILES'] = ['CCCO', 'CCN', 'c1ccccc1', 'not-a-smiles']

    predicted, failures = predict_source_msdataset(
        dataset, _generator(),
        smiles_column='SMILES', adduct_type_column='AdductType',
        collision_energy_column='CollisionEnergy', precursor_mz_column='PrecursorMZ',
        instrument_column=None, num_workers=num_workers, chunk_size=chunk_size,
    )

    assert len(predicted) == 3
    assert list(predicted.metadata['SMILES']) == ['CCCO', 'CCN', 'c1ccccc1']
    assert len(failures) == 1
    assert failures.iloc[0]['index'] == 3


def test_build_prediction_cache_serial_and_parallel_agree_on_keys_and_reactions():
    generator = _generator()
    groups = {'CCCO': ['[M+H]+'], 'CCN': ['[M+H]+'], 'not-a-smiles': ['[M+H]+']}
    prep_config = {
        'fragmenter_params': generator.fragmenter.to_dict(),
        'symbols': list(generator.mol_encoder.symbols),
        'adduct_type_strs': list(generator.adduct_type_strs),
    }
    serial = build_prediction_cache(groups, prep_config, num_workers=1)
    parallel = build_prediction_cache(groups, prep_config, num_workers=2, chunk_size=1)

    assert set(serial) == set(parallel) == {'CCCO', 'CCN', 'not-a-smiles'}
    assert serial['not-a-smiles']['ok'] is False and parallel['not-a-smiles']['ok'] is False
    assert serial['CCCO']['ok'] is True and parallel['CCCO']['ok'] is True
    assert serial['CCCO']['reactions'] == parallel['CCCO']['reactions']
    from fredis.domain.fragment.reaction.ReactionRelations import ReactionRelations
    for smiles in ('CCCO', 'CCN'):
        assert serial[smiles]['relations'] == parallel[smiles]['relations']
        assert serial[smiles]['relations'] == ReactionRelations.from_reactions(serial[smiles]['reactions'])
    assert serial['CCCO']['precursor_sequences'] == parallel['CCCO']['precursor_sequences']


def test_build_prediction_cache_keep_temp_controls_whether_task_files_survive(tmp_path):
    generator = _generator()
    groups = {'CCCO': ['[M+H]+'], 'CCN': ['[M+H]+']}
    prep_config = {
        'fragmenter_params': generator.fragmenter.to_dict(),
        'symbols': list(generator.mol_encoder.symbols),
        'adduct_type_strs': list(generator.adduct_type_strs),
    }

    default_root = tmp_path / 'default'
    default_root.mkdir()
    build_prediction_cache(groups, prep_config, num_workers=2, chunk_size=1, temp_dir=str(default_root))
    assert list(default_root.iterdir()) == []

    kept_root = tmp_path / 'kept'
    kept_root.mkdir()
    build_prediction_cache(groups, prep_config, num_workers=2, chunk_size=1, temp_dir=str(kept_root), keep_temp=True)
    kept = list(kept_root.iterdir())
    assert len(kept) == 1
    assert any(p.suffix == '.pkl' for p in kept[0].iterdir())


def test_prediction_prepare_reuses_worker_relations(monkeypatch):
    from fredis.ml.specgen.prepare_worker import prepare_compound_cache
    from fredis.domain.fragment.reaction.ReactionRelations import ReactionRelations
    from fredis.libs.mmkit.mmkit import Compound, Adduct
    generator = _generator()
    source = Compound.from_smiles('CCCO')
    cached = prepare_compound_cache(source.smiles, ['[M+H]+'], generator.fragmenter)
    expected = generator.prepare([source], [Adduct.parse('[M+H]+')], [20.])[0]
    def unexpected(*args, **kwargs):
        raise AssertionError('Prediction must reuse relations computed by CPU preparation')
    monkeypatch.setattr(ReactionRelations, 'from_reactions', unexpected)
    actual = generator.prepare([source], [Adduct.parse('[M+H]+')], [20.],
                               precomputed={source.smiles: cached})[0]
    for name in ('reaction_conflict_index', 'reaction_invalidation_index', 'reaction_dominance_index'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
