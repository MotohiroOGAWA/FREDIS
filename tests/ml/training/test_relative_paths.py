"""Moved output trees preserve document-relative references."""
import json
import pytest
from pathlib import Path

from fredis.ml.common.paths import relative_paths, resolve_paths
from fredis.ml.training.fragment_tree_training.configuration import write_training_configuration


def test_move_training_and_dataset_together(tmp_path):
    root = tmp_path / 'original'
    dataset = root / 'dataset'
    model = dataset / 'pretrained_models' / 'pattern.pt'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'weights')
    output = root / 'run'
    config = {'fine_tuning': {'pattern_imports': [{'path': str(model)}]}}
    target = write_training_configuration(output, config, {
        'train_dir': str(dataset / 'train_structures'),
        'val_dir': str(dataset / 'validation_structures'),
        'resume': str(output / 'last.pt'),
    })
    saved = json.loads(target.read_text())
    assert saved['outputDir'] == '.'
    assert saved['datasetDir'] == '../dataset'
    assert saved['modelConfig']['fine_tuning']['pattern_imports'][0]['path'] == '../dataset/pretrained_models/pattern.pt'
    moved = tmp_path / 'moved'
    root.rename(moved)
    restored = resolve_paths(saved, moved / 'run')
    assert Path(restored['outputDir']) == moved / 'run'
    assert Path(restored['datasetDir']) == moved / 'dataset'
    assert Path(restored['modelConfig']['fine_tuning']['pattern_imports'][0]['path']).read_bytes() == b'weights'


def test_path_conversion_leaves_chemistry_and_existing_relative_paths(tmp_path):
    value = {'source': 'pretrained_models/model.pt', 'smiles': 'CC/O',
             'datasets': [{'path': str(tmp_path / 'input.msds')}],
             'outputDir': str(tmp_path), 'resume': ''}
    saved = relative_paths(value, tmp_path)
    assert saved == {**value, 'datasets': [{'path': 'input.msds'}], 'outputDir': '.'}
    assert value['outputDir'] == str(tmp_path)


@pytest.mark.parametrize("interval,epochs", [(0, 1), (2, 2)])
def test_moved_dataset_training_output_prediction_and_resume(tmp_path, interval, epochs):
    import csv
    import torch
    from ..reaction.test_canonical_workflow import prepared
    from ..reaction.TestReactionPipeline import config, chemistry
    from fredis.ml.input.source_reaction_structure import save_fragment_tree_structure
    from fredis.ml.specgen.predict_spectrum import load_generator
    from fredis.libs.mmkit.mmkit import Compound, Adduct
    from fredis.ml.training.fragment_tree_training.training import train_reactions
    from fredis.ml.training.fragment_tree_training.sources import prepared_dataset_paths

    _, data = prepared()
    original = tmp_path / 'original'
    dataset = original / 'dataset'
    dataset.mkdir(parents=True)
    (dataset / 'fragment-tree.training-data-config.pft').write_text(json.dumps({'chemistry_config': chemistry()}))
    for split in ('train_structures', 'validation_structures'):
        directory = dataset / split
        filename = 'sample.preft.pt'
        save_fragment_tree_structure(structure=data, output_file=directory / 'data' / filename)
        with (directory / 'manifest.tsv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['file', 'num_valid_samples', 'status'], delimiter='\t')
            writer.writeheader()
            writer.writerow(dict(file=filename, num_valid_samples=data.num_samples, status='completed'))
    moved = tmp_path / 'moved'
    original.rename(moved)
    train, validation = prepared_dataset_paths(moved / 'dataset')
    output = moved / 'run'
    train_reactions(model_config=config(), train_dir=train, val_dir=validation,
                    output_dir=output, epochs=epochs, device='cpu', warmup_steps=0,
                    checkpoint_interval_epochs=interval)
    checkpoint_name = f'checkpoints/epoch_{epochs:06d}.pt' if interval else 'last.pt'
    if interval:
        assert [p.name for p in (output / 'checkpoints').glob('*.pt')] == ['epoch_000002.pt']
        periodic = torch.load(output / checkpoint_name, weights_only=False)
        assert periodic['epoch'] == epochs
        assert periodic['training_settings']['train_dir'] == '../../dataset/train_structures'
        assert {'optimizer_state_dict', 'scheduler_state_dict', 'torch_rng_state'} <= periodic.keys()
    else:
        assert not (output / 'checkpoints').exists()
    source = Compound.from_smiles('CC(O)N')
    def predict(checkpoint):
        model = load_generator(model_path=str(checkpoint), params_path=None, device=torch.device('cpu'), strict=True)
        return model.predict([source], [Adduct.parse('[M+H]+')], [20.0]).spectra
    before = predict(output / checkpoint_name)
    relocated = tmp_path / 'relocated'
    moved.rename(relocated)
    after = predict(relocated / 'run' / checkpoint_name)
    torch.testing.assert_close(before.mz, after.mz)
    torch.testing.assert_close(before.intensity, after.intensity)
    train, validation = prepared_dataset_paths(relocated / 'dataset')
    train_reactions(model_config=config(), train_dir=train, val_dir=validation,
                    output_dir=relocated / 'run', resume=relocated / 'run' / checkpoint_name,
                    epochs=1, device='cpu', warmup_steps=0, checkpoint_interval_epochs=interval)
    checkpoint = torch.load(relocated / 'run' / 'last.pt', weights_only=False)
    assert checkpoint['epoch'] == epochs + 1
    assert checkpoint['training_settings']['train_dir'] == '../dataset/train_structures'
    assert checkpoint['training_settings']['resume'] == checkpoint_name
