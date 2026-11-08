from pathlib import Path
import zipfile
import pytest
import torch
from fredis.ml.input.training_storage import save_training_payload, compress_training_file


def test_compressed_payload_preserves_tensor_views_and_shared_storage(tmp_path):
    tensor = torch.arange(100_000, dtype=torch.int64).reshape(1000, 100)
    payload = {'schema': 'test', 'tensor': tensor, 'view': tensor[:, ::2],
               'zeros': torch.zeros(200_000), 'metadata': {'smiles': 'CCO'}}
    raw, compressed = tmp_path / 'raw.pt', tmp_path / 'compressed.pt'
    torch.save(payload, raw)
    save_training_payload(payload, compressed)
    assert compressed.stat().st_size < raw.stat().st_size / 2
    loaded = torch.load(compressed, weights_only=False)
    assert loaded['metadata'] == payload['metadata']
    for name in ('tensor', 'view', 'zeros'):
        assert torch.equal(loaded[name], payload[name])
        assert loaded[name].stride() == payload[name].stride()
    assert loaded['view'].untyped_storage().data_ptr() == loaded['tensor'].untyped_storage().data_ptr()
    with zipfile.ZipFile(compressed) as archive:
        assert archive.testzip() is None
        assert any(entry.compress_type == zipfile.ZIP_DEFLATED for entry in archive.infolist())


def test_compacting_old_files_is_atomic_and_readable(tmp_path, monkeypatch):
    path = tmp_path / 'old.preft.pt'
    torch.save({'tensor': torch.zeros(100_000)}, path)
    before = path.read_bytes()
    import fredis.ml.input.training_storage as storage
    replace = storage.os.replace
    monkeypatch.setattr(storage.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('disk error')))
    with pytest.raises(OSError, match='disk error'):
        compress_training_file(path, path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('.training-*'))
    monkeypatch.setattr(storage.os, 'replace', replace)
    compress_training_file(path, path)
    assert path.stat().st_size < len(before) / 10
    assert torch.equal(torch.load(path, weights_only=False)['tensor'], torch.zeros(100_000))
