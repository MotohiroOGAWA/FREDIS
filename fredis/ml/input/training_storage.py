"""Lossless, atomic storage of prepared training tensors, readable by torch.load."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import torch


def _temporary(parent):
    fd, name = tempfile.mkstemp(prefix='.training-', suffix='.pt', dir=parent)
    os.close(fd)
    return Path(name)


def compress_training_file(source, destination):
    """Copy a torch ZIP archive with DEFLATE; preserve tensor bytes and shared storage.

    Can atomically compact an existing file when source == destination. Peak
    memory is bounded by the copy buffer rather than the tensor archive size.
    """
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(destination.parent)
    try:
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(
            temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as compressed:
            for entry in original.infolist():
                with original.open(entry) as reader, compressed.open(entry.filename, 'w', force_zip64=True) as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
            expected = [(entry.filename, entry.file_size, entry.CRC) for entry in original.infolist()]
        with zipfile.ZipFile(temporary) as saved:
            actual = [(entry.filename, entry.file_size, entry.CRC) for entry in saved.infolist()]
        if actual != expected:
            raise ValueError('Compressed training archive differs from its source')
        # Compression should never make an archive larger.
        if temporary.stat().st_size >= source.stat().st_size:
            if source.resolve() == destination.resolve():
                return
            shutil.copyfile(source, temporary)
        os.chmod(temporary, source.stat().st_mode & 0o777)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_training_payload(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _temporary(path.parent)
    try:
        if path.exists():
            os.chmod(raw, path.stat().st_mode & 0o777)
        torch.save(payload, raw)
        compress_training_file(raw, path)
    finally:
        raw.unlink(missing_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Losslessly compact prepared FREDIS tensors in place, one file at a time.')
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error('Expected a training dataset directory')
    files = sorted(args.directory.rglob('*.preft.pt'))
    before = after = 0
    for index, file in enumerate(files, 1):
        size = file.stat().st_size
        with zipfile.ZipFile(file) as archive:
            already_compressed = all(entry.compress_type != zipfile.ZIP_STORED or not entry.file_size for entry in archive.infolist())
        if not already_compressed:
            compress_training_file(file, file)
        before += size
        after += file.stat().st_size
        print(f'{index}/{len(files)} files | {before:,} → {after:,} bytes | {file.name}', flush=True)
    print(f'Saved {before-after:,} bytes across {len(files)} files.', flush=True)


if __name__ == '__main__':
    main()
