"""Archive pretrained dependencies with portable dataset-relative references."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def relocate_sources(chemistry, source_base, destination_base):
    result = deepcopy(chemistry)
    for record in result.get("reaction_models", []):
        if record.get("source"):
            original = (Path(source_base) / record["source"]).resolve()
            record["source"] = Path(
                os.path.relpath(original, destination_base)
            ).as_posix()
    return result


class StagedPretrainedModels:
    def __init__(self, chemistry, directory):
        self.chemistry = deepcopy(chemistry)
        self.files = {}
        resolved = {}
        for record in self.chemistry.get("reaction_models", []):
            if not record.get("source"):
                continue
            source = Path(record["source"]).resolve()
            if source not in resolved:
                before = source.stat()
                temporary = Path(directory) / "copy.pt"
                digest = hashlib.sha256()
                with source.open("rb") as reader, temporary.open("wb") as writer:
                    for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                        writer.write(chunk)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise ValueError(
                        f"Pretrained model changed during copying: {source}"
                    )
                sha = digest.hexdigest()
                if sha not in self.files:
                    suffix = (
                        ".reaction-model.pt"
                        if source.name.endswith(".reaction-model.pt")
                        else ".pt"
                    )
                    filename = sha + suffix
                    staged = Path(directory) / filename
                    temporary.rename(staged)
                    self.files[sha] = dict(
                        file=filename, sha256=sha, bytes=after.st_size, staged=staged
                    )
                else:
                    temporary.unlink()
                resolved[source] = self.files[sha]["file"]
            record["source"] = "pretrained_models/" + resolved[source]

    def install(self, output):
        output = Path(output)
        if self.files:
            directory = output / "pretrained_models"
            directory.mkdir(parents=True, exist_ok=True)
            for item in self.files.values():
                shutil.copyfile(item["staged"], directory / item["file"])
            manifest = {
                "files": [
                    {key: value for key, value in item.items() if key != "staged"}
                    for item in self.files.values()
                ],
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
        return deepcopy(self.chemistry)


@contextmanager
def stage_pretrained_models(chemistry):
    # Stage before resetting outputs: restoring a dataset's preparation settings
    # may select the model copies already inside that same output directory.
    with tempfile.TemporaryDirectory(prefix="fredis-pretrained-") as directory:
        yield StagedPretrainedModels(chemistry, directory)
