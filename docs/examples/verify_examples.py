"""Execute the complete documentation examples in a fresh temporary workspace."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="New directory for logs and example outputs."
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output is None:
        output = Path(tempfile.mkdtemp(prefix="fredis-doc-examples-"))
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    (output / "docs").symlink_to(root / "docs", target_is_directory=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    pages = [
        "quickstart.md",
        "guides/fine-tuning.md",
        "api/chemistry.md",
        "api/data.md",
        "api/prediction.md",
        "api/training.md",
    ]
    results = []
    for page in pages:
        snippets = re.findall(
            r"```(bash|python)\n(.*?)\n```", (root / "docs" / page).read_text(), re.S
        )
        namespace = ""
        for index, (language, body) in enumerate(snippets):
            if language == "bash":
                command = shlex.split(body.replace("\\\n", " "))
                if command[0] == "python":
                    command[0] = sys.executable
            else:
                namespace += body + "\n"
                command = [sys.executable, "-c", namespace]
            log = output / (page.replace("/", "-").replace(".md", "") + f"-{index}.log")
            started = time.monotonic()
            with log.open("w") as handle:
                result = subprocess.run(
                    command,
                    cwd=output,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                print(log.read_text()[-8000:])
                raise SystemExit(f"Example failed: {page}:{index}; {log}")
            results.append(
                dict(
                    page=page,
                    block=index,
                    language=language,
                    seconds=round(time.monotonic() - started, 3),
                    log=str(log),
                )
            )
            print(f"{page}:{index}: passed", flush=True)
    report = output / "verification.json"
    report.write_text(json.dumps(dict(all_passed=True, examples=results), indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    main()
