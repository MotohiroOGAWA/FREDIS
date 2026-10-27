"""Build and install FREDIS and both native Python bindings in this environment."""
from pathlib import Path
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4, help="Parallel compiler jobs")
    parser.add_argument("--no-deps", action="store_true",
                        help="Use dependencies already installed in this environment")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parents[1]

    def run(*command):
        subprocess.run([sys.executable, *command], cwd=root, check=True)

    if not args.no_deps:
        run("-m", "pip", "install", "setuptools>=61", "wheel", "pyarrow>=20",
            "torch", "torch-geometric", "rdkit", "bidict", "pytest")
    run(str(root / "scripts/build-native.py"), "--similarity", "--jobs", str(args.jobs))
    options = ["--no-deps"] if args.no_deps else []
    run("-m", "pip", "install", "--no-build-isolation", *options, "-e", ".[similarity]")
    run("-c", "import fredis, msentity, msentity_similarity; "
        "print('FREDIS and native libraries are ready.')")


if __name__ == "__main__":
    main()
