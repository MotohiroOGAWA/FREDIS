"""Build a platform wheel that bundles the msentity C library.

    python build_wheel.py --native ../natives/linux-x64/libmsentity_c.so \
                          --platform-tag manylinux_2_35_x86_64 --out dist

The package itself is pure Python (ctypes), so one wheel per platform works
for every Python 3 version: the result is tagged py3-none-<platform>.
Requires the `wheel` package (for `python -m wheel tags`).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB_DIR = HERE / "msentity" / "lib"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--native", required=True, type=Path,
                        help="libmsentity_c.so / msentity_c.dll / libmsentity_c.dylib")
    parser.add_argument("--platform-tag", required=True,
                        help="e.g. manylinux_2_35_x86_64, win_amd64, macosx_14_0_arm64")
    parser.add_argument("--out", type=Path, default=HERE / "dist")
    args = parser.parse_args()

    native = args.native.resolve()
    if not native.is_file():
        parser.error(f"native library not found: {native}")
    args.out.mkdir(parents=True, exist_ok=True)

    shutil.rmtree(LIB_DIR, ignore_errors=True)
    LIB_DIR.mkdir(parents=True)
    shutil.copy2(native, LIB_DIR / native.name)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, "-m", "pip", "wheel", str(HERE), "--no-deps",
                            "--no-build-isolation", "-w", tmp], check=True)
            (built,) = Path(tmp).glob("msentity-*.whl")
            subprocess.run([sys.executable, "-m", "wheel", "tags", "--remove",
                            "--python-tag", "py3", "--abi-tag", "none",
                            "--platform-tag", args.platform_tag, str(built)], check=True)
            for wheel in Path(tmp).glob("msentity-*.whl"):
                shutil.move(str(wheel), args.out / wheel.name)
                print(args.out / wheel.name)
    finally:
        shutil.rmtree(LIB_DIR, ignore_errors=True)
        shutil.rmtree(HERE / "build", ignore_errors=True)
        for info in HERE.glob("*.egg-info"):
            shutil.rmtree(info, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
