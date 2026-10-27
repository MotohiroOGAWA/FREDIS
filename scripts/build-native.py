"""Build source snapshots with the Arrow/Parquet shipped in pyarrow.

Requires CMake, a C++20 compiler, HDF5 development files and nlohmann-json.
Extra CMake options may be passed after --, e.g. -DCMAKE_PREFIX_PATH=/deps.
"""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--similarity', action='store_true')
parser.add_argument('--jobs', default='4')
args, extra = parser.parse_known_args()
extra = [x for x in extra if x != '--']
root = Path(__file__).resolve().parents[1]
import pyarrow
arrow = Path(pyarrow.__file__).parent
config = root / 'build/native-deps'
for name in ['Arrow', 'Parquet']:
    folder = config / name
    folder.mkdir(parents=True, exist_ok=True)
    library = name.lower()
    candidates = list(arrow.glob(f'lib{library}.so.*')) + list(arrow.glob(f'lib{library}.*.dylib'))
    if not candidates:
        raise SystemExit('This helper supports Linux/macOS pyarrow wheels; on Windows use the upstream CMake build.')
    (folder / f'{name}Config.cmake').write_text(f'''if(NOT TARGET {name}::{library}_shared)
add_library({name}::{library}_shared SHARED IMPORTED)
set_target_properties({name}::{library}_shared PROPERTIES IMPORTED_LOCATION "{candidates[0]}" INTERFACE_INCLUDE_DIRECTORIES "{arrow / 'include'}")
endif()
''')
modules = ['msentity'] + (['msentity-similarity'] if args.similarity else [])
for name in modules:
    source = root / 'upstream' / name
    build = source / 'build'
    options = [f'-DArrow_DIR={config / "Arrow"}', f'-DParquet_DIR={config / "Parquet"}',
               '-DMSENTITY_BUILD_TESTS=OFF', '-DMSENTITY_BUILD_TOOLS=OFF']
    if name == 'msentity-similarity':
        options += ['-DMSENTITY_SIMILARITY_BUILD_TESTS=OFF', '-DMSENTITY_SIMILARITY_BUILD_TOOLS=OFF',
                    f'-DFETCHCONTENT_SOURCE_DIR_MSENTITY={root / "upstream/msentity"}']
    subprocess.run(['cmake', '-S', str(source), '-B', str(build), *options, *extra], check=True)
    subprocess.run(['cmake', '--build', str(build), '-j', args.jobs], check=True)
    # Install the Python facade and bundle the built C library beside it.
    package = source / 'bindings/python' / name.replace('-', '_')
    libdir = package / 'lib'
    libdir.mkdir(exist_ok=True)
    import shutil
    for lib in build.rglob('lib' + name.replace('-', '_') + '_c.*'):
        if lib.suffix in {'.so', '.dylib'}:
            shutil.copy2(lib, libdir / lib.name)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', str(source / 'bindings/python')], check=True)
