"""Exercise the shared msentity backend used by the embedded Workbench viewer."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

backend = Path(__file__).resolve().parents[2] / 'fredis/libs/msentity/vscode-extension/python/backend.py'
with tempfile.TemporaryDirectory() as directory:
    source = Path(directory) / 'example.msp'
    source.write_text('Name: Ethanol\nSMILES: CCO\nNum Peaks: 1\n31 100\n\nName: Benzene\nSMILES: c1ccccc1\nNum Peaks: 1\n78 100\n\n')
    requests = [
        {'type': 'page', 'dataset_id': str(source), 'filters': [
            {'column': 'SMILES', 'operator': 'smarts', 'value': '[OX2H]'}]},
        {'type': 'render-structure', 'smiles': 'CCO'},
        {'type': 'update-cell', 'dataset_id': str(source), 'row_id': 0, 'column': 'SMILES', 'value': 'CCC'},
        {'type': 'page', 'dataset_id': str(source), 'filters': [
            {'column': 'SMILES', 'operator': 'smarts', 'value': '[OX2H]'}]},
    ]
    result = subprocess.run([sys.executable, '-u', str(backend), str(source)],
                            input=''.join(json.dumps(request) + '\n' for request in requests),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    messages = [json.loads(line.removeprefix('MSENTITY_JSON:'))
                for line in result.stdout.splitlines() if line.startswith('MSENTITY_JSON:')]
    assert not [message for message in messages if message['type'] in ('error', 'filter-error', 'chemistry-error')], messages
    assert next(message for message in messages if message['type'] == 'backend-ready')['dataset']['total_rows'] == 2
    capabilities = next(message['chemistry'] for message in messages if message['type'] == 'capabilities')
    assert capabilities['smarts_filter'] and capabilities['structure_render'], capabilities
    pages = [message['value'] for message in messages if message['type'] == 'dataset-page']
    assert [page['total_rows'] for page in pages] == [1, 0], pages
    assert pages[0]['structure_smiles_column'] == 'SMILES'
    depiction = next(message for message in messages if message['type'] == 'structure-rendered')
    assert '<svg' in depiction['svg']
    # Opening a file directly remains supported by the same backend.
    direct = subprocess.run([sys.executable, '-u', str(backend), str(source)],
                            input='', capture_output=True, text=True, timeout=30)
    assert direct.returncode == 0, direct.stderr
    direct_messages = [json.loads(line.removeprefix('MSENTITY_JSON:'))
                       for line in direct.stdout.splitlines() if line.startswith('MSENTITY_JSON:')]
    ready = next(message for message in direct_messages if message['type'] == 'backend-ready')
    assert ready['dataset']['total_rows'] == 2
print('Shared spectrum backend checks passed.')
