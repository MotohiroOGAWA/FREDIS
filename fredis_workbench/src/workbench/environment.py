"""Read-only environment probe executed by the extension host."""
import json
import sys
import fredis
import torch
from rdkit import rdBase
print(json.dumps({"executable": sys.executable, "fredis": fredis.__file__, "torch": torch.__version__, "rdkit": rdBase.rdkitVersion, "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))
