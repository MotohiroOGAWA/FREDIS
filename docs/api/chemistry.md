# Chemistry API

Fragmenter owns the configured reaction search, fragment-ion rules, and peak
assignment tolerance. Construct it from a complete chemistry definition before
running a source search.

## Load a Fragmenter

```python
import json
from pathlib import Path
from fredis.domain.fragment import Fragmenter
from fredis.libs.mmkit.mmkit import Compound

configuration = json.loads(Path("docs/examples/chemistry.fragmenter.pft").read_text())
fragmenter = Fragmenter.from_dict(configuration)
print(fragmenter.tree_max_reaction_count)
print(len(fragmenter.reaction_pattern_set.patterns))
source = Compound.from_smiles("CCO")
tree = fragmenter.build_fragment_tree(
    source,
    max_unique_fragment_smiles=100,
    max_reaction_combinations=1000,
)
```

The example configuration produces a maximum reaction depth of 2 and one
pattern definition. `tree` is a chemical FragmentTree, rather than a neural
prediction or measured-spectrum target.

## Search and persistence contract

`build_fragment_tree()` materializes compatible source reaction sets.
`build_fragment_ion_tree()` additionally enumerates configured ion/shift formula
explanations. Source reactions are defined by SMARTS; main-adduct resolution and
mass-tolerance assignment belong to the Fragmenter.

`max_unique_fragment_smiles` excludes the source structure. Combination limits
count attempted raw sets before RDKit execution. `-1` is unlimited; an overflow
can reduce retained search depth in the preparation workflow. See
[preparation limits](../guides/data-preparation.md#bound-expensive-searches).

Use `to_dict()` to serialize the resolved chemistry. The generated pattern IDs
belong to this definition set and can change when patterns are combined.

```{eval-rst}
.. autoclass:: fredis.domain.fragment.fragmenter.Fragmenter
   :members: from_dict, to_dict, build_fragment_tree, build_fragment_ion_tree, get_index_by_adduct_type
   :undoc-members:
   :exclude-members: __new__
```

## Reaction objects

Pattern sets describe transformation families; concrete reaction instances
refer to matches on one source. ReactionSequence combines compatible instances
and applies their effects. The [chemistry concepts](../concepts/chemistry.md)
explain pattern, match, set, Compound, and history identity.

The complete neural/prepared-data workflow usually enters through
`ReactionStructureBuilder` rather than manually translating tree objects into
training indices. See the [data API](data.md).
