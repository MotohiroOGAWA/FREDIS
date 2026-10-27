import math

import pandas as pd
import pytest

from fredis.domain.mass import parse_ce_to_ev


@pytest.mark.parametrize('ce,mz,instrument,expected', [
    ('50%', '250.0', None, 25.),
    ('20 V', '250.0', None, 20.),
    ('30 eV', None, None, 30.),
    ('0.04 keV', 'invalid', None, 40.),
    ('50', '250.0', 'Orbitrap', 25.),
    (20, None, None, 20.),
])
def test_ce_conversion_handles_numeric_metadata_strings(ce, mz, instrument, expected):
    assert parse_ce_to_ev(ce, mz, instrument) == expected


@pytest.mark.parametrize('ce,mz,instrument', [
    ('50%', 'invalid', None), ('50%', None, None), ('50%', pd.NA, None),
    ('50%', math.inf, None), (50, 'invalid', 'Orbitrap'),
    ('unknown', '250', None), (None, '250', None), (pd.NA, '250', None),
    (float('nan'), '250', None), ('inf', '250', None), ([], '250', None),
])
def test_failed_ce_conversion_returns_missing_value(ce, mz, instrument):
    assert parse_ce_to_ev(ce, mz, instrument) is None
