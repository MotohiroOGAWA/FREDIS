import pytest
from fredis.ml.common.torch_utils.early_stopping import EarlyStopping


def test_improvement_resets_patience_and_uses_global_best():
    stop = EarlyStopping(patience=2)
    assert stop.update(1.0)
    assert not stop.update(1.1)
    assert stop.update(0.9)
    assert stop.bad_epochs == 0
    assert not stop.update(0.9 - 1e-7)
    assert not stop.should_stop
    assert not stop.update(1.2)
    assert stop.should_stop
    assert stop.best == 0.9


def test_resume_and_disabled_stopping():
    stop = EarlyStopping(patience=3, best=0.4, bad_epochs=2)
    assert not stop.update(0.5)
    assert stop.should_stop
    disabled = EarlyStopping(patience=0)
    for loss in [1.0, 2.0, 3.0]:
        disabled.update(loss)
    assert not disabled.should_stop and disabled.best == 1.0


def test_invalid_settings():
    with pytest.raises(ValueError):
        EarlyStopping(patience=-1)
