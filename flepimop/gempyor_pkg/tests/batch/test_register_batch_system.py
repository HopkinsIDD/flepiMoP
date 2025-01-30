import pytest

from gempyor.batch import BatchSystem, register_batch_system, _reset_batch_systems


class TestBatchSystem(BatchSystem):
    name = "test"


def test_registration_adds_batch_system() -> None:
    _reset_batch_systems()
    from gempyor.batch import _batch_systems

    initial_len = len(_batch_systems)
    assert register_batch_system(TestBatchSystem()) is None
    from gempyor.batch import _batch_systems

    assert len(_batch_systems) == initial_len + 1


def test_registration_raises_error_on_duplicate() -> None:
    _reset_batch_systems()
    register_batch_system(TestBatchSystem())
    with pytest.raises(ValueError, match="^Batch system 'test' already registered.$"):
        register_batch_system(TestBatchSystem())
