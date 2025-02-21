import pytest

from gempyor.batch import (
    BatchSystem,
    SlurmBatchSystem,
    LocalBatchSystem,
    get_batch_system,
)


@pytest.mark.parametrize(
    ("name", "class_"),
    (("local", LocalBatchSystem), ("slurm", SlurmBatchSystem)),
)
def test_getting_default_batch_systems(name: str, class_: type[BatchSystem]) -> None:
    batch_system = get_batch_system(name)
    assert isinstance(batch_system, class_)


@pytest.mark.parametrize("name", ("missing batch system",))
@pytest.mark.parametrize("raise_on_missing", (True, False))
def test_getting_batch_system_missing(name: str, raise_on_missing: bool) -> None:
    from gempyor.batch.systems import _batch_systems

    assert name not in {bs.name for bs in _batch_systems}
    if raise_on_missing:
        with pytest.raises(
            ValueError,
            match=f"Batch system '{name}' not found in registered batch systems.",
        ):
            get_batch_system(name)
    else:
        batch_system = get_batch_system(name, raise_on_missing=False)
        assert batch_system is None
