from datetime import timedelta
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess
import sys

import pytest

from gempyor.batch import JobSize, JobSubmission, LocalBatchSystem, get_batch_system
from gempyor.testing import sample_script


def test_local_batch_system_registered_by_default() -> None:
    batch_system = get_batch_system("local")
    assert isinstance(batch_system, LocalBatchSystem)
    assert batch_system.name == "local"


@pytest.mark.parametrize(
    ("chains", "simulations", "blocks", "expected"),
    (
        (1, 1, 1, JobSize(chains=1, simulations=1, blocks=1)),
        (2, 1, 1, JobSize(chains=1, simulations=1, blocks=1)),
        (1, 20, 1, JobSize(chains=1, simulations=10, blocks=1)),
        (1, 1, 20, JobSize(chains=1, simulations=10, blocks=1)),
        (1, 5, 5, JobSize(chains=1, simulations=10, blocks=1)),
        (5, 5, 5, JobSize(chains=1, simulations=10, blocks=1)),
        (1, 10, 1, JobSize(chains=1, simulations=10, blocks=1)),
        (1, 1, 10, JobSize(chains=1, simulations=10, blocks=1)),
        (1, 2, 5, JobSize(chains=1, simulations=10, blocks=1)),
        (1, 3, 3, JobSize(chains=1, simulations=9, blocks=1)),
    ),
)
def test_size_from_jobs_simulations_blocks_for_select_values(
    chains: int, simulations: int, blocks: int, expected: JobSize
) -> None:
    batch_system = get_batch_system("local")
    jobs_warning = chains != 1
    simulations_warning = blocks * simulations > 10
    if jobs_warning and simulations_warning:
        with pytest.warns(
            UserWarning,
            match=(
                "^Local batch system only supports 1 chain "
                f"but was given {chains}, overriding.$"
            ),
        ):
            with pytest.warns(
                UserWarning,
                match=(
                    "^Local batch system only supports 10 blocks x simulations "
                    f"but was given {blocks * simulations}, overriding.$"
                ),
            ):
                assert (
                    batch_system.size_from_jobs_simulations_blocks(
                        chains, simulations, blocks
                    )
                    == expected
                )
    elif jobs_warning:
        with pytest.warns(
            UserWarning,
            match=(
                "^Local batch system only supports 1 chain "
                f"but was given {chains}, overriding.$"
            ),
        ):
            assert (
                batch_system.size_from_jobs_simulations_blocks(chains, simulations, blocks)
                == expected
            )
    elif simulations_warning:
        with pytest.warns(
            UserWarning,
            match=(
                "^Local batch system only supports 10 blocks x simulations "
                f"but was given {blocks * simulations}, overriding.$"
            ),
        ):
            assert (
                batch_system.size_from_jobs_simulations_blocks(chains, simulations, blocks)
                == expected
            )
    else:
        assert (
            batch_system.size_from_jobs_simulations_blocks(chains, simulations, blocks)
            == expected
        )


@pytest.mark.parametrize("executable", (True, False))
@pytest.mark.parametrize("verbosity", (None, logging.DEBUG, logging.INFO, logging.WARNING))
@pytest.mark.parametrize("dry_run", (True, False))
def test_local_submit(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    executable: bool,
    verbosity: int,
    dry_run: bool,
) -> None:
    batch_system = get_batch_system("local")
    script = sample_script("example", tmp_path, executable)

    with patch("gempyor.batch.subprocess.Popen") as subprocess_popen_patch:
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"Hello local\n", b"")
        mock_process.returncode = 0
        mock_process.args = [str(script.absolute())]
        mock_process.pid = 12345
        subprocess_popen_patch.return_value = mock_process

        job_result = batch_system.submit(script, None, verbosity, dry_run)
        assert job_result is None if dry_run else isinstance(job_result, JobSubmission)

        if dry_run:
            subprocess_popen_patch.assert_not_called()
        else:
            subprocess_popen_patch.assert_called_once()

        log_messages_by_level = {
            hash((logging.DEBUG, True)): 2,
            hash((logging.DEBUG, False)): 4,
            hash((logging.INFO, True)): 1,
            hash((logging.INFO, False)): 2,
        }
        assert len(caplog.records) == log_messages_by_level.get(
            hash((verbosity, dry_run)), 0
        ) + (1 if not executable and verbosity is not None else 0)
