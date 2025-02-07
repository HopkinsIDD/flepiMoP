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
    ("blocks", "chains", "samples", "simulations", "expected"),
    (
        (1, 1, 10, 25, JobSize(blocks=1, chains=1, samples=10, simulations=25)),
        (2, 1, 10, 25, JobSize(blocks=1, chains=1, samples=20, simulations=50)),
        (1, 2, 10, 25, JobSize(blocks=1, chains=1, samples=20, simulations=50)),
        (2, 2, 5, 10, JobSize(blocks=1, chains=1, samples=20, simulations=40)),
        (5, 1, 10, 20, JobSize(blocks=1, chains=1, samples=25, simulations=50)),
        (1, 5, 10, 20, JobSize(blocks=1, chains=1, samples=25, simulations=50)),
        (5, 5, 1, 2, JobSize(blocks=1, chains=1, samples=25, simulations=50)),
        (4, 10, 100, 200, JobSize(blocks=1, chains=1, samples=25, simulations=50)),
    ),
)
def test_size_from_jobs_simulations_blocks_for_select_values(
    recwarn: pytest.WarningsRecorder,
    blocks: int,
    chains: int,
    samples: int,
    simulations: int,
    expected: JobSize,
) -> None:
    batch_system = get_batch_system("local")
    assert (
        batch_system.size_from_jobs_simulations_blocks(blocks, chains, samples, simulations)
        == expected
    )
    prelim_size = JobSize(
        blocks=blocks, chains=chains, samples=samples, simulations=simulations
    )
    assert len(recwarn) == sum(
        (
            False if prelim_size.chains is None else prelim_size.chains > 1,
            False if prelim_size.blocks is None else prelim_size.blocks > 1,
            False if prelim_size.total_samples is None else prelim_size.total_samples > 25,
            (
                False
                if prelim_size.total_simulations is None
                else prelim_size.total_simulations > 50
            ),
        )
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
