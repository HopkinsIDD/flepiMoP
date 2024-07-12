from datetime import datetime, timezone
import re

import pytest

from gempyor.file_paths import run_id


class TestRunId:
    """Unit tests for the `gempyor.file_paths.run_id` function."""

    run_id_regex = re.compile(r"(?i)^[0-9]{8}_[0-9]{6}([a-z]+)?$")

    def test_get_run_id_default_timestamp(self) -> None:
        # Setup
        before = datetime.now()
        rid = run_id()
        after = datetime.now()

        # Basic assertions
        assert isinstance(rid, str)
        assert self.run_id_regex.match(rid)

        # Run id is between before/after
        before = before.replace(microsecond=0)
        after = after.replace(microsecond=0, second=after.second + 1)
        run_time = datetime.strptime(rid, "%Y%m%d_%H%M%S")
        assert run_time >= before
        assert run_time <= after

    @pytest.mark.parametrize(
        "timestamp",
        [(None), (datetime.now()), (datetime(2024, 1, 1, tzinfo=timezone.utc))],
    )
    def test_get_run_id_user_provided_timestamp(
        self, timestamp: None | datetime
    ) -> None:
        # Setup
        rid = run_id(timestamp=timestamp)

        # Assertions
        assert isinstance(rid, str)
        assert self.run_id_regex.match(rid)
        try:
            run_time = datetime.strptime(rid, "%Y%m%d_%H%M%S%Z")
        except:
            run_time = datetime.strptime(rid, "%Y%m%d_%H%M%S")
        assert isinstance(run_time, datetime)
