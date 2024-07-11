from datetime import datetime
import re

from gempyor.file_paths import run_id


class TestRunId:
    """Unit tests for the `gempyor.file_paths.run_id` function."""

    def test_get_run_id(self) -> None:
        # Setup
        before = datetime.now()
        rid = run_id()
        after = datetime.now()

        # Basic assertions
        assert isinstance(rid, str)
        assert re.match(r"^[0-9]{8}_[0-9]{6}$", rid)

        # Run id is between before/after
        before = before.replace(microsecond=0)
        after = after.replace(microsecond=0, second=after.second + 1)
        try:
            run_time = datetime.strptime(rid, "%Y%m%d_%H%M%S%Z")
        except:
            run_time = datetime.strptime(rid, "%Y%m%d_%H%M%S")
        assert run_time >= before
        assert run_time <= after
