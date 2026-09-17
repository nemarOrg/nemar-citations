"""Tests for the shared freshness-state cache (issue #197).

Real temp files and real clock arithmetic; no mocks. These cover the gate that
keeps `find-mentions` inside OpenAlex's daily credit budget, so the ordering
and cold-start cases matter as much as the round-trip.
"""

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import TestCase

from dataset_citations.core.run_state import (
    checked_within,
    load_state,
    parse_checked_at,
    save_state,
    stalest_first,
    stamp_checked,
)


def _ago(**kwargs) -> str:
    return (datetime.now(UTC) - timedelta(**kwargs)).isoformat()


class StateRoundTripTests(TestCase):
    def test_absent_file_loads_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_state(str(Path(tmp) / "nope.json")), {})

    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / ".mention_state.json")
            save_state(path, {"ds1": "2026-06-18T00:00:00+00:00"})
            self.assertEqual(load_state(path), {"ds1": "2026-06-18T00:00:00+00:00"})

    def test_corrupt_file_loads_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mention_state.json"
            path.write_text("not json at all")
            self.assertEqual(load_state(str(path)), {})

    def test_non_object_json_loads_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mention_state.json"
            path.write_text(json.dumps(["a", "list"]))
            self.assertEqual(load_state(str(path)), {})

    def test_non_string_values_are_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".mention_state.json"
            path.write_text(json.dumps({"ds1": "2026-06-18T00:00:00+00:00", "ds2": 17}))
            self.assertEqual(
                load_state(str(path)), {"ds1": "2026-06-18T00:00:00+00:00"}
            )

    def test_save_failure_does_not_raise(self) -> None:
        # A directory path is not writable as a file; the cron must survive it.
        with tempfile.TemporaryDirectory() as tmp:
            save_state(tmp, {"ds1": _ago(days=1)})


class FreshnessTests(TestCase):
    def test_recent_is_fresh(self) -> None:
        self.assertTrue(checked_within("ds1", {"ds1": _ago(days=1)}, 7 * 86400))

    def test_older_than_window_is_stale(self) -> None:
        self.assertFalse(checked_within("ds1", {"ds1": _ago(days=8)}, 7 * 86400))

    def test_missing_and_unparseable_are_stale(self) -> None:
        self.assertFalse(checked_within("ds1", {}, 7 * 86400))
        self.assertFalse(checked_within("ds1", {"ds1": "not-a-date"}, 7 * 86400))
        self.assertFalse(checked_within("ds1", {"ds1": 42}, 7 * 86400))  # type: ignore[dict-item]

    def test_naive_timestamp_read_as_utc(self) -> None:
        naive = datetime.now(UTC).replace(tzinfo=None).isoformat()
        self.assertTrue(checked_within("ds1", {"ds1": naive}, 7 * 86400))

    def test_stamp_then_fresh(self) -> None:
        state: dict[str, str] = {}
        stamp_checked(state, "ds1")
        self.assertTrue(checked_within("ds1", state, 60))
        self.assertIsNotNone(parse_checked_at(state, "ds1"))


class StalestFirstTests(TestCase):
    def test_never_checked_sort_before_checked(self) -> None:
        state = {"b": _ago(days=1)}
        self.assertEqual(stalest_first(["b", "a"], state), ["a", "b"])

    def test_oldest_checked_comes_first(self) -> None:
        state = {"a": _ago(days=1), "b": _ago(days=9), "c": _ago(days=5)}
        self.assertEqual(stalest_first(["a", "b", "c"], state), ["b", "c", "a"])

    def test_cold_start_preserves_input_order(self) -> None:
        # Every dataset unchecked: a truncated first run must be reproducible,
        # not arbitrary, so ties keep the caller's (alphabetical) order.
        ids = ["ds1", "ds2", "ds3", "ds4"]
        self.assertEqual(stalest_first(ids, {}), ids)

    def test_returns_all_inputs(self) -> None:
        ids = ["a", "b", "c"]
        self.assertCountEqual(stalest_first(ids, {"a": _ago(days=2)}), ids)

    def test_unparseable_timestamp_sorts_as_never_checked(self) -> None:
        state = {"a": _ago(days=1), "b": "garbage"}
        self.assertEqual(stalest_first(["a", "b"], state), ["b", "a"])


class RollingCoverageTests(TestCase):
    """The whole point of #197: a capped nightly slice still covers everything.

    Simulates the cron's `--max-age-days 7 --max-datasets 250` against a
    750-dataset corpus and asserts full coverage without ever exceeding the
    per-run cap.
    """

    def test_capped_runs_cover_the_corpus_without_exceeding_the_cap(self) -> None:
        corpus = [f"ds{i:06d}" for i in range(750)]
        state: dict[str, str] = {}
        cap = 250
        seen: set[str] = set()

        for _ in range(3):
            stale = [d for d in corpus if not checked_within(d, state, 7 * 86400)]
            batch = stalest_first(stale, state)[:cap]
            self.assertLessEqual(len(batch), cap)
            for dataset_id in batch:
                stamp_checked(state, dataset_id)
                seen.add(dataset_id)

        self.assertEqual(seen, set(corpus))
        # Everything is now fresh, so a fourth run would do no work at all.
        remaining = [d for d in corpus if not checked_within(d, state, 7 * 86400)]
        self.assertEqual(remaining, [])


class BoundaryFreshnessTests(TestCase):
    """The freshness window is half-open: exactly at the boundary is stale.

    Regression for #80. With `<=`, a cron whose period equals the freshness
    window landed on the boundary every single run and never refreshed.
    """

    def test_exactly_at_the_window_is_stale(self) -> None:
        window = 7 * 86400
        state = {
            "on000001": (datetime.now(UTC) - timedelta(seconds=window)).isoformat()
        }
        self.assertFalse(checked_within("on000001", state, window))

    def test_just_inside_the_window_is_fresh(self) -> None:
        window = 7 * 86400
        state = {
            "on000001": (datetime.now(UTC) - timedelta(seconds=window - 60)).isoformat()
        }
        self.assertTrue(checked_within("on000001", state, window))

    def test_past_the_window_is_stale(self) -> None:
        window = 7 * 86400
        state = {
            "on000001": (datetime.now(UTC) - timedelta(seconds=window + 60)).isoformat()
        }
        self.assertFalse(checked_within("on000001", state, window))
