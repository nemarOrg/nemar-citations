"""Tests for run_find_mentions orchestration. Issues #169, #246.

No Mocks and no internet. An empty catalog cache file short-circuits the
catalog fetch, and the real `AccessionSearchBackend` searches a local HTTP
server that answers `/works?filter=fulltext.search:<term>` the way OpenAlex
does: pages of works, an empty page, a 429, or a body that is not a results
page. Two things differ from production: the base URL, and a single attempt per
request, so a 429 fails at once instead of sleeping through retries. Real files
on disk. The live OpenAlex round trip is gated behind RUN_INTEGRATION_TESTS=1.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import stat
import tempfile
import threading
import unittest
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self
from unittest import TestCase
from urllib.parse import parse_qs, urlsplit

from opencite.clients.openalex import OpenAlexClient
from opencite.config import Config

from dataset_citations.backends.accession_search import AccessionSearchBackend
from dataset_citations.cli import find_mentions as cli

# One answer from the local server: an HTTP status and a body.
Reply = tuple[int, bytes]
# Decides the answer to a search from its term and its paging cursor.
ReplyFn = Callable[[str, str], Reply]


def _page(works: list[dict[str, Any]], next_cursor: str | None = None) -> Reply:
    body = {"meta": {"next_cursor": next_cursor}, "results": works}
    return 200, json.dumps(body).encode()


EMPTY = _page([])
# OpenAlex's answer once the budget is spent. The server sends Retry-After: 0,
# and opencite makes a single attempt here, so the term fails at once.
RATE_LIMITED: Reply = (429, b'{"error": "Rate limit exceeded"}')
# Valid JSON that is not a results page: an upstream contract break.
NOT_A_PAGE: Reply = (200, b"[]")


def _work(openalex_id: str, doi: str) -> dict[str, Any]:
    """One OpenAlex work, in the shape the API returns."""
    return {
        "id": f"https://openalex.org/{openalex_id}",
        "doi": f"https://doi.org/{doi}",
        "title": "A paper naming the dataset",
        "publication_year": 2026,
        "cited_by_count": 0,
        "authorships": [{"author": {"display_name": "Someone"}}],
        "primary_location": {"source": {"display_name": "A Journal"}},
        "ids": {"openalex": f"https://openalex.org/{openalex_id}"},
    }


def _bypass_proxies_for_loopback(test: TestCase) -> None:
    """httpx honors HTTP(S)_PROXY and ALL_PROXY and does not exempt loopback,
    so a developer's proxy would swallow the requests meant for the local
    server. Restored at cleanup.
    """
    for var in ("NO_PROXY", "no_proxy"):
        previous = os.environ.get(var)
        os.environ[var] = "127.0.0.1"
        if previous is None:
            test.addCleanup(os.environ.pop, var, None)
        else:
            test.addCleanup(os.environ.__setitem__, var, previous)


class _OpenAlexServer:
    """A local HTTP server answering OpenAlex full-text searches for one test.

    `reply` decides each answer and may be swapped mid-test; `searched` logs
    the term of every search request, in order; `backend` is a real
    `AccessionSearchBackend` pointed here.
    """

    def __init__(self, test: TestCase) -> None:
        self.reply: ReplyFn = lambda _term, _cursor: EMPTY
        self.searched: list[str] = []
        _bypass_proxies_for_loopback(test)
        server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        # A short poll interval keeps shutdown (run at cleanup) fast.
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        test.addCleanup(server.server_close)
        test.addCleanup(server.shutdown)
        self.backend = self._backend(f"http://127.0.0.1:{server.server_port}")

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urlsplit(self.path)
                query = parse_qs(url.query)
                if url.path == "/works":
                    term = query["filter"][0].removeprefix("fulltext.search:")
                    outer.searched.append(term)
                    status, body = outer.reply(term, query["cursor"][0])
                else:
                    status, body = 404, b'{"error": "Not Found"}'
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                if status == 429:
                    self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                """Keep the test output quiet."""

        return Handler

    @staticmethod
    def _backend(base_url: str) -> AccessionSearchBackend:
        class LocalOpenAlex(OpenAlexClient):
            def __init__(self, config: Config) -> None:
                super().__init__(config)
                self.base_url = base_url
                self.max_retries = 1

            async def __aenter__(self) -> Self:
                entered = await super().__aenter__()
                # If opencite stopped honoring base_url, these tests would
                # quietly search live OpenAlex again (#246); fail instead.
                assert self._client is not None
                assert str(self._client.base_url).startswith(base_url)
                return entered

        class LocalBackend(AccessionSearchBackend):
            _openalex_client_cls = LocalOpenAlex

        return LocalBackend(max_results=0)


def _args(
    citations_dir: Path,
    *,
    dataset_list_file: str | None = None,
    max_age_days: int = 0,
    max_datasets: int = 0,
    catalog: list[dict[str, Any]] | None = None,
) -> argparse.Namespace:
    cache = citations_dir.parent / "catalog.json"
    # A fresh cache file means _load_source_id_map makes no request.
    cache.write_text(json.dumps(catalog or []))
    return argparse.Namespace(
        citations_dir=str(citations_dir),
        dataset_list_file=dataset_list_file,
        catalog_cache=cache,
        catalog_cache_max_age=3600,
        # 0 = freshness gate off, so the pre-existing cases below still exercise
        # every seeded dataset regardless of the state cache (issue #197).
        max_age_days=max_age_days,
        max_datasets=max_datasets,
    )


def _seed(citations_dir: Path, dataset_id: str) -> Path:
    citations_dir.mkdir(parents=True, exist_ok=True)
    path = citations_dir / f"{dataset_id}_citations.json"
    path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "num_citations": 0,
                "date_last_updated": "2026-06-18T00:00:00+00:00",
                "metadata": {"fetch_status": "success"},
                "citation_details": [],
            },
            indent=2,
        )
    )
    return path


def _state(citations_dir: Path) -> dict[str, str]:
    return json.loads((citations_dir / ".mention_state.json").read_text())


class RunFindMentionsTests(TestCase):
    def setUp(self) -> None:
        self.openalex = _OpenAlexServer(self)
        self.backend = self.openalex.backend

    def test_processes_valid_datasets_via_glob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            _seed(cdir, "nm000207")
            cli.run_find_mentions(_args(cdir), self.backend)
            self.assertEqual(self.openalex.searched, ["ds002718", "nm000207"])
            # Processed datasets gain the searched_accessions marker.
            for did, acc in (("ds002718", "ds002718"), ("nm000207", "nm000207")):
                data = json.loads((cdir / f"{did}_citations.json").read_text())
                self.assertEqual(data["metadata"]["searched_accessions"], [acc])

    def test_dataset_list_file_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            _seed(cdir, "ds000247")  # present but NOT in the list -> untouched
            list_file = Path(tmp) / "list.txt"
            list_file.write_text("ds002718\n")
            cli.run_find_mentions(
                _args(cdir, dataset_list_file=str(list_file)), self.backend
            )
            self.assertEqual(self.openalex.searched, ["ds002718"])
            listed = json.loads((cdir / "ds002718_citations.json").read_text())
            self.assertIn("searched_accessions", listed["metadata"])
            unlisted = json.loads((cdir / "ds000247_citations.json").read_text())
            self.assertNotIn("searched_accessions", unlisted["metadata"])

    def test_skips_missing_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            cdir.mkdir(parents=True)
            list_file = Path(tmp) / "list.txt"
            list_file.write_text("ds999999\n")  # no JSON on disk
            # Should return normally (nothing processed), no exit.
            cli.run_find_mentions(
                _args(cdir, dataset_list_file=str(list_file)), self.backend
            )
            self.assertFalse((cdir / "ds999999_citations.json").exists())
            self.assertEqual(self.openalex.searched, [])

    def test_skips_dataset_with_no_valid_accession(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "experiment-1")  # fails the accession regex
            cli.run_find_mentions(_args(cdir), self.backend)
            data = json.loads((cdir / "experiment-1_citations.json").read_text())
            # Skipped before merge -> no marker written, no search made.
            self.assertNotIn("searched_accessions", data["metadata"])
            self.assertEqual(self.openalex.searched, [])

    def test_all_writes_fail_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            path = _seed(cdir, "ds002718")
            os.chmod(path, stat.S_IRUSR)  # read-only file -> open('w') raises
            try:
                with self.assertRaises(SystemExit) as ctx:
                    cli.run_find_mentions(_args(cdir), self.backend)
                self.assertEqual(ctx.exception.code, 2)
            finally:
                os.chmod(path, stat.S_IRWXU)

    def test_idempotent_second_run_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            path = _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir), self.backend)
            after_first = path.read_text()
            cli.run_find_mentions(_args(cdir), self.backend)
            self.assertEqual(path.read_text(), after_first)

    def test_real_mention_advances_date_last_updated_on_disk(self) -> None:
        """End-to-end regression guard for issue #229.

        A real content change must reach disk with a new `date_last_updated`.
        The default reply is an empty page, so no other test here writes a
        changed citation list through `run_find_mentions` ->
        `merge_accession_mentions` -> `write_citation_json_if_changed`.
        """
        self.openalex.reply = lambda term, _cursor: (
            _page([_work("W1", "10.1/found")]) if term == "ds002718" else EMPTY
        )
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            path = _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir), self.backend)
            data = json.loads(path.read_text())
            self.assertEqual(data["num_citations"], 1)
            found = data["citation_details"][0]
            self.assertEqual(found["doi"], "10.1/found")
            self.assertEqual(found["matched_accession"], "ds002718")
            self.assertNotEqual(data["date_last_updated"], "2026-06-18T00:00:00+00:00")
            stamped = datetime.fromisoformat(data["date_last_updated"])
            self.assertGreater(stamped, datetime(2026, 6, 18, tzinfo=UTC))

    def test_follows_the_cursor_across_pages(self) -> None:
        pages = {
            "*": _page([_work("W1", "10.1/one")], next_cursor="c2"),
            "c2": _page([_work("W2", "10.1/two")]),
        }
        self.openalex.reply = lambda _term, cursor: pages[cursor]
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            path = _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir), self.backend)
            data = json.loads(path.read_text())
            self.assertEqual(
                [c["doi"] for c in data["citation_details"]],
                ["10.1/one", "10.1/two"],
            )
            self.assertEqual(self.openalex.searched, ["ds002718", "ds002718"])


class FreshnessGateTests(TestCase):
    """Rolling `--max-age-days` / `--max-datasets` gating (issue #197).

    Real state files on disk; the freshness decision is real clock arithmetic,
    and the server's request log shows which datasets were actually searched.
    """

    def setUp(self) -> None:
        self.openalex = _OpenAlexServer(self)
        self.backend = self.openalex.backend

    def test_state_file_written_and_gitignored_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir), self.backend)
            self.assertTrue((cdir / ".mention_state.json").is_file())

    def test_fresh_dataset_is_skipped_on_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
            state = _state(cdir)
            self.assertIn("ds002718", state)

            # Second run inside the window must not re-search or re-stamp.
            cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
            self.assertEqual(self.openalex.searched, ["ds002718"])
            self.assertEqual(_state(cdir), state)

    def test_stale_dataset_is_reprocessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            stale = (datetime.now(UTC) - timedelta(days=30)).isoformat()
            (cdir / ".mention_state.json").write_text(json.dumps({"ds002718": stale}))
            cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
            self.assertEqual(self.openalex.searched, ["ds002718"])
            self.assertNotEqual(_state(cdir)["ds002718"], stale)

    def test_max_datasets_caps_the_run_and_rolls_over(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            for i in range(5):
                _seed(cdir, f"ds00271{i}")
            args = _args(cdir, max_age_days=7, max_datasets=2)
            cli.run_find_mentions(args, self.backend)
            self.assertEqual(len(_state(cdir)), 2)

            # The remaining three roll over to subsequent runs.
            cli.run_find_mentions(args, self.backend)
            self.assertEqual(len(_state(cdir)), 4)
            cli.run_find_mentions(args, self.backend)
            self.assertEqual(len(_state(cdir)), 5)
            # Each dataset was searched exactly once across the three runs.
            self.assertCountEqual(
                self.openalex.searched, [f"ds00271{i}" for i in range(5)]
            )

    def test_max_age_zero_disables_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            cli.run_find_mentions(_args(cdir, max_age_days=0), self.backend)
            first = _state(cdir)
            cli.run_find_mentions(_args(cdir, max_age_days=0), self.backend)
            # Gate off -> searched again -> stamp advances.
            self.assertEqual(self.openalex.searched, ["ds002718", "ds002718"])
            self.assertNotEqual(first["ds002718"], _state(cdir)["ds002718"])

    def test_state_persists_when_the_search_raises(self) -> None:
        """A failure escaping mid-loop must not discard the progress already
        made, or the next run redoes everything.
        """
        calls = itertools.count()
        self.openalex.reply = lambda _term, _cursor: (
            EMPTY if next(calls) == 0 else NOT_A_PAGE
        )
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            for i in range(3):
                _seed(cdir, f"ds00271{i}")
            with self.assertRaisesRegex(TypeError, "not a results page"):
                cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
            self.assertEqual(list(_state(cdir)), ["ds002710"])  # the one that succeeded

    def test_write_failure_does_not_stamp_and_retries_next_run(self) -> None:
        """A dataset whose write failed must NOT be recorded as checked.

        Regression guard for the silent-skip failure mode: if `stamp_checked`
        ever moved above the write's try/except, a transient write error would
        mark the dataset checked and drop it from citation coverage for a full
        --max-age-days window with nothing in the cron log to show for it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            path = _seed(cdir, "ds002718")
            _seed(cdir, "ds002719")
            os.chmod(path, stat.S_IRUSR)  # read-only -> open('w') raises
            try:
                cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
                state = _state(cdir)
                self.assertNotIn("ds002718", state)  # failed write -> unstamped
                self.assertIn("ds002719", state)  # its neighbor still recorded
            finally:
                os.chmod(path, stat.S_IRWXU)

            # Still stale, so the next run picks it up again.
            cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
            self.assertIn("ds002718", _state(cdir))

    def test_unreadable_citation_json_does_not_stamp(self) -> None:
        """A corrupt citation JSON must stay stale rather than be marked done."""
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            cdir.mkdir(parents=True)
            (cdir / "ds002718_citations.json").write_text("{not valid json")
            _seed(cdir, "ds002719")
            cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
            state = _state(cdir)
            self.assertNotIn("ds002718", state)
            self.assertIn("ds002719", state)

    def test_no_terms_dataset_is_stamped(self) -> None:
        """Locks in the deliberate choice at find_mentions.py: a dataset with no
        usable accession IS stamped, so it stops consuming a --max-datasets slot
        every night even though no request was made for it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "experiment-1")  # fails the accession regex
            cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
            state = _state(cdir)
            self.assertIn("experiment-1", state)

            # And is therefore filtered out as fresh on the next run.
            cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
            self.assertEqual(_state(cdir), state)
            self.assertEqual(self.openalex.searched, [])


class DegradedSearchTests(TestCase):
    """A rate-limited search must NOT be recorded as a completed one.

    `AccessionSearchBackend` logs and skips a failed term rather than raising
    (accession_search.py `_search_all`), so an empty result is ambiguous.
    Stamping on a degraded search would hide the dataset for a full
    --max-age-days window with no citation coverage and no error anywhere.
    The server answers 429: exactly what an exhausted OpenAlex budget produces
    once opencite's attempts are used up.
    """

    SEARCH_LOGGER = "dataset_citations.backends.accession_search"

    def setUp(self) -> None:
        self.openalex = _OpenAlexServer(self)
        self.openalex.reply = lambda _term, _cursor: RATE_LIMITED
        self.backend = self.openalex.backend

    def test_degraded_search_is_not_stamped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            with self.assertLogs(self.SEARCH_LOGGER, "WARNING") as logs:
                cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
            self.assertIn("accession search failed for ds002718", logs.output[0])
            self.assertEqual(self.openalex.searched, ["ds002718"])
            self.assertEqual(_state(cdir), {})

    def test_degraded_dataset_is_retried_next_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "ds002718")
            with self.assertLogs(self.SEARCH_LOGGER, "WARNING") as logs:
                cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
            self.assertIn("accession search failed for ds002718", logs.output[0])
            # OpenAlex recovers; the dataset is still stale so it gets searched.
            self.openalex.reply = lambda _term, _cursor: EMPTY
            cli.run_find_mentions(_args(cdir, max_age_days=7), self.backend)
            self.assertEqual(self.openalex.searched, ["ds002718", "ds002718"])
            self.assertIn("ds002718", _state(cdir))

    def test_partial_term_failure_also_withholds_the_stamp(self) -> None:
        # An on-* dataset is searched under its own id and its OpenNeuro
        # source id; only the second is rate limited.
        self.openalex.reply = lambda term, _cursor: (
            RATE_LIMITED if term == "ds005964" else EMPTY
        )
        catalog = [{"dataset_id": "on005964", "source_id": "ds005964"}]
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            _seed(cdir, "on005964")
            with self.assertLogs(self.SEARCH_LOGGER, "WARNING") as logs:
                cli.run_find_mentions(
                    _args(cdir, max_age_days=7, catalog=catalog), self.backend
                )
            self.assertEqual(len(logs.output), 1)
            self.assertIn("accession search failed for ds005964", logs.output[0])
            self.assertEqual(self.openalex.searched, ["on005964", "ds005964"])
            self.assertNotIn("on005964", _state(cdir))


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS") == "1",
    "live OpenAlex call; set RUN_INTEGRATION_TESTS=1 to enable",
)
class LiveFindMentionsTests(TestCase):
    """Not in any CI job; run by hand before changing the search request."""

    def test_live_search_is_merged_and_stamped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "json_opencite"
            path = _seed(cdir, "ds002718")
            cli.run_find_mentions(
                _args(cdir, max_age_days=7), AccessionSearchBackend(max_results=5)
            )
            data = json.loads(path.read_text())
            self.assertEqual(data["metadata"]["searched_accessions"], ["ds002718"])
            self.assertGreater(data["num_citations"], 0)
            self.assertIn("ds002718", _state(cdir))
