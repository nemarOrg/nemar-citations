"""`OpenCiteBackend.get_paper` against a local HTTP server (#241 review).

opencite's `OpenAlexClient.lookup_doi` swallows every error, so an OpenAlex
outage used to come back as `not_found`, which the anchor judge treats as a
permanent miss and retries only monthly. These tests run the real opencite
client and the real backend over real HTTP against a local server that answers
like OpenAlex: a recorded work for one DOI, a 404 for an unknown DOI, a 503 for
an outage. Only the base URL differs from production.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import TestCase

from opencite.clients.openalex import OpenAlexClient
from opencite.config import Config

from dataset_citations.backends import OpenCiteBackend
from dataset_citations.sources.models import FetchError, FetchSuccess

_WORK = (
    Path(__file__).parent / "test_data" / "openalex_work_s41597-019-0027-4.json"
).read_bytes()


class _OpenAlexLike(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/works/doi:10.1038/s41597-019-0027-4":
            status, body = 200, _WORK
        elif path.startswith("/works/doi:10.9999/down"):
            status, body = 503, b'{"error": "unavailable"}'
        else:
            status, body = 404, b'{"error": "Not Found"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the test output quiet."""


class GetPaperErrorTests(TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAlexLike)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.backend = self._backend(f"http://127.0.0.1:{self.server.server_port}")

    def _backend(self, base_url: str) -> OpenCiteBackend:
        class LocalOpenAlex(OpenAlexClient):
            def __init__(self, config: Config) -> None:
                super().__init__(config)
                self.base_url = base_url
                self.max_retries = 1

        class LocalBackend(OpenCiteBackend):
            _openalex_client_cls = LocalOpenAlex

        return LocalBackend(Config.from_env())

    def test_a_known_doi_resolves(self) -> None:
        result = self.backend.get_paper("10.1038/s41597-019-0027-4")
        assert isinstance(result, FetchSuccess)
        self.assertEqual(
            result.value.title,
            "Multi-channel EEG recordings during a sustained-attention driving task",
        )

    def test_only_a_404_is_not_found(self) -> None:
        result = self.backend.get_paper("10.38119/openneuro.ds002721")
        self.assertIsInstance(result, FetchError)
        assert isinstance(result, FetchError)
        self.assertEqual(result.reason, "not_found")

    def test_a_server_outage_is_transient(self) -> None:
        with self.assertLogs("opencite", "WARNING"):
            result = self.backend.get_paper("10.9999/down.1")
        assert isinstance(result, FetchError)
        self.assertEqual(result.reason, "network")

    def test_an_unreachable_host_is_transient(self) -> None:
        # A port nothing listens on: the connection is refused.
        probe = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAlexLike)
        port = probe.server_port
        probe.server_close()
        backend = self._backend(f"http://127.0.0.1:{port}")
        with self.assertLogs("opencite", "WARNING"):
            result = backend.get_paper("10.1038/s41597-019-0027-4")
        assert isinstance(result, FetchError)
        self.assertEqual(result.reason, "network")
