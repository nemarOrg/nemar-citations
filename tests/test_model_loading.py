"""Cache-first checkpoint loading (issue #217).

No mocks. The property under test, "a cached checkpoint loads without talking
to the Hub", is directly observable: `huggingface_hub` logs every outbound
request through `httpx` / `urllib3`, so a log handler counts them for real.

The first version of this fix set `HF_HUB_OFFLINE=1` around the load and was
asserted only through the environment variable. Those assertions passed while
the loader still made 38 Hub round trips per call, because `huggingface_hub`
reads the variable once at import time. Counting requests tests the behavior
rather than the mechanism, so it cannot pass the same way.
"""

import logging
import os

import pytest

from dataset_citations.utils.model_loading import load_sentence_transformer

_CACHED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class _HubRequestCounter(logging.Handler):
    """Count log records that mention an outbound huggingface.co request."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "huggingface.co" in record.getMessage():
            self.count += 1


def _skip_unless_cached(name: str) -> None:
    pytest.importorskip("sentence_transformers")
    from huggingface_hub import constants, try_to_load_from_cache

    if (
        try_to_load_from_cache(name, "modules.json", cache_dir=constants.HF_HUB_CACHE)
        is None
    ):
        pytest.skip(f"{name} is not in the local hub cache")


class TestLoadSentenceTransformer:
    @pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        reason="reaches huggingface.co; set RUN_INTEGRATION_TESTS=1 to enable",
    )
    def test_missing_checkpoint_still_raises(self):
        """A genuinely absent model must fail loudly, not return None.

        Gated: an uncached repo id falls through to the Hub, so this makes a
        real outbound call. Every other live-network test in this suite is
        gated the same way to keep the default run offline.
        """
        pytest.importorskip("sentence_transformers")
        with pytest.raises((OSError, ValueError, RuntimeError)):
            load_sentence_transformer(
                "nemar-citations/definitely-not-a-real-model", "cpu"
            )

    def test_cached_checkpoint_loads_with_no_hub_requests(self):
        """The regression guard: a warm cache must cost zero Hub round trips.

        This is what the nightly run on hallu depends on. Skipped where the
        checkpoint is not cached, which is the condition the fix targets.
        """
        _skip_unless_cached(_CACHED_MODEL)

        counter = _HubRequestCounter()
        watched = [
            logging.getLogger(name)
            for name in ("httpx", "urllib3.connectionpool", "requests")
        ]
        previous_levels = [(lg, lg.level) for lg in watched]
        for lg in watched:
            lg.setLevel(logging.INFO)
            lg.addHandler(counter)
        try:
            model = load_sentence_transformer(_CACHED_MODEL, "cpu")
        finally:
            for lg in watched:
                lg.removeHandler(counter)
            for lg, level in previous_levels:
                lg.setLevel(level)

        assert model is not None
        assert counter.count == 0, (
            f"cached load made {counter.count} huggingface.co request(s); "
            "the offline path is not engaging"
        )

    def test_loads_a_cached_checkpoint_despite_a_rejected_token(self):
        """When the checkpoint is cached, a bad credential must not matter.

        Reproduces the 2026-09-15 / 2026-09-16 outage shape: the Hub rejects
        the stored token, and the load has to come off disk anyway.
        """
        _skip_unless_cached(_CACHED_MODEL)

        previous = os.environ.get("HF_TOKEN")
        # Not a credential: a deliberately malformed value the Hub rejects,
        # reproducing the expired-token 401 that broke the nightly run.
        os.environ["HF_TOKEN"] = "hf_" + ("invalid" * 4)
        try:
            model = load_sentence_transformer(_CACHED_MODEL, "cpu")
            assert model.get_embedding_dimension() == 384
        finally:
            if previous is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = previous
