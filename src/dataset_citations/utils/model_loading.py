"""Load sentence-transformer checkpoints from the local cache first.

The nightly GPU pipeline on hallu runs unattended against two public,
non-gated checkpoints that are already in the host's HuggingFace hub cache. It
should therefore need no credential at all. It nevertheless went down on
2026-09-15 and 2026-09-16 because a stored HuggingFace OAuth token had expired:

    OAuth token has expired: "exp" claim timestamp check failed
    Repository Not Found for url: .../all-MiniLM-L6-v2/resolve/main/...

An expired token is worse than no token. Anonymous reads of a public repo
succeed, but a token the Hub rejects turns the same request into a 401, which
`sentence_transformers` surfaces as "not a valid model identifier" and the
scorer turns into a hard failure -- while a perfectly good copy sits in the
cache.

So: try the cache with the network disabled, and only reach for the Hub when
the checkpoint genuinely is not local yet. A bad or expired credential can no
longer take the pipeline down, and a warm cache also skips the revalidation
round trips entirely.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def _offline_env():
    """Force huggingface_hub offline for the duration of the block.

    Both variables are set: `HF_HUB_OFFLINE` is what current versions read,
    `TRANSFORMERS_OFFLINE` is honored by the transformers layer underneath
    sentence-transformers. Previous values are restored on exit so this never
    leaks into the rest of the process.
    """
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ.update(dict.fromkeys(keys, "1"))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_sentence_transformer(model_name: str, device: str) -> Any:
    """Return a SentenceTransformer, preferring the local hub cache.

    Raises the network-path exception if the model is neither cached nor
    downloadable, so a genuinely missing checkpoint still fails loudly.
    """
    from sentence_transformers import SentenceTransformer

    try:
        with _offline_env():
            model = SentenceTransformer(model_name, device=device)
        logger.info("Loaded %s from the local cache (offline)", model_name)
        return model
    except Exception as exc:
        # Broad by necessity: "not cached yet" is the expected case, but a
        # corrupted cache entry, a full disk, a permissions problem or a CUDA
        # OOM surface here too, and several of those will ALSO fail on the
        # network path with a misleading message ("not a valid model
        # identifier" when the cache write fails). Log the real message at
        # WARNING, not just the exception type, or the true cause is gone by
        # the time anyone reads the log.
        logger.warning(
            "%s not usable from cache (%s: %s); retrying against the Hub",
            model_name,
            type(exc).__name__,
            exc,
        )
        # Python unbinds the `except` name at block exit; keep it for the
        # combined error below.
        cache_exc: Exception = exc

    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as network_exc:
        # Surface BOTH causes: the cache failure is usually the real one.
        raise RuntimeError(
            f"{model_name}: could not load from cache ({cache_exc!r}) "
            f"nor from the Hub ({network_exc!r})"
        ) from network_exc
    logger.info("Loaded %s from the Hub", model_name)
    return model
