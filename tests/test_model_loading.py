"""Cache-first checkpoint loading (issue #217).

No mocks: the env-var contract is observable directly, and the loader is
exercised against the real `sentence_transformers` only when a checkpoint is
actually cached on the machine running the tests.
"""

import os

import pytest

from dataset_citations.utils.model_loading import (
    _offline_env,
    load_sentence_transformer,
)


class TestOfflineEnv:
    def test_sets_both_offline_flags_inside_the_block(self):
        with _offline_env():
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    def test_restores_absent_vars_on_exit(self):
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ.pop(key, None)
        with _offline_env():
            pass
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ

    def test_restores_preexisting_value_on_exit(self):
        os.environ["HF_HUB_OFFLINE"] = "0"
        try:
            with _offline_env():
                assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["HF_HUB_OFFLINE"] == "0"
        finally:
            os.environ.pop("HF_HUB_OFFLINE", None)

    def test_restores_even_when_the_block_raises(self):
        os.environ.pop("HF_HUB_OFFLINE", None)
        with pytest.raises(ValueError), _offline_env():
            raise ValueError("boom")
        assert "HF_HUB_OFFLINE" not in os.environ


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

    def test_loads_a_cached_checkpoint_without_network(self):
        """When the checkpoint is cached, an invalid token must not matter.

        Skipped unless the model is already in this machine's hub cache, which
        is exactly the condition the fix targets on hallu.
        """
        pytest.importorskip("sentence_transformers")
        from huggingface_hub import constants, try_to_load_from_cache

        name = "sentence-transformers/all-MiniLM-L6-v2"
        if (
            try_to_load_from_cache(
                name, "modules.json", cache_dir=constants.HF_HUB_CACHE
            )
            is None
        ):
            pytest.skip(f"{name} is not in the local hub cache")

        previous = os.environ.get("HF_TOKEN")
        # Not a credential: a deliberately malformed value the Hub rejects,
        # reproducing the expired-token 401 that broke the nightly run.
        os.environ["HF_TOKEN"] = "hf_" + ("invalid" * 4)
        try:
            model = load_sentence_transformer(name, "cpu")
            assert model.get_sentence_embedding_dimension() == 384
        finally:
            if previous is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = previous
