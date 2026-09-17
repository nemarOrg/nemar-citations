"""Dataset metadata assembly for the embeddings step.

No mocks: `load_dataset_metadata` reads a JSON file off disk, so the tests
write real files in a tmp_path and assert on the real return value. The null
shapes below are copied from live files on hallu (on002712, on005289,
on006110), where a present-but-null "description" made the join raise
TypeError and cost 61 of 763 datasets their embedding refresh.
"""

import json
import logging

from dataset_citations.cli.generate_embeddings import load_dataset_metadata


def _write(tmp_path, dataset_id, payload):
    path = tmp_path / f"{dataset_id}_datasets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


class TestLoadDatasetMetadata:
    def test_all_null_fields_are_no_metadata_not_an_error(self, tmp_path, caplog):
        """The live on002712 shape: every text field present but null.

        The return value alone does not pin this down; the broad `except`
        swallowed the TypeError and returned None too. What distinguishes a
        handled "this dataset has no text" from a crash is the log level, so
        assert on that: no ERROR record may be emitted.
        """
        datasets_dir = _write(
            tmp_path,
            "on002712",
            {
                "description": None,
                "readme_content": None,
                "dataset_description": None,
            },
        )
        with caplog.at_level(logging.DEBUG):
            assert load_dataset_metadata("on002712", datasets_dir) is None
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, f"expected no ERROR, got: {[r.getMessage() for r in errors]}"

    def test_null_description_does_not_discard_a_usable_readme(self, tmp_path):
        """A null in one field must not lose the text in another."""
        datasets_dir = _write(
            tmp_path,
            "on000001",
            {"description": None, "readme_content": "Resting state EEG, 32 channels."},
        )
        assert (
            load_dataset_metadata("on000001", datasets_dir)
            == "Resting state EEG, 32 channels."
        )

    def test_combines_description_readme_and_dataset_description(self, tmp_path):
        datasets_dir = _write(
            tmp_path,
            "on000002",
            {
                "description": "Auditory oddball.",
                "readme_content": "Twenty subjects.",
                "dataset_description": {"Name": "Oddball", "TaskName": "oddball"},
            },
        )
        result = load_dataset_metadata("on000002", datasets_dir)
        assert result is not None
        for fragment in ("Auditory oddball.", "Twenty subjects.", "Oddball", "oddball"):
            assert fragment in result

    def test_missing_file_returns_none(self, tmp_path):
        assert load_dataset_metadata("on999999", tmp_path) is None

    def test_empty_strings_are_treated_as_no_metadata(self, tmp_path):
        datasets_dir = _write(
            tmp_path, "on000003", {"description": "", "readme_content": ""}
        )
        assert load_dataset_metadata("on000003", datasets_dir) is None
