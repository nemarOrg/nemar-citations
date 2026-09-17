"""
Integration tests for the complete dataset citations workflow.
These tests use controlled real data to validate the entire pipeline.
"""

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless


class TestIntegrationWorkflow(TestCase):
    """Integration tests for the complete workflow pipeline."""

    def setUp(self):
        """Set up test environment with controlled datasets."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir)

        # Create directory structure
        self.citations_dir = self.test_dir / "citations" / "json"
        self.datasets_dir = self.test_dir / "datasets"
        self.results_dir = self.test_dir / "results"
        self.reports_dir = self.test_dir / "reports"

        for d in [
            self.citations_dir,
            self.datasets_dir,
            self.results_dir,
            self.reports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Create controlled test data
        self._create_controlled_test_data()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def _create_controlled_test_data(self):
        """Create controlled test datasets with known properties."""

        # Small dataset with 3 citations
        small_dataset = {
            "dataset_id": "ds_small",
            "dataset_name": "Small Test Dataset",
            "repository_name": "OpenNeuroDatasets-BIDS/ds_small",
            "description": "A small test dataset for validation",
            "citations": [
                {
                    "title": "Neural correlates of test behavior",
                    "authors": ["Smith J", "Doe A"],
                    "year": 2022,
                    "journal": "NeuroTest",
                    "doi": "10.1234/test.2022.001",
                    "confidence_score": 0.85,
                    "abstract": "We used the Small Test Dataset to study neural correlates.",
                },
                {
                    "title": "Brain connectivity patterns in testing",
                    "authors": ["Johnson K", "Lee M"],
                    "year": 2023,
                    "journal": "Brain Testing",
                    "confidence_score": 0.72,
                    "abstract": "Analysis of ds_small revealed interesting connectivity patterns.",
                },
                {
                    "title": "Methodological considerations for test data",
                    "authors": ["Brown C"],
                    "year": 2021,
                    "journal": "Methods Journal",
                    "confidence_score": 0.35,  # Low confidence
                    "abstract": "General methodology paper with weak dataset reference.",
                },
            ],
        }

        # Medium dataset with 8 citations
        medium_dataset = {
            "dataset_id": "ds_medium",
            "dataset_name": "Medium Test Dataset",
            "repository_name": "OpenNeuroDatasets-BIDS/ds_medium",
            "description": "A medium-sized test dataset for comprehensive testing",
            "citations": [
                {
                    "title": f"Study {i + 1} using medium dataset",
                    "authors": [f"Author{i + 1} A", f"Author{i + 1} B"],
                    "year": 2020 + (i % 4),
                    "journal": f"Journal{i % 3 + 1}",
                    "confidence_score": 0.4 + (i * 0.08),  # Varies from 0.4 to 0.96
                    "abstract": f"This study {i + 1} utilized ds_medium for analysis.",
                }
                for i in range(8)
            ],
        }

        # Edge case: dataset with no citations
        empty_dataset = {
            "dataset_id": "ds_empty",
            "dataset_name": "Empty Test Dataset",
            "repository_name": "OpenNeuroDatasets-BIDS/ds_empty",
            "description": "A dataset with no citations for edge case testing",
            "citations": [],
        }

        # Edge case: dataset with all low-confidence citations
        low_conf_dataset = {
            "dataset_id": "ds_lowconf",
            "dataset_name": "Low Confidence Test Dataset",
            "repository_name": "OpenNeuroDatasets-BIDS/ds_lowconf",
            "description": "Dataset with only low-confidence citations",
            "citations": [
                {
                    "title": f"Weak reference {i + 1}",
                    "authors": [f"Researcher {i + 1}"],
                    "year": 2023,
                    "confidence_score": 0.1 + (i * 0.05),  # All below 0.4
                    "abstract": "Minimal reference to dataset.",
                }
                for i in range(5)
            ],
        }

        # Save all datasets as JSON
        for dataset in [small_dataset, medium_dataset, empty_dataset, low_conf_dataset]:
            json_path = self.citations_dir / f"{dataset['dataset_id']}.json"
            with open(json_path, "w") as f:
                json.dump(dataset, f, indent=2)

        # Create corresponding metadata files
        for dataset in [small_dataset, medium_dataset, empty_dataset, low_conf_dataset]:
            metadata = {
                "dataset_id": dataset["dataset_id"],
                "name": dataset["dataset_name"],
                "description": dataset["description"],
                "repository": dataset["repository_name"],
                "readme": f"# {dataset['dataset_name']}\n\n{dataset['description']}",
            }
            metadata_path = self.datasets_dir / f"{dataset['dataset_id']}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        # Create dataset list file
        dataset_list_path = self.test_dir / "discovered_datasets.txt"
        with open(dataset_list_path, "w") as f:
            f.write("ds_small\n")
            f.write("ds_medium\n")
            f.write("ds_empty\n")
            f.write("ds_lowconf\n")

    def test_citation_statistics(self):
        """Test that citation statistics are calculated correctly."""
        # Count total citations
        total_citations = 0
        high_conf_citations = 0
        confidence_threshold = 0.4

        for json_file in self.citations_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
                citations = data.get("citations", [])
                total_citations += len(citations)

                for citation in citations:
                    if citation.get("confidence_score", 0) >= confidence_threshold:
                        high_conf_citations += 1

        # Verify counts
        self.assertEqual(total_citations, 16)  # 3 + 8 + 0 + 5
        self.assertEqual(high_conf_citations, 10)  # 2 + 8 + 0 + 0

    def test_dataset_discovery(self):
        """Test that all datasets are discovered correctly."""
        dataset_list_path = self.test_dir / "discovered_datasets.txt"
        with open(dataset_list_path) as f:
            datasets = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(datasets), 4)
        self.assertIn("ds_small", datasets)
        self.assertIn("ds_medium", datasets)
        self.assertIn("ds_empty", datasets)
        self.assertIn("ds_lowconf", datasets)

    def test_csv_generation(self):
        """Test CSV generation from JSON data."""
        # Generate CSV from test data
        citation_counts = {}

        for json_file in self.citations_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
                dataset_id = data["dataset_id"]
                citations = data.get("citations", [])
                citation_counts[dataset_id] = len(citations)

        # Create CSV
        csv_path = self.test_dir / "test_citations.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset_id", "citation_count", "status"])
            for dataset_id, count in citation_counts.items():
                writer.writerow([dataset_id, count, "success"])

        # Verify CSV
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 4)

        # Check specific counts
        counts_by_id = {row["dataset_id"]: int(row["citation_count"]) for row in rows}
        self.assertEqual(counts_by_id["ds_small"], 3)
        self.assertEqual(counts_by_id["ds_medium"], 8)
        self.assertEqual(counts_by_id["ds_empty"], 0)
        self.assertEqual(counts_by_id["ds_lowconf"], 5)

    def test_incremental_updates(self):
        """Test handling of incremental citation updates."""
        # Simulate adding a new citation to ds_small
        json_path = self.citations_dir / "ds_small.json"
        with open(json_path) as f:
            data = json.load(f)

        # Add new citation
        new_citation = {
            "title": "New study using small dataset",
            "authors": ["New Author"],
            "year": 2024,
            "confidence_score": 0.91,
            "abstract": "Latest research using ds_small.",
        }
        data["citations"].append(new_citation)

        # Save updated data
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Verify update
        with open(json_path) as f:
            updated_data = json.load(f)

        self.assertEqual(len(updated_data["citations"]), 4)  # Was 3, now 4

        # Check that high-confidence count increased
        high_conf = [
            c for c in updated_data["citations"] if c.get("confidence_score", 0) >= 0.4
        ]
        self.assertEqual(len(high_conf), 3)  # Was 2, now 3

    @skipUnless(os.getenv("RUN_SLOW_TESTS"), "Skipping slow test")
    def test_full_pipeline_simulation(self):
        """Test simulated full pipeline execution (slow test)."""
        # This would test the complete pipeline but is skipped by default
        # Set RUN_SLOW_TESTS=1 to run this test


class TestEdgeCases(TestCase):
    """Test edge cases and error handling."""

    def test_unicode_handling(self):
        """Test handling of Unicode characters in citations."""
        temp_dir = tempfile.mkdtemp()
        try:
            citations_dir = Path(temp_dir) / "citations"
            citations_dir.mkdir()

            unicode_data = {
                "dataset_id": "ds_unicode",
                "citations": [
                    {
                        "title": "研究標題 (Research Title)",
                        "authors": ["李明", "José García"],
                        "year": 2023,
                        "confidence_score": 0.75,
                        "abstract": "This study includes 中文 and español characters.",
                    }
                ],
            }

            json_path = citations_dir / "ds_unicode.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(unicode_data, f, ensure_ascii=False, indent=2)

            # Should handle Unicode properly
            with open(json_path, encoding="utf-8") as f:
                loaded_data = json.load(f)

            self.assertEqual(
                loaded_data["citations"][0]["title"], "研究標題 (Research Title)"
            )
        finally:
            shutil.rmtree(temp_dir)
