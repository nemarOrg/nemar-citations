"""
Citation confidence scoring using sentence transformers.

This module calculates confidence scores for citations by comparing
dataset metadata with citation content using semantic similarity.

Copyright (c) 2025 Seyed Yahya Shirazi (neuromechanist)
All rights reserved.

Author: Seyed Yahya Shirazi
GitHub: https://github.com/neuromechanist
Email: shirazi@ieee.org
"""

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from dataset_citations.utils.model_loading import load_sentence_transformer

from .dataset_metadata import extract_dataset_text, load_dataset_metadata

logger = logging.getLogger(__name__)


class CitationConfidenceScorer:
    """Calculate confidence scores for citations using semantic similarity."""

    def __init__(
        self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "mps"
    ):
        """
        Initialize the confidence scorer.

        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to use ('mps', 'auto', 'cpu', 'cuda', or specific device)
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.model = None
        self._load_model()

    def _determine_device(self, device: str) -> str:
        """
        Determine the best device to use for sentence transformers.

        Args:
            device: Device preference ('auto', 'cpu', 'cuda', 'mps', or specific device)

        Returns:
            Device string to use
        """
        if device != "auto":
            logger.info(f"Using user-specified device: {device}")
            return device

        try:
            # Lazy import to avoid import errors during CLI help
            import torch

            # Check for MPS (Metal Performance Shaders on macOS)
            if torch.backends.mps.is_available():
                logger.info("GPU acceleration available: Using MPS (Metal)")
                return "mps"
            # Check for CUDA
            elif torch.cuda.is_available():
                logger.info("GPU acceleration available: Using CUDA")
                return "cuda"
            else:
                logger.info("GPU acceleration not available, using CPU")
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not available, defaulting to CPU")
            return "cpu"

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            # Lazy import to avoid import errors during CLI help
            import torch

            logger.info(f"Loading sentence transformer model: {self.model_name}")
            logger.info(f"Using device: {self.device}")

            # Initialize model with device. Cache-first so an expired or wrong
            # HuggingFace credential cannot 401 a public, already-downloaded
            # checkpoint out from under the nightly run (issue #217).
            self.model = load_sentence_transformer(self.model_name, self.device)

            # Explicitly move model to device if it's not already there
            if hasattr(torch, "device"):
                device_obj = torch.device(self.device)
                self.model.to(device_obj)

            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a smaller, more commonly available model
            logger.info("Falling back to all-MiniLM-L6-v2 model")
            try:
                import torch

                self.model = load_sentence_transformer("all-MiniLM-L6-v2", self.device)

                # Explicitly move model to device
                if hasattr(torch, "device"):
                    device_obj = torch.device(self.device)
                    self.model.to(device_obj)

                logger.info(f"Fallback model loaded successfully on {self.device}")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise RuntimeError(
                    "Could not load any sentence transformer model"
                ) from fallback_error

    def extract_citation_text(self, citation: dict[str, Any]) -> str:
        """
        Extract combined text content from citation for similarity scoring.

        Args:
            citation: Citation dict from citation_details

        Returns:
            Combined text string for embedding
        """
        text_parts = []

        # Add citation title (most important)
        if citation.get("title"):
            text_parts.append(f"Title: {citation['title']}")

        # Add abstract if available (very important for context)
        if citation.get("abstract"):
            abstract_text = citation["abstract"][:1500]  # Limit abstract length
            text_parts.append(f"Abstract: {abstract_text}")

        # Add venue/journal for context
        if citation.get("venue"):
            text_parts.append(f"Venue: {citation['venue']}")
        elif citation.get("journal"):
            text_parts.append(f"Journal: {citation['journal']}")

        # Add author information
        if citation.get("author"):
            text_parts.append(f"Authors: {citation['author']}")

        return "\n\n".join(text_parts)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text strings.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0

        try:
            # Encode both texts with device specification
            embeddings = self.model.encode([text1, text2], device=self.device)

            # Calculate cosine similarity
            similarity_matrix = self.model.similarity(embeddings, embeddings)
            similarity_score = similarity_matrix[0][1].item()

            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, similarity_score))

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def score_citation(
        self, citation: dict[str, Any], dataset_text: str
    ) -> dict[str, Any]:
        """
        Calculate confidence score for a single citation.

        Args:
            citation: Citation dict from citation_details
            dataset_text: Combined dataset metadata text

        Returns:
            Dict with confidence score and components
        """
        citation_text = self.extract_citation_text(citation)

        if not citation_text or not dataset_text:
            return {
                "confidence_score": 0.0,
                "similarity_score": 0.0,
                "citation_text_length": len(citation_text) if citation_text else 0,
                "dataset_text_length": len(dataset_text) if dataset_text else 0,
                "scoring_method": "sentence_transformers",
                "model_used": self.model_name,
            }

        # Calculate base similarity score
        similarity_score = self.calculate_similarity(citation_text, dataset_text)

        # Apply basic confidence adjustments
        confidence_score = self._adjust_confidence_score(
            similarity_score, citation, citation_text
        )

        return {
            "confidence_score": confidence_score,
            "similarity_score": similarity_score,
            "citation_text_length": len(citation_text),
            "dataset_text_length": len(dataset_text),
            "scoring_method": "sentence_transformers",
            "model_used": self.model_name,
        }

    def _adjust_confidence_score(
        self, similarity_score: float, citation: dict[str, Any], citation_text: str
    ) -> float:
        """
        Apply adjustments to the base similarity score.

        Args:
            similarity_score: Base similarity score
            citation: Citation metadata
            citation_text: Combined citation text

        Returns:
            Adjusted confidence score
        """
        confidence = similarity_score

        # Boost confidence if citation has abstract (more context available)
        if citation.get("abstract") and len(citation["abstract"]) > 100:
            confidence *= 1.1

        # Slight penalty if citation has very high citation count (might be generic)
        cited_by = citation.get("cited_by", 0)
        if cited_by > 1000:
            confidence *= 0.95

        # Boost confidence for recent citations (2020+)
        year = citation.get("year", 0)
        if year >= 2020:
            confidence *= 1.05

        # Ensure score stays within valid range
        return max(0.0, min(1.0, confidence))

    def score_all_citations(
        self, citations_data: dict[str, Any], dataset_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate confidence scores for all citations in a dataset.

        Args:
            citations_data: Citation data dict (from JSON file)
            dataset_metadata: Dataset metadata dict

        Returns:
            Updated citations data with confidence scores
        """
        logger.info(
            f"Scoring citations for dataset: {citations_data.get('dataset_id', 'unknown')}"
        )

        # Extract dataset text for comparison
        dataset_text = extract_dataset_text(dataset_metadata)

        if not dataset_text:
            logger.warning("No dataset text available for scoring")
            return citations_data

        # Create a copy to avoid modifying original data
        scored_data = citations_data.copy()

        # Add confidence scoring metadata
        scored_data["confidence_scoring"] = {
            "model_used": self.model_name,
            "scoring_date": pd.Timestamp.now(tz="UTC").isoformat(),
            "dataset_text_length": len(dataset_text),
            "num_citations_scored": len(citations_data.get("citation_details", [])),
        }

        # Score each citation
        citation_details = citations_data.get("citation_details", [])
        scored_citations = []

        for i, citation in enumerate(citation_details):
            try:
                confidence_info = self.score_citation(citation, dataset_text)

                # Add confidence info to citation
                citation_with_confidence = citation.copy()
                citation_with_confidence["confidence_scoring"] = confidence_info

                scored_citations.append(citation_with_confidence)

                if (i + 1) % 10 == 0:
                    logger.info(f"Scored {i + 1}/{len(citation_details)} citations")

            except Exception as e:
                logger.error(f"Error scoring citation {i}: {e}")
                # Keep original citation without confidence score
                scored_citations.append(citation)

        scored_data["citation_details"] = scored_citations

        # Add summary statistics
        confidence_scores = [
            c.get("confidence_scoring", {}).get("confidence_score", 0.0)
            for c in scored_citations
        ]

        if confidence_scores:
            scored_data["confidence_scoring"]["summary_stats"] = {
                "mean_confidence": np.mean(confidence_scores),
                "median_confidence": np.median(confidence_scores),
                "std_confidence": np.std(confidence_scores),
                "min_confidence": np.min(confidence_scores),
                "max_confidence": np.max(confidence_scores),
                "high_confidence_count": sum(
                    1 for score in confidence_scores if score >= 0.7
                ),
                "medium_confidence_count": sum(
                    1 for score in confidence_scores if 0.4 <= score < 0.7
                ),
                "low_confidence_count": sum(
                    1 for score in confidence_scores if score < 0.4
                ),
            }

        logger.info(
            f"Completed confidence scoring for {len(scored_citations)} citations"
        )
        return scored_data


def score_dataset_citations(
    citations_file: str,
    dataset_metadata_file: str,
    output_file: str | None = None,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str = "mps",
) -> str:
    """
    Score citations for a single dataset and save results.

    Args:
        citations_file: Path to citation JSON file
        dataset_metadata_file: Path to dataset metadata JSON file
        output_file: Output file path (if None, overwrites citations_file)
        model_name: Sentence transformer model to use
        device: Device to use ('mps', 'auto', 'cpu', 'cuda', or specific device)

    Returns:
        Path to output file
    """
    # Load data
    with open(citations_file, encoding="utf-8") as f:
        citations_data = json.load(f)

    dataset_metadata = load_dataset_metadata(dataset_metadata_file)

    # Score citations
    scorer = CitationConfidenceScorer(model_name, device)
    scored_data = scorer.score_all_citations(citations_data, dataset_metadata)

    # Save results
    output_path = output_file or citations_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scored_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved confidence-scored citations to {output_path}")
    return output_path


def batch_score_citations(
    citations_dir: str,
    datasets_dir: str,
    output_dir: str | None = None,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str = "mps",
) -> list[str]:
    """
    Score citations for multiple datasets in batch.

    Args:
        citations_dir: Directory containing citation JSON files
        datasets_dir: Directory containing dataset metadata JSON files
        output_dir: Output directory (if None, overwrites original files)
        model_name: Sentence transformer model to use
        device: Device to use ('mps', 'auto', 'cpu', 'cuda', or specific device)

    Returns:
        List of output file paths
    """
    output_files = []

    # Find matching citation and dataset files
    citation_files = [
        f for f in os.listdir(citations_dir) if f.endswith("_citations.json")
    ]

    for citation_file in citation_files:
        dataset_id = citation_file.replace("_citations.json", "")
        dataset_file = f"{dataset_id}_datasets.json"

        citations_path = os.path.join(citations_dir, citation_file)
        dataset_path = os.path.join(datasets_dir, dataset_file)

        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset metadata not found for {dataset_id}, skipping")
            continue

        try:
            # Determine output path
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, citation_file)
            else:
                output_path = citations_path

            # Score citations
            output_file = score_dataset_citations(
                citations_path, dataset_path, output_path, model_name, device
            )
            output_files.append(output_file)

        except Exception as e:
            logger.error(f"Error scoring citations for {dataset_id}: {e}")

    logger.info(f"Completed batch scoring for {len(output_files)} datasets")
    return output_files


class SentenceTransformerModel:
    """Simple wrapper for SentenceTransformer for CLI usage."""

    def __init__(
        self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "auto"
    ):
        """
        Initialize the sentence transformer model.

        Args:
            model_name: Name of the model to use
            device: Device to use for computation
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.model = None
        self._load_model()

    def _determine_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            try:
                import torch

                if torch.backends.mps.is_available():
                    return "mps"
                elif torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading model: {self.model_name} on {self.device}")
            self.model = load_sentence_transformer(self.model_name, self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a commonly available model
            try:
                logger.info("Falling back to all-MiniLM-L6-v2 model")
                self.model = load_sentence_transformer("all-MiniLM-L6-v2", self.device)
                logger.info("Fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise RuntimeError(
                    "Could not load any sentence transformer model"
                ) from fallback_error

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            Array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            embeddings = self.model.encode(
                texts, convert_to_tensor=False, show_progress_bar=False
            )
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
