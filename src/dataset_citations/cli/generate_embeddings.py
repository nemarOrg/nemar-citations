"""
CLI command for generating embeddings from existing dataset and citation data.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ..core.citation_utils import load_citations_from_json
from ..embeddings.storage_manager import EmbeddingStorageManager
from ..quality.confidence_scoring import SentenceTransformerModel

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_dataset_metadata(dataset_id: str, datasets_dir: Path) -> str | None:
    """
    Load dataset metadata text for embedding generation.

    Args:
        dataset_id: BIDS dataset ID
        datasets_dir: Path to datasets directory

    Returns:
        Combined dataset metadata text or None if not found
    """
    dataset_file = datasets_dir / f"{dataset_id}_datasets.json"

    if not dataset_file.exists():
        logger.warning(f"Dataset metadata not found: {dataset_file}")
        return None

    try:
        with open(dataset_file) as f:
            dataset_data = json.load(f)

        # Combine relevant metadata fields. A key can be present with a null
        # value: 61 of 763 datasets in the 2026-09-17 run carry
        # "description": null, and appending that raised
        # "sequence item 0: expected str instance, NoneType found" out of the
        # join below, so those datasets silently got no embedding refresh.
        # A dataset with no text is the "no metadata" case, not an error.
        metadata_parts = []

        for field in ("description", "readme_content"):
            value = dataset_data.get(field)
            if value:
                metadata_parts.append(str(value))

        if "dataset_description" in dataset_data:
            desc = dataset_data["dataset_description"]
            if isinstance(desc, dict):
                metadata_parts.extend(
                    str(desc[field])
                    for field in ["Name", "Description", "TaskName", "Instructions"]
                    if field in desc
                )

        combined_text = " ".join(metadata_parts).strip()

        if not combined_text:
            logger.warning(f"No metadata text found for dataset {dataset_id}")
            return None

        logger.debug(
            f"Loaded {len(combined_text)} characters of metadata for {dataset_id}"
        )
        return combined_text

    except Exception as e:
        logger.error(f"Error loading dataset metadata for {dataset_id}: {e}")
        return None


def generate_dataset_embeddings(
    datasets_dir: Path,
    embeddings_dir: Path,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 10,
    force_regenerate: bool = False,
    device: str = "mps",
) -> int:
    """
    Generate embeddings for all datasets.

    Args:
        datasets_dir: Path to datasets directory
        embeddings_dir: Path to embeddings directory
        model_name: Sentence transformer model name
        batch_size: Number of datasets to process at once
        force_regenerate: Whether to regenerate existing embeddings
        device: Torch device for the sentence-transformer model

    Returns:
        Number of embeddings generated
    """
    logger.info("Starting dataset embedding generation...")

    # Initialize components
    storage_manager = EmbeddingStorageManager(embeddings_dir)
    model = SentenceTransformerModel(model_name=model_name, device=device)

    # Find all dataset files. NEMAR's catalog uses three prefixes: `ds-*`
    # (legacy OpenNeuro), `nm-*` (NEMAR-native), `on-*` (OpenNeuro imported
    # into NEMAR). A narrower `ds*` glob would silently skip every modern
    # dataset; match all three prefixes explicitly.
    dataset_files = sorted(
        f
        for pattern in ("ds*_datasets.json", "nm*_datasets.json", "on*_datasets.json")
        for f in datasets_dir.glob(pattern)
    )
    total_datasets = len(dataset_files)

    if total_datasets == 0:
        logger.error(f"No dataset files found in {datasets_dir}")
        return 0

    logger.info(f"Found {total_datasets} dataset files")

    generated_count = 0
    skipped_count = 0

    # Process datasets in batches
    for i in range(0, total_datasets, batch_size):
        batch_files = dataset_files[i : i + batch_size]

        logger.info(
            f"Processing batch {i // batch_size + 1}/{(total_datasets + batch_size - 1) // batch_size}"
        )

        for dataset_file in batch_files:
            dataset_id = dataset_file.stem.replace("_datasets", "")

            # Check if embedding already exists
            if not force_regenerate:
                existing = storage_manager.registry.get_current_dataset_embedding(
                    dataset_id
                )
                if existing:
                    logger.debug(f"Skipping {dataset_id} - embedding already exists")
                    skipped_count += 1
                    continue

            # Load dataset metadata
            metadata_text = load_dataset_metadata(dataset_id, datasets_dir)
            if not metadata_text:
                logger.warning(f"Skipping {dataset_id} - no metadata available")
                continue

            try:
                # Generate embedding
                logger.info(f"Generating embedding for {dataset_id}")
                embedding = model.encode([metadata_text])[0]

                # Store embedding
                storage_manager.store_dataset_embedding(
                    dataset_id=dataset_id,
                    embedding=embedding,
                    content_sources={"combined_metadata": metadata_text},
                    model=model_name,
                    metadata={
                        "text_length": len(metadata_text),
                        "embedding_dim": len(embedding),
                    },
                )

                generated_count += 1
                logger.info(f"Generated embedding for {dataset_id}")

            except Exception as e:
                logger.error(f"Error generating embedding for {dataset_id}: {e}")
                continue

    logger.info(
        f"Dataset embedding generation complete: {generated_count} generated, {skipped_count} skipped"
    )
    return generated_count


def _resolve_citation_files(citations_dir: Path) -> tuple[Path, list[Path]]:
    """Locate per-dataset citation JSON files under ``citations_dir``.

    The opencite pipeline writes flat to `<citations_dir>` (e.g.
    `citations/json_opencite/<id>_citations.json`), which is the sole
    citation source since epic #180 phase 5 removed the legacy
    `<citations_dir>/json/` layout.

    Matches all three NEMAR catalog prefixes (`ds-`, `nm-`, `on-`) so the
    full ~594-dataset corpus is covered, not just legacy `ds-*` entries.
    """

    def _glob(root: Path) -> list[Path]:
        return sorted(
            f
            for pattern in (
                "ds*_citations.json",
                "nm*_citations.json",
                "on*_citations.json",
            )
            for f in root.glob(pattern)
        )

    return citations_dir, _glob(citations_dir)


def generate_citation_embeddings(
    citations_dir: Path,
    embeddings_dir: Path,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 50,
    force_regenerate: bool = False,
    min_confidence: float = 0.4,
    device: str = "mps",
) -> int:
    """
    Generate embeddings for all high-confidence citations.

    Args:
        citations_dir: Path to citations directory
        embeddings_dir: Path to embeddings directory
        model_name: Sentence transformer model name
        batch_size: Number of citations to process at once
        force_regenerate: Whether to regenerate existing embeddings
        min_confidence: Minimum confidence score for citations
        device: Torch device for the sentence-transformer model

    Returns:
        Number of embeddings generated
    """
    logger.info("Starting citation embedding generation...")

    # Initialize components
    storage_manager = EmbeddingStorageManager(embeddings_dir)
    model = SentenceTransformerModel(model_name=model_name, device=device)

    # Find all citation files under the flat citations/json_opencite/ layout.
    search_dir, citation_files = _resolve_citation_files(citations_dir)
    total_files = len(citation_files)

    if total_files == 0:
        logger.error(f"No citation files found in {search_dir}")
        return 0

    logger.info(f"Found {total_files} citation files")

    generated_count = 0
    skipped_count = 0
    batch_texts = []
    batch_metadata = []

    # Process all citation files
    for file_idx, citation_file in enumerate(citation_files):
        dataset_id = citation_file.stem.replace("_citations", "")

        try:
            # Load citations
            citations_data = load_citations_from_json(citation_file)

            if "citation_details" not in citations_data:
                logger.warning(f"No citation details in {citation_file}")
                continue

            # Process each citation
            for citation in citations_data["citation_details"]:
                # Check confidence score - handle nested structure
                confidence_data = citation.get("confidence_scoring", {})
                confidence = confidence_data.get("confidence_score", 0.0)

                # Skip citations below confidence threshold
                if confidence < min_confidence:
                    continue

                # Prepare citation text
                title = citation.get("title", "")
                abstract = citation.get("abstract", "")
                citation_text = f"{title} {abstract}".strip()

                if not citation_text:
                    continue

                # Generate citation hash for uniqueness
                import hashlib

                citation_hash = hashlib.sha256(citation_text.encode()).hexdigest()[:8]

                # Check if embedding already exists
                if not force_regenerate:
                    existing = storage_manager.registry.get_current_citation_embedding(
                        citation_hash
                    )
                    if existing:
                        skipped_count += 1
                        continue

                # Add to batch
                batch_texts.append(citation_text)
                batch_metadata.append(
                    {
                        "citation_hash": citation_hash,
                        "title": title,
                        "dataset_id": dataset_id,
                        "confidence_score": confidence,
                        "text_length": len(citation_text),
                    }
                )

                # Process batch when full
                if len(batch_texts) >= batch_size:
                    generated_count += process_citation_batch(
                        batch_texts, batch_metadata, model, storage_manager, model_name
                    )
                    batch_texts = []
                    batch_metadata = []

        except Exception as e:
            logger.error(f"Error processing citation file {citation_file}: {e}")
            continue

        # Progress logging
        if (file_idx + 1) % 10 == 0:
            logger.info(f"Processed {file_idx + 1}/{total_files} citation files")

    # Process remaining batch
    if batch_texts:
        generated_count += process_citation_batch(
            batch_texts, batch_metadata, model, storage_manager, model_name
        )

    logger.info(
        f"Citation embedding generation complete: {generated_count} generated, {skipped_count} skipped"
    )
    return generated_count


def process_citation_batch(
    batch_texts, batch_metadata, model, storage_manager, model_name
):
    """Process a batch of citations for embedding generation."""
    try:
        # Generate embeddings for batch
        logger.debug(f"Generating embeddings for batch of {len(batch_texts)} citations")
        embeddings = model.encode(batch_texts)

        # Store each embedding
        batch_generated = 0
        for i, (embedding, metadata) in enumerate(
            zip(embeddings, batch_metadata, strict=False)
        ):
            try:
                storage_manager.store_citation_embedding(
                    citation_text=batch_texts[i],
                    title=metadata["title"],
                    embedding=embedding,
                    text_sources={"title_abstract": batch_texts[i]},
                    model=model_name,
                    metadata={
                        "dataset_id": metadata["dataset_id"],
                        "confidence_score": metadata["confidence_score"],
                        "text_length": metadata["text_length"],
                        "embedding_dim": len(embedding),
                    },
                )
                batch_generated += 1

            except Exception as e:
                logger.error(
                    f"Error storing embedding for citation {metadata['citation_hash']}: {e}"
                )

        logger.info(
            f"Generated {batch_generated}/{len(batch_texts)} embeddings in batch"
        )
        return batch_generated

    except Exception as e:
        logger.error(f"Error processing citation batch: {e}")
        return 0


def _resolve_device(spec: str) -> str:
    """Resolve a device spec to the actual torch device string.

    `auto` picks the best available device on the host so the same
    invocation works on hallu (NVIDIA → cuda) and Mac workstations
    (Apple Silicon → mps) without flag changes. Mirrors the resolution
    helper in `quality/confidence_scoring.py` — keep them in sync if
    either changes. Non-`auto` values are returned verbatim so an
    operator can still force a specific backend.
    """
    if spec != "auto":
        return spec
    try:
        import torch

        # Prefer the host's accelerator. cuda is checked first because hallu
        # (the production cron host) is Linux+NVIDIA; on a Mac workstation
        # cuda is unavailable so we fall through to mps; cpu is the last
        # resort.
        if torch.cuda.is_available():
            logger.info("device=auto -> cuda detected")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("device=auto -> mps detected")
            return "mps"
    except ImportError:
        pass
    logger.info("device=auto -> falling back to cpu")
    return "cpu"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for BIDS dataset citations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all embeddings
  dataset-citations-generate-embeddings --datasets datasets/ --citations citations/

  # Generate only dataset embeddings
  dataset-citations-generate-embeddings --datasets datasets/ --embedding-type datasets

  # Force regenerate with custom model
  dataset-citations-generate-embeddings --datasets datasets/ --citations citations/ \\
    --force-regenerate --model Qwen/Qwen3-Embedding-0.6B

  # Generate with higher confidence threshold
  dataset-citations-generate-embeddings --citations citations/ \\
    --min-confidence 0.6 --embedding-type citations
        """,
    )

    parser.add_argument(
        "--datasets",
        type=Path,
        default="datasets",
        help="Path to datasets directory (default: datasets)",
    )

    parser.add_argument(
        "--citations",
        type=Path,
        default="citations",
        help="Path to citations directory (default: citations)",
    )

    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default="embeddings",
        help="Path to embeddings storage directory (default: embeddings)",
    )

    parser.add_argument(
        "--embedding-type",
        choices=["datasets", "citations", "both"],
        default="both",
        help="Type of embeddings to generate (default: both)",
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Sentence transformer model name (default: Qwen/Qwen3-Embedding-0.6B)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.4,
        help="Minimum confidence score for citations (default: 0.4)",
    )

    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of existing embeddings",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Alias for the default behavior (the embedding registry already "
            "skips datasets/citations with a matching content hash). Accepted "
            "for invocation parity with score-confidence + update; sole "
            "operational effect is to override --force-regenerate when both "
            "are passed."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help=(
            "Torch device for sentence transformers. 'auto' (default) picks "
            "cuda if available, then mps, then cpu — so the same invocation "
            "works on hallu (NVIDIA) and Mac workstations (Apple Metal) "
            "without explicit flags."
        ),
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    # Validate inputs
    if args.embedding_type in ["datasets", "both"] and not args.datasets.exists():
        logger.error(f"Datasets directory not found: {args.datasets}")
        return 1

    if args.embedding_type in ["citations", "both"] and not args.citations.exists():
        logger.error(f"Citations directory not found: {args.citations}")
        return 1

    # Create embeddings directory
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)

    # `--skip-existing` is the explicit form of the registry-skip default.
    # If a caller passes both `--skip-existing` and `--force-regenerate`,
    # the former wins (matches the score-confidence convention where
    # skip-existing short-circuits before any regeneration work).
    force_regenerate = args.force_regenerate and not args.skip_existing

    # Resolve `auto` once so both embedding passes use the same device.
    resolved_device = _resolve_device(args.device)

    # Track total generated
    total_generated = 0
    start_time = time.time()

    try:
        # Generate dataset embeddings
        if args.embedding_type in ["datasets", "both"]:
            logger.info("=" * 60)
            logger.info("GENERATING DATASET EMBEDDINGS")
            logger.info("=" * 60)

            dataset_count = generate_dataset_embeddings(
                datasets_dir=args.datasets,
                embeddings_dir=args.embeddings_dir,
                model_name=args.model,
                batch_size=args.batch_size // 5,  # Smaller batch for datasets
                force_regenerate=force_regenerate,
                device=resolved_device,
            )
            total_generated += dataset_count

        # Generate citation embeddings
        if args.embedding_type in ["citations", "both"]:
            logger.info("=" * 60)
            logger.info("GENERATING CITATION EMBEDDINGS")
            logger.info("=" * 60)

            citation_count = generate_citation_embeddings(
                citations_dir=args.citations,
                embeddings_dir=args.embeddings_dir,
                model_name=args.model,
                batch_size=args.batch_size,
                force_regenerate=force_regenerate,
                min_confidence=args.min_confidence,
                device=resolved_device,
            )
            total_generated += citation_count

        # Final summary
        elapsed_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total embeddings generated: {total_generated}")
        logger.info(f"Total time: {elapsed_time:.1f} seconds")

        # Show storage stats
        storage_manager = EmbeddingStorageManager(args.embeddings_dir)
        stats = storage_manager.get_storage_stats()
        logger.info(
            f"Storage stats: {stats['total_files']} files, {stats['total_file_size_mb']:.1f} MB"
        )

        return 0

    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
