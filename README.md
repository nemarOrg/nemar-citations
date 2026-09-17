# NEMAR Citations

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Tests](https://github.com/nemarOrg/nemar-citations/actions/workflows/test.yml/badge.svg)](https://github.com/nemarOrg/nemar-citations/actions)

Automated BIDS dataset citation tracking system with AI-powered confidence scoring for 300+ neuroscience datasets.

## Overview

Track and analyze citations for OpenNeuro and NEMAR datasets with a complete pipeline from discovery to interactive dashboards. Citations are discovered via the [opencite](https://github.com/neuromechanist/opencite) library, anchored on DOIs surfaced by each dataset's `.nemar/metadata.json` (nm-datasets) or `dataset_description.json` (legacy ds-datasets). Features semantic similarity scoring, network analysis, and automated updates via GitHub Actions.

**Key Features**: Dataset discovery • Citation tracking • AI confidence scoring • Network analysis • Interactive dashboards • JSON/CSV export • GitHub Actions automation

## Installation

```bash
git clone https://github.com/nemarOrg/nemar-citations.git
cd nemar-citations
uv sync --all-extras
```

**Requirements**: Python 3.13+ • GitHub token (recommended for higher API limits) • Optional Semantic Scholar / OpenAlex / PubMed API keys for opencite (higher rate limits)

## Quick Start

```bash
# 1. Setup environment (choose .env or .secrets)
# Option A: Using .env file
echo "GITHUB_TOKEN=your_token_here" > .env
# Optional opencite keys for higher API rate limits:
# echo "SEMANTIC_SCHOLAR_API_KEY=your_key" >> .env
# echo "OPENALEX_API_KEY=your_key" >> .env

# Option B: Using .secrets file (auto-loaded by workflow script)
echo "GITHUB_TOKEN=your_token_here" > .secrets

# 2. Run complete pipeline
chmod +x run_end_to_end_workflow.sh
./run_end_to_end_workflow.sh test              # Test mode (no API calls)
./run_end_to_end_workflow.sh full              # Full pipeline (recommended)
./run_end_to_end_workflow.sh local-ci-test     # Test CI/CD test workflow locally
./run_end_to_end_workflow.sh local-ci-update   # Test CI/CD update workflow locally
```

## Shell Scripts

The repository includes several shell scripts for different workflows:

| Script | Purpose | Runtime | When to Use |
|--------|---------|---------|-------------|
| `run_end_to_end_workflow.sh` | Complete pipeline from discovery to dashboard | 1-3 hours | Production updates, full analysis |
| `run_full_analysis.sh` | Analysis and dashboard generation only | 10-30 min | When citations already exist |
| `migrate_to_json.sh` | Convert pickle files to JSON format | 1-2 min | One-time migration |

## Pipeline Workflow

### Running the Complete Pipeline

The `run_end_to_end_workflow.sh` script automates the entire workflow:

| Mode | Description | Runtime | API Calls | Steps Executed | Branch/PR |
|------|-------------|---------|-----------|----------------|-----------|
| `test` | Controlled test data (3-8 citations) | ~1 min | None | 4-5 only (Analyze, Generate) | No |
| `full` | **Recommended**: Direct pipeline execution | 30-90 min | opencite (OpenAlex / Semantic Scholar / PubMed), GitHub | 1-5 (All steps) | Yes (auto) |
| `local-ci-test` | Test GitHub Actions test workflow via Docker | ~5-10 min | None | Runs test suite | No |
| `local-ci-update` | Test GitHub Actions update workflow via Docker | ~10-30 min | Real API calls | 1-5 (All steps) | Yes (auto) |

**Workflow Steps**:
1. **Discover** → Find BIDS datasets (EEG/MEG/iEEG)
2. **Collect** → Fetch citing works for each dataset's reference DOIs via opencite
3. **Enhance** → Add metadata & AI confidence scores
4. **Analyze** → Network, temporal, theme analysis
5. **Generate** → Interactive HTML dashboard

**Mode Selection Guide**:
- Use `test` for quick validation during development
- Use `full` for actual citation updates (runs natively, faster)
- Use `local-ci-test` to test/debug GitHub Actions test workflow issues
- Use `local-ci-update` to test/debug GitHub Actions update workflow issues

**Branch Protection**: Both `full` and `local-ci-update` modes automatically create a feature branch and pull request to protect the main branch from direct commits.

### Automated Updates (Cron)

```bash
# 1. Create update script
cat > ~/update_citations.sh << 'EOF'
#!/bin/bash
cd /path/to/dataset_citations
uv run ./run_end_to_end_workflow.sh full
EOF
chmod +x ~/update_citations.sh

# 2. Add to crontab (choose one)
crontab -e
0 2 1 * * ~/update_citations.sh >> ~/citations.log 2>&1  # Monthly
0 3 * * 0 ~/update_citations.sh >> ~/citations.log 2>&1  # Weekly
0 4 * * * ~/update_citations.sh >> ~/citations.log 2>&1  # Daily

# 3. Monitor
tail -f ~/citations.log
```

## Python API

```python
from dataset_citations.core import citation_utils
from dataset_citations.quality.confidence_scoring import CitationConfidenceScorer
from dataset_citations.quality.dataset_metadata import DatasetMetadataRetriever

# Convert pickle to JSON
json_path = citation_utils.migrate_pickle_to_json(
    'citations/pickle/ds002718.pkl', 
    'citations/json', 
    'ds002718'
)

# Load citation data
citations = citation_utils.load_citation_json(json_path)
print(f"Dataset {citations['dataset_id']} has {citations['num_citations']} citations")

# Calculate confidence scores
scorer = CitationConfidenceScorer()
confidence_scores = scorer.score_citations_for_dataset('ds002718', citations, dataset_metadata)

# Retrieve dataset metadata
retriever = DatasetMetadataRetriever()
metadata = retriever.get_dataset_metadata('ds002718')
```


## Key Commands

```bash
# Discovery & Updates
uv run dataset-citations-discover                    # Find datasets
uv run dataset-citations-update                      # Fetch citations
uv run dataset-citations-migrate                     # Pickle->JSON

# Quality & Analysis
uv run dataset-citations-retrieve-metadata           # Get GitHub data
uv run dataset-citations-score-confidence            # AI scoring
uv run dataset-citations-analyze-temporal            # Trends
uv run dataset-citations-analyze-networks            # Networks

# Dashboard (Astro app in web/; built and deployed by CI)
cd web && bun install && bun run build              # Build locally

# All commands support --help for detailed usage
```

## Data Formats

### JSON Output
```json
{
  "dataset_id": "ds002718",
  "num_citations": 13,
  "citation_details": [{
    "title": "Paper title",
    "author": "Authors",
    "year": 2021,
    "confidence_score": 0.82  // AI similarity score
  }]
}
```

### Confidence Scoring
AI-powered relevance scoring (0.0-1.0) using sentence transformers to compare dataset metadata with citation abstracts. Helps filter high-confidence citations and identify misattributions.

## Development

```bash
# Setup (UV installs Python and dependencies)
git clone https://github.com/nemarOrg/nemar-citations.git
cd nemar-citations
uv sync --all-extras

# Testing
uv run pytest tests/ -v                # Fast tests
uv run pytest --cov=dataset_citations  # With coverage

# Code quality
uv run ruff format src/ tests/         # Format
uv run ruff check --fix src/ tests/    # Lint
uv run ty check src/                   # Type check
```

## Architecture

**Core Components**:
- **Discovery**: Find BIDS datasets via GitHub API
- **Collection**: opencite (OpenAlex / Semantic Scholar / PubMed) citation graph traversal, DOI-anchored
- **Processing**: Parallel processing, format conversion, validation
- **Analysis**: Network graphs, temporal trends, theme clustering
- **Dashboard**: Interactive HTML with D3.js visualizations

**Data Flow**: Discovery → Fetching → Processing → Analysis → Dashboard


## Troubleshooting

| Issue | Solution |
|-------|----------|
| opencite rate limit | Add `SEMANTIC_SCHOLAR_API_KEY` / `OPENALEX_API_KEY` to `.env` for higher limits |
| GitHub API rate limit | Add `GITHUB_TOKEN` to `.env` |
| MPS memory error (macOS) | Use `--device cpu` |
| Import errors | Reinstall: `uv sync --reinstall --all-extras` |

**Debug**: Add `--verbose` flag to any command

**Support**: [GitHub Issues](https://github.com/nemarOrg/nemar-citations/issues)

## Contributing

1. Fork & create feature branch
2. Make changes with tests
3. Run `uv run pytest`, `uv run ruff format`, and `uv run ty check`
4. Submit PR with issue reference

**Guidelines**: Type hints • Docstrings • Tests • No mocks

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) - Attribution, NonCommercial, ShareAlike

## Citation

If you use this software in your research, please cite:

```bibtex
@software{shirazi2025nemarcitations,
  title={NEMAR Citations: Automated BIDS Dataset Citation Tracking System},
  author={Shirazi, Seyed Yahya},
  year={2025},
  url={https://github.com/nemarOrg/nemar-citations},
  organization={Swartz Center for Computational Neuroscience (SCCN)}
}
```

## Acknowledgments

- **Author**: [Seyed Yahya Shirazi](https://github.com/neuromechanist)
- **Organization**: [Swartz Center for Computational Neuroscience (SCCN)](https://sccn.ucsd.edu/)
- **Project**: [NEMAR - NeuroElectroMagnetic Archive](https://nemar.org/)
- **GitHub**: [@neuromechanist](https://github.com/neuromechanist)

Built for NEMAR and the neuroscience open science community.
