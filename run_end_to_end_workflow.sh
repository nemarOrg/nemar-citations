#!/bin/zsh
#
# Comprehensive end-to-end workflow for Dataset Citations project
# This script runs the complete pipeline: discover -> update -> score -> analyze.
# Dashboard rendering moved to the Astro app in web/ (#127 P8).
#
# Usage:
#   ./run_end_to_end_workflow.sh [mode]
#
# Modes:
#   test            - Run with test dataset (fast, no API calls)
#   local-ci-test   - Test the test workflow locally with act
#   local-ci-update - Test the update workflow locally with act
#   full            - Run full workflow with real API calls
#   help            - Show this help message
#
# Environment variables:
#   GITHUB_TOKEN              - Recommended for discovery and metadata
#   SEMANTIC_SCHOLAR_API_KEY  - Optional; raises opencite rate limits
#   OPENALEX_API_KEY          - Optional; raises opencite rate limits
#   PUBMED_API_KEY            - Optional; raises opencite rate limits
#

set -e # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
MODE="${1:-help}"
# Support backward compatibility
if [ "$MODE" = "local" ] || [ "$MODE" = "local-ci" ]; then
    print_warning "'$MODE' mode is deprecated. Use 'local-ci-test' or 'local-ci-update' instead."
    MODE="local-ci-update"
fi
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_FILE="${LOG_DIR}/workflow_${TIMESTAMP}.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to load secrets from .secrets file if available
load_secrets() {
    if [ -f ".secrets" ]; then
        print_info "Loading secrets from .secrets file..."
        # Source the .secrets file to load environment variables
        set -a  # Mark all new variables for export
        source .secrets
        set +a  # Turn off auto-export
    fi
}

# Function to check requirements
check_requirements() {
    print_status "Checking requirements..."

    # Load secrets if available and not already set
    load_secrets

    # Check Python environment (uv-managed)
    if ! command -v uv &> /dev/null; then
        print_error "uv not found; install from https://docs.astral.sh/uv/"
        return 1
    fi
    if [ ! -d ".venv" ]; then
        print_warning ".venv not found; bootstrapping with 'uv sync --all-extras'"
        uv sync --all-extras || return 1
    fi

    # Check environment variables for full mode
    if [ "$MODE" = "full" ]; then
        if [ -z "$GITHUB_TOKEN" ]; then
            print_warning "GITHUB_TOKEN not set. Discovery may be limited."
            print_info "Set it with: export GITHUB_TOKEN=your_token or add to .secrets file"
        fi
    fi

    # Check if act is installed for local-ci modes
    if [[ "$MODE" == local-ci-* ]]; then
        if ! command -v act &> /dev/null; then
            print_error "act not installed. Required for local-ci mode."
            print_info "Install with: brew install act"
            return 1
        fi
    fi

    print_status "Requirements check passed ✓"
    return 0
}

# Function to run test workflow
run_test_workflow() {
    print_status "Running test workflow with controlled dataset..."

    # Use existing test data
    TEST_DATA_DIR="tests/test_data/workflow_test"

    if [ ! -d "$TEST_DATA_DIR" ]; then
        print_error "Test data directory not found: $TEST_DATA_DIR"
        return 1
    fi

    # Create temporary output directories
    TEST_OUTPUT_DIR="test_output_${TIMESTAMP}"
    mkdir -p "$TEST_OUTPUT_DIR"/{citations,datasets,results,interactive_reports}

    print_status "Step 1/6: Using test dataset list..."
    cp "$TEST_DATA_DIR/discovered_datasets.txt" "$TEST_OUTPUT_DIR/"

    print_status "Step 2/6: Simulating citation update with test data..."
    cp "$TEST_DATA_DIR/previous_citations.csv" "$TEST_OUTPUT_DIR/citations/"

    # Run minimal citation update with test dataset
    print_status "Step 3/6: Processing test citations..."
    uv runpython -c "
import json
from pathlib import Path
test_citations = {
    'ds003555': {
        'dataset_id': 'ds003555',
        'dataset_name': 'Test Dataset',
        'citations': [
            {'title': f'Citation {i}', 'year': 2020+i, 'confidence_score': 0.5+i*0.1}
            for i in range(8)
        ]
    }
}
output_path = Path('$TEST_OUTPUT_DIR/citations/json_opencite')
output_path.mkdir(parents=True, exist_ok=True)
(output_path / 'ds003555.json').write_text(json.dumps(test_citations['ds003555'], indent=2))
print('Test citations generated')
"

    print_status "Step 4/6: Generating test metadata..."
    uv runpython -c "
import json
from pathlib import Path
test_metadata = {
    'dataset_id': 'ds003555',
    'name': 'Test Dataset',
    'description': 'A test dataset for workflow validation',
    'readme': 'This is a test README content'
}
output_path = Path('$TEST_OUTPUT_DIR/datasets')
output_path.mkdir(parents=True, exist_ok=True)
(output_path / 'ds003555_metadata.json').write_text(json.dumps(test_metadata, indent=2))
print('Test metadata generated')
"

    print_status "Step 5/6: Running analysis..."
    # Temporal analysis (positional argument for citations_dir)
    uv rundataset-citations-analyze-temporal \
        "$TEST_OUTPUT_DIR/citations/json_opencite" \
        --output-dir "$TEST_OUTPUT_DIR/results/temporal_analysis" || true

    # Network analysis (no citations_dir argument needed)
    uv rundataset-citations-analyze-networks \
        --output-dir "$TEST_OUTPUT_DIR/results/network_analysis" || true

    # Validate outputs
    print_status "Validating test outputs..."
    VALIDATION_PASSED=true

    if [ ! -f "$TEST_OUTPUT_DIR/citations/json_opencite/ds003555.json" ]; then
        print_error "Citation JSON not generated"
        VALIDATION_PASSED=false
    fi

    if [ ! -f "$TEST_OUTPUT_DIR/datasets/ds003555_metadata.json" ]; then
        print_error "Metadata JSON not generated"
        VALIDATION_PASSED=false
    fi

    if [ -d "$TEST_OUTPUT_DIR/dashboard_data" ]; then
        print_status "Dashboard generated successfully"
        ls -lh "$TEST_OUTPUT_DIR/interactive_reports/"*.html
    else
        print_warning "Dashboard not generated (non-critical)"
    fi

    if [ "$VALIDATION_PASSED" = true ]; then
        print_status "Test workflow completed successfully ✓"
        print_info "Test outputs in: $TEST_OUTPUT_DIR"
        return 0
    else
        print_error "Test workflow validation failed"
        return 1
    fi
}

# Function to run local CI/CD test workflow with act
run_local_ci_test_workflow() {
    print_status "Testing CI/CD test workflow locally with act..."
    print_info "Note: This tests the GitHub Actions test workflow locally"

    # Check for .secrets file
    if [ ! -f ".secrets" ]; then
        print_warning "No .secrets file found. Creating template..."
        cat > .secrets <<EOF
GITHUB_TOKEN=${GITHUB_TOKEN:-your_github_token}
EOF
        print_info "Edit .secrets file with your GitHub token if needed"
    fi

    print_status "Executing GitHub Actions test workflow locally..."
    act push -W .github/workflows/test.yml --secret-file .secrets --verbose 2>&1 | tee "$LOG_FILE"

    # Capture the exit status
    local exit_status=${PIPESTATUS[0]}

    if [ "$exit_status" -eq 0 ]; then
        print_status "CI/CD test workflow completed successfully ✓"
    else
        print_error "CI/CD test workflow failed. Check log: $LOG_FILE"
        return 1
    fi
}

# Function to run local CI/CD update workflow with act
run_local_ci_update_workflow() {
    print_status "Testing CI/CD update workflow locally with act..."
    print_info "Note: This tests the GitHub Actions update workflow locally"

    # Check for .secrets file
    if [ ! -f ".secrets" ]; then
        print_warning "No .secrets file found. Creating template..."
        cat > .secrets <<EOF
GITHUB_TOKEN=${GITHUB_TOKEN:-your_github_token}
# Optional, raise opencite rate limits:
# SEMANTIC_SCHOLAR_API_KEY=your_key
# OPENALEX_API_KEY=your_key
# PUBMED_API_KEY=your_key
EOF
        print_info "Edit .secrets file with your API keys"
    fi

    print_status "Executing GitHub Actions update workflow locally..."
    act workflow_dispatch -W .github/workflows/update_citations.yml --secret-file .secrets --verbose 2>&1 | tee "$LOG_FILE"

    # Capture the exit status
    local exit_status=${PIPESTATUS[0]}

    if [ "$exit_status" -eq 0 ]; then
        print_status "CI/CD update workflow completed successfully ✓"
        print_info "The workflow created a branch and PR automatically"
    else
        print_error "CI/CD update workflow failed. Check log: $LOG_FILE"
        return 1
    fi
}

# Function to run full workflow using git worktree (keeps main directory untouched)
run_full_workflow() {
    print_status "Running full end-to-end workflow..."

    # Save current directory - this will NOT be modified
    ORIGINAL_DIR=$(pwd)
    BRANCH_NAME="auto-update/$(date +'%Y-%m-%d_%H-%M')"
    PR_BASE_BRANCH="main"

    # Create worktree in temp directory
    WORKTREE_DIR=$(mktemp -d)
    print_status "Creating isolated worktree at: $WORKTREE_DIR"

    # Fetch latest from origin first
    print_info "Fetching latest changes from origin..."
    git fetch origin main --quiet

    # Create new branch and worktree from origin/main
    print_status "Creating branch: $BRANCH_NAME"
    git worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR" origin/main 2>&1 | tee -a "$LOG_FILE"

    # Change to worktree directory - all work happens here
    cd "$WORKTREE_DIR"
    print_info "Working in isolated directory: $(pwd)"

    # Configure git in worktree
    git config user.name "citations-bot"
    git config user.email "shirazi@ieee.org"

    # Step 1: Discovering datasets
    print_status "Step 1/8: Discovering datasets..."
    uv rundataset-citations-discover \
        --output-file "discovered_datasets.txt" 2>&1 | tee -a "$LOG_FILE"

    DATASET_COUNT=$(wc -l < "discovered_datasets.txt" | tr -d ' ')
    print_info "Found $DATASET_COUNT datasets"

    # Step 2: Updating citations
    print_status "Step 2/8: Updating citations (this may take a while)..."
    if [ ! -f "citations/previous_citations.csv" ]; then
        print_warning "No previous citations file found, creating empty one"
        mkdir -p citations
        echo "Dataset,Title,Year,Authors,DOI,URL" > citations/previous_citations.csv
    fi

    uv run dataset-citations-update \
        --dataset-list-file "discovered_datasets.txt" \
        --output-dir citations 2>&1 | tee -a "$LOG_FILE"

    # Step 3: Retrieving dataset metadata
    print_status "Step 3/8: Retrieving dataset metadata..."
    uv rundataset-citations-retrieve-metadata \
        --citations-dir citations/json_opencite \
        --output-dir datasets \
        --skip-existing \
        --log-level INFO 2>&1 | tee -a "$LOG_FILE"

    # Step 4: Calculating confidence scores
    print_status "Step 4/8: Calculating confidence scores..."
    uv rundataset-citations-score-confidence \
        --citations-dir citations/json_opencite \
        --datasets-dir datasets \
        --model Qwen/Qwen3-Embedding-0.6B \
        --skip-existing \
        --log-level INFO 2>&1 | tee -a "$LOG_FILE"

    # Step 5: Running analysis
    print_status "Step 5/8: Running analysis..."

    rm -rf dashboard_data/network dashboard_data/themes dashboard_data/temporal 2>/dev/null || true
    mkdir -p dashboard_data/{network,themes,temporal,visualizations}

    uv rundataset-citations-generate-embeddings \
        --citations citations/json_opencite \
        --datasets datasets \
        --embeddings-dir embeddings \
        --embedding-type both \
        --batch-size 32 \
        --verbose 2>&1 | tee -a "$LOG_FILE" || print_warning "Some embeddings may have failed"

    uv rundataset-citations-analyze-umap \
        --embeddings-dir embeddings \
        --output-dir dashboard_data \
        --embedding-type both \
        --n-components 2 \
        --clustering \
        --clustering-method kmeans \
        --n-clusters 4 \
        --create-visualizations \
        --verbose 2>&1 | tee -a "$LOG_FILE" || print_warning "UMAP analysis had issues"

    uv runpython -m dataset_citations.analysis.generate_themes \
        --citations-dir citations/json_opencite \
        --output-dir dashboard_data/themes 2>&1 | tee -a "$LOG_FILE" || print_warning "Theme generation had issues"

    uv runpython -m dataset_citations.analysis.generate_network \
        --citations-dir citations/json_opencite \
        --output-dir dashboard_data/network 2>&1 | tee -a "$LOG_FILE"

    uv runpython -m dataset_citations.analysis.generate_temporal \
        --citations-dir citations/json_opencite \
        --output-dir dashboard_data/temporal 2>&1 | tee -a "$LOG_FILE"

    # Step 7: Updating previous_citations.csv
    print_status "Step 7/8: Updating previous_citations.csv for next run..."
    TODAY_DATE=$(date +%d%m%Y)
    LATEST_CITATIONS_FILE="citations/citations_${TODAY_DATE}.csv"
    TARGET_PREVIOUS_FILE="citations/previous_citations.csv"

    if [ -f "$LATEST_CITATIONS_FILE" ]; then
        print_info "Found $LATEST_CITATIONS_FILE. Copying to $TARGET_PREVIOUS_FILE..."
        cp "$LATEST_CITATIONS_FILE" "$TARGET_PREVIOUS_FILE"
        print_status "Successfully updated $TARGET_PREVIOUS_FILE ✓"
    else
        print_warning "Today's citation file ($LATEST_CITATIONS_FILE) not found."
        print_info "previous_citations.csv not updated"
    fi

    # Step 8: Creating Pull Request
    print_status "Step 8/8: Creating Pull Request..."

    # Check for changes
    git add -A
    if git diff --cached --quiet; then
        print_warning "No changes detected. Nothing to commit."
        cd "$ORIGINAL_DIR"
        git worktree remove "$WORKTREE_DIR" --force 2>/dev/null || rm -rf "$WORKTREE_DIR"
        git branch -D "$BRANCH_NAME" 2>/dev/null || true
        return 0
    fi

    # Commit changes
    COMMIT_MSG="Update dataset citations - $(date +'%Y-%m-%d')

Workflow: Full end-to-end pipeline
Updated: $(git diff --cached --name-only | wc -l | tr -d ' ') files
"
    git commit -m "$COMMIT_MSG"
    print_status "Changes committed to branch: $BRANCH_NAME"

    # Push branch
    print_status "Pushing branch to origin..."
    if git push origin "$BRANCH_NAME"; then
        print_status "Branch pushed successfully"

        # Create PR using gh CLI if available
        if command -v gh &> /dev/null; then
            print_status "Creating Pull Request against $PR_BASE_BRANCH..."
            PR_TITLE="[Auto] Update citations - $(date +'%Y-%m-%d')"
            PR_BODY="## Automated Citation Update

This PR was generated by the end-to-end workflow script.

### Changes:
- Updated citation data from opencite (OpenAlex / Semantic Scholar / PubMed)
- Retrieved dataset metadata from GitHub
- Calculated confidence scores
- Generated analysis and dashboard

### Files Updated:
$(git diff origin/main..HEAD --name-only | head -20)

---
*Generated by run_end_to_end_workflow.sh*"

            if gh pr create \
                --base "$PR_BASE_BRANCH" \
                --head "$BRANCH_NAME" \
                --title "$PR_TITLE" \
                --body "$PR_BODY" \
                --label "automated,citations-update" \
                --assignee "@me"; then
                print_status "Pull Request created successfully ✓"
                PR_URL=$(gh pr view --json url -q .url 2>/dev/null || echo "")
                if [ -n "$PR_URL" ]; then
                    print_info "PR URL: $PR_URL"
                fi
            else
                print_warning "Failed to create PR. Create manually at:"
                print_info "https://github.com/sccn/nemar-citations/pull/new/$BRANCH_NAME"
            fi
        else
            print_warning "gh CLI not found. Create PR manually at:"
            print_info "https://github.com/sccn/nemar-citations/pull/new/$BRANCH_NAME"
        fi
    else
        print_error "Failed to push branch. You may need to push manually."
    fi

    # Save dashboard path for deployment
    DASHBOARD_PATH="$WORKTREE_DIR/interactive_reports"

    print_status "Full workflow completed ✓"
    print_info "Branch: $BRANCH_NAME"
    print_info "Log file: $LOG_FILE"

    # The GitHub Pages dashboard deploy was retired with the Python generator
    # (#127 P8). The dashboard is an Astro app in web/, deployed to Cloudflare
    # Pages by .github/workflows/deploy-dashboard.yml on push to main.


    # Return to original directory and clean up worktree
    cd "$ORIGINAL_DIR"
    print_status "Cleaning up worktree..."
    git worktree remove "$WORKTREE_DIR" --force 2>/dev/null || rm -rf "$WORKTREE_DIR"

    print_status "Workflow completed - original directory unchanged ✓"
}

# Function to show help
show_help() {
    cat <<EOF
Dataset Citations - End-to-End Workflow Runner

Usage: $0 [mode]

Modes:
  test      Run with test dataset (fast, no API calls)
            - Uses controlled test data
            - No external API calls
            - Validates pipeline components
            - ~30 seconds runtime

  local-ci-test   Test the test workflow locally with act
            - Runs the test workflow via Docker
            - Tests exact GitHub Actions test behavior
            - Useful for debugging test issues
            - Requires act and Docker
            - ~5-10 minutes runtime

  local-ci-update Test the update workflow locally with act
            - Runs the update workflow via Docker
            - Tests exact GitHub Actions update behavior
            - Useful for debugging update issues
            - Requires act and Docker
            - ~10-30 minutes runtime

  full      Run full pipeline directly (recommended)
            - Runs pipeline natively with uv
            - Creates feature branch automatically
            - Live opencite API calls (OpenAlex / Semantic Scholar / PubMed)
            - Creates PR for review
            - ~30-90 minutes runtime

  help    Show this help message

Environment Variables:
  GITHUB_TOKEN              GitHub personal access token (recommended for higher GitHub API limits)
  SEMANTIC_SCHOLAR_API_KEY  Optional; raises opencite rate limits
  OPENALEX_API_KEY          Optional; raises opencite rate limits
  PUBMED_API_KEY            Optional; raises opencite rate limits

Examples:
  # Quick test run
  $0 test

  # Test the test workflow locally
  export GITHUB_TOKEN=your_token
  $0 local-ci-test

  # Test the update workflow locally
  export GITHUB_TOKEN=your_token
  $0 local-ci-update

  # Full production run (recommended for actual updates)
  export GITHUB_TOKEN=your_token
  $0 full

Output:
  test:      test_output_<timestamp>/ (temporary directory)
  local-ci-test:   Tests run via Docker
  local-ci-update: Creates branch with PR (via GitHub Actions in Docker)
  full:      Updates repository on new branch, creates PR

Logs are saved to: logs/workflow_<timestamp>.log

EOF
}

# Main execution
main() {
    print_status "Dataset Citations - End-to-End Workflow"
    print_info "Mode: $MODE"
    print_info "Timestamp: $TIMESTAMP"

    case "$MODE" in
        test)
            check_requirements || exit 1
            run_test_workflow
            ;;
        local-ci-test)
            check_requirements || exit 1
            run_local_ci_test_workflow
            ;;
        local-ci-update)
            check_requirements || exit 1
            run_local_ci_update_workflow
            ;;
        full)
            check_requirements || exit 1
            run_full_workflow
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Invalid mode: $MODE"
            show_help
            exit 1
            ;;
    esac

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        print_status "Workflow completed successfully ✓"
    else
        print_error "Workflow failed with exit code: $EXIT_CODE"
    fi

    exit $EXIT_CODE
}

# Run main function
main