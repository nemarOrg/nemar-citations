"""CLI: cross-dataset citation attribution audit.

Surfaces the inflation caused by shared umbrella / methods anchors leaking
their citers onto every dataset that lists them. Run report-only in CI to keep
the true number visible; pass ``--fail-on-violation`` to use it as a deploy
guard. Now that the fail-closed anchor gate (#241) keeps such anchors out, a
violation means a gate regression.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..analysis.attribution_audit import run_audit

logger = logging.getLogger(__name__)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Audit cross-dataset citation attribution (umbrella-anchor leakage)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--citations-dir",
        type=Path,
        default=Path("citations/json_opencite"),
        help="Directory containing schema-v2 citation JSON files",
    )
    parser.add_argument(
        "--max-datasets-per-anchor",
        type=int,
        default=5,
        help="An anchor attributed across more datasets than this is a violation",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many anchors to list in the console / markdown report",
    )
    parser.add_argument(
        "--report-json", type=Path, help="Write the full report as JSON to this path"
    )
    parser.add_argument(
        "--report-md", type=Path, help="Write a markdown report to this path"
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit non-zero if any anchor exceeds the spread threshold",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    report = run_audit(
        citations_dir=args.citations_dir,
        max_datasets_per_anchor=args.max_datasets_per_anchor,
        top=args.top,
        report_json=args.report_json,
        report_md=args.report_md,
    )

    if args.fail_on_violation and report.violations:
        logger.error(
            "%d anchor(s) exceed the spread threshold; failing.",
            len(report.violations),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
