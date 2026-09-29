"""The hallu cron closes the nightly PRs that tonight's PR supersedes.

Every nightly run starts from `git reset --hard origin/main`, so the newest
nightly PR carries the whole pipeline state; an older one left open (failed CI,
or now conflicting) must not merge later on top of it. These tests run the
script's own `superseded_nightly_prs` function, extracted from
scripts/hallu_cron_pipeline.sh, in real bash on `gh pr list` style lines.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

CRON_SCRIPT = Path(__file__).parent.parent / "scripts" / "hallu_cron_pipeline.sh"
TONIGHT = "auto-update/2026-09-30T10-00-01Z"


def _function() -> str:
    text = CRON_SCRIPT.read_text()
    match = re.search(
        r"^superseded_nightly_prs\(\) \{\n.*?^\}\n", text, re.MULTILINE | re.DOTALL
    )
    assert match, "superseded_nightly_prs missing from hallu_cron_pipeline.sh"
    return match.group(0)


def _superseded(current: str, listing: str) -> list[str]:
    result = subprocess.run(
        ["bash", "-c", _function() + 'superseded_nightly_prs "$1"', "_", current],
        input=listing,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.split()


def test_only_older_nightly_prs_are_superseded() -> None:
    listing = "\n".join(
        [
            "240 auto-update/2026-09-29T10-00-01Z",
            "248 fix/246-offline-find-mentions-tests",
            "239 auto-update/2026-09-28T10-00-01Z",
            f"250 {TONIGHT}",
            "251 auto-update/2026-10-01T10-00-01Z",  # newer: left alone
            "252 chore/auto-update/2026-01-01",  # not a nightly branch
            "234 auto-update/2026-09-23T10-00-01Z",  # last line, no newline
        ]
    )
    assert _superseded(TONIGHT, listing) == ["240", "239", "234"]


def test_no_open_prs_closes_nothing() -> None:
    assert _superseded(TONIGHT, "") == []
    assert _superseded(TONIGHT, "\n") == []


def test_nothing_is_closed_without_tonights_pr() -> None:
    """A failed `gh pr create` must stop the run before the close step, or the
    older PRs would be closed with nothing to replace them.
    """
    text = CRON_SCRIPT.read_text()
    guard = text.index('if [[ -z "$PR_URL" ]]; then')
    assert text.index("exit 1", guard) < text.index("gh pr close")
    assert guard < text.index("gh pr merge --auto")
