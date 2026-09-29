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
TONIGHT = 250  # tonight's PR number


def _function() -> str:
    text = CRON_SCRIPT.read_text()
    match = re.search(
        r"^superseded_nightly_prs\(\) \{\n.*?^\}\n", text, re.MULTILINE | re.DOTALL
    )
    assert match, "superseded_nightly_prs missing from hallu_cron_pipeline.sh"
    return match.group(0)


def _code() -> str:
    """The script with comment lines removed, for the ordering checks."""
    lines = CRON_SCRIPT.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _superseded(tonight: int, listing: str) -> list[str]:
    result = subprocess.run(
        ["bash", "-c", _function() + 'superseded_nightly_prs "$1"', "_", str(tonight)],
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
            f"{TONIGHT} auto-update/2026-09-30T10-00-01Z",
            "251 auto-update/2026-10-01T10-00-01Z",  # opened after tonight's
            "252 chore/auto-update/2026-01-01",  # not a nightly branch
            "245 auto-update/manual",  # hand-named: never closed
            "246 auto-update/20260930T100001Z",  # not the cron's form
            "234 auto-update/2026-09-23T10-00-01Z",  # last line, no newline
        ]
    )
    assert _superseded(TONIGHT, listing) == ["240", "239", "234"]


def test_no_open_prs_closes_nothing() -> None:
    assert _superseded(TONIGHT, "") == []
    assert _superseded(TONIGHT, "\n") == []


def test_nothing_is_closed_without_tonights_pr() -> None:
    """A failed push or `gh pr create` must stop the run before the close step,
    or the older PRs would be closed with nothing to replace them.
    """
    code = _code()
    push = code.index('if ! git push -u --quiet origin "$BRANCH"; then')
    create = code.index("if ! PR_URL=$(gh pr create")
    assert push < code.index("exit 2", push) < create
    assert create < code.index("exit 2", create) < code.index("gh pr close")
    assert create < code.index("gh pr merge --auto")


def test_publish_errors_fail_the_run_after_cleanup() -> None:
    """A failed auto-merge, PR listing, or close leaves tonight's PR open; the
    run still finishes the cleanup, then exits 2 instead of reporting success.
    """
    code = _code()
    tail = code[code.index("PUBLISH_OK=1") :]
    assert tail.count("PUBLISH_OK=0") == 3
    final_check = tail.index('if [[ "$PUBLISH_OK" != 1 ]]; then')
    assert tail.index("gh pr close") < final_check
    assert tail.index("exit 2", final_check) < tail.index("hallu-cron $TS done")
