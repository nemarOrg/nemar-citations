"""The cron script must run the code it just checked out, not the old copy.

Background: bash streams a script from an open file descriptor as it executes.
`git reset --hard` replaces a file by writing a temp file and renaming it over
the original, which gives the path a new inode while the descriptor bash is
reading still refers to the old one. A script that resets its own working tree
therefore runs the PREVIOUS revision of itself to completion, even though every
subprocess it spawns sees the new tree.

On 2026-09-17 that cost the nightly run: commit 2f2fc985 renamed a CLI flag in
`cli/judge_anchors.py` and in `scripts/hallu_cron_pipeline.sh` atomically, and
the 03:00 run still paired the old script with the new CLI and aborted with
"the following arguments are required: --dataset-list-file".

No mocks: these tests build a real git repository in tmp_path, run real bash
against it, and assert on what the process actually printed. The first test
demonstrates the hazard, the second demonstrates that the re-exec guard fixes
it, and the third checks the production script carries that guard.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

CRON_SCRIPT = Path(__file__).parent.parent / "scripts" / "hallu_cron_pipeline.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs real git and bash",
)

_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
set -uo pipefail
REPO_DIR="$1"
cd "$REPO_DIR"
echo "START version={version}"
{reset_block}
echo "END version={version}"
"""

# Stage 1 resets the tree, then hands over to whatever it just checked out.
_REEXEC_BLOCK = """if [ -z "${STAGE2:-}" ]; then
  git fetch --quiet origin main
  git reset --quiet --hard origin/main
  export STAGE2=1
  exec "$REPO_DIR/run.sh" "$REPO_DIR"
fi"""

# The shape that broke: reset in place and keep going in the same process.
_PLAIN_BLOCK = """git fetch --quiet origin main
git reset --quiet --hard origin/main"""


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _build_repo(tmp_path: Path, reset_block: str) -> Path:
    """Return a clone whose origin/main holds version 2 of run.sh.

    The clone's working tree is left at version 1, reproducing the state hallu
    is in at 03:00: on disk is last night's script, origin has tonight's.
    """
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "--quiet", "--initial-branch=main")
    _git(upstream, "config", "user.email", "test@example.com")
    _git(upstream, "config", "user.name", "Test")

    script = upstream / "run.sh"
    for version in (1, 2):
        script.write_text(
            _SCRIPT_TEMPLATE.format(version=version, reset_block=reset_block),
            encoding="utf-8",
        )
        script.chmod(0o755)
        _git(upstream, "add", "run.sh")
        _git(upstream, "commit", "--quiet", "-m", f"version {version}")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "--quiet", str(upstream), str(clone))
    # Roll the working tree back to version 1 without moving origin/main.
    _git(clone, "reset", "--quiet", "--hard", "HEAD~1")
    assert "version=1" in (clone / "run.sh").read_text(encoding="utf-8")
    return clone


def _run(clone: Path) -> str:
    result = subprocess.run(
        ["bash", str(clone / "run.sh"), str(clone)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


class TestInodeHazard:
    def test_plain_reset_keeps_running_the_old_script(self, tmp_path):
        """Documents the bug: the process finishes on version 1's code.

        If a future git or bash ever stops exhibiting this, the re-exec becomes
        belt-and-braces rather than load-bearing, and this test failing is the
        signal to revisit that comment in the cron script.
        """
        clone = _build_repo(tmp_path, _PLAIN_BLOCK)
        output = _run(clone)

        assert "START version=1" in output
        # The decisive line: after the reset, execution continued from the old
        # inode even though the file on disk is now version 2.
        assert "END version=1" in output
        assert "END version=2" not in output
        assert "version=2" in (clone / "run.sh").read_text(encoding="utf-8")

    def test_reexec_runs_the_freshly_checked_out_script(self, tmp_path):
        """The fix: stage 2 is version 2, the revision that was checked out."""
        clone = _build_repo(tmp_path, _REEXEC_BLOCK)
        output = _run(clone)

        assert "START version=1" in output
        assert "START version=2" in output
        assert "END version=2" in output
        # Version 1 never reaches its own END; it execs away first.
        assert "END version=1" not in output

    def test_reexec_happens_exactly_once(self, tmp_path):
        """The guard must stop stage 2 from resetting and re-execing again."""
        clone = _build_repo(tmp_path, _REEXEC_BLOCK)
        output = _run(clone)

        assert output.count("START version=2") == 1
        assert output.count("END version=2") == 1


class TestProductionScript:
    def test_cron_script_reexecs_after_the_reset(self):
        """A structural guard on the real script.

        Cheap and narrow on purpose: the behavioral proof is above, but nothing
        there would notice if someone deleted the handover from the production
        script, which is the regression that actually costs a night of data.
        """
        source = CRON_SCRIPT.read_text(encoding="utf-8")

        assert "HALLU_CRON_STAGE2" in source, "the stage guard is gone"
        assert 'exec "$REPO_DIR/scripts/hallu_cron_pipeline.sh"' in source, (
            "the re-exec handover is gone"
        )

        reset_at = source.index("git reset --quiet --hard origin/main")
        exec_at = source.index('exec "$REPO_DIR/scripts/hallu_cron_pipeline.sh"')
        assert reset_at < exec_at, "the re-exec must follow the reset, not precede it"

        # Nothing may run the pipeline between the reset and the handover, or
        # it runs as the stale script again.
        between = source[reset_at:exec_at]
        assert "uv run" not in between, (
            "a pipeline step runs before the re-exec; it would use the old script"
        )
