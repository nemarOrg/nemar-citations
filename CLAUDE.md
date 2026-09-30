@AGENTS.md

## Claude Code Specific Instructions

The shared project instructions live in `AGENTS.md`; this file imports them for Claude Code with `@AGENTS.md`.

### Plan documents
- `plan.md` (gitignored, repo root) — personal planning scratchpad. Update regularly during development.
- `scratch_notes.md` (gitignored, repo root) — technical notes and research findings.

### Tooling
- **Playwright MCP** — web automation and visual analysis (NEMAR.org design research, dashboard QA)
- **Serena MCP** — code intelligence (see `.rules/serena_mcp.md`)
- `/review-pr` — multi-agent PR review skill from the pr-review-toolkit plugin

### Plan mode
Use `/plan` before non-trivial implementations; align on the approach before changing files.

### Cross-repo work (NEMAR triangle)
This repo is one of three sibling repos that jointly produce the public NEMAR surface. The contracts are in `.rules/cross_repo.md`; here is the Claude-specific procedural guidance.

- **Sibling repos exist locally**, but **not at `../`** relative to this repo. They are at:
  - `/Users/yahya/Documents/git/nemar/website/` (Astro 6 SSR for `nemar.org`)
  - `/Users/yahya/Documents/git/nemar/nemar-cli/` (Bun CLI + Cloudflare Worker backend for `api.nemar.org` and `data.nemar.org`)
- When the user says "the website" or "nemar-cli", they mean those paths. Don't assume sibling layout.
- **Before claiming a route or schema exists, verify it.** The live API is the ground truth, not stale docs:
  - `curl -s "https://api.nemar.org/datasets?limit=2"` — confirms catalog availability and shape.
  - `curl -s "https://data.nemar.org/<id>/metadata.json"` — works for `nm-*` IDs, returns 404 for legacy `ds-*`.
- **Don't edit sibling repos from this session unless explicitly asked.** Cross-repo changes (e.g., adding a column or a sync to `nemar-cli/backend/`) need their own branch + PR in the target repo and should be raised as an issue here first.
- **When reading sibling code**, prefer `Explore` agent over `grep` for breadth — the relevant files span TypeScript + Python and naming conventions differ. Already-validated entry points: `nemar-cli/backend/src/routes/data.ts`, `nemar-cli/backend/src/routes/datasets.ts`, `nemar-cli/shared/datacite-constants.ts`, `website/src/lib/data-api.ts`.

### Rate-limit posture (Claude-facing)
GitHub and Semantic Scholar both throttled us during the May 2026 backfill. Operational rules in `.rules/cross_repo.md` § Fetch Strategy. Claude-specific:
- **Don't kick the cron** during interactive sessions to "see what happens" — the workflow uses real opencite calls and costs us our rate-limit budget. Trigger only when you have a fix to verify.
- **Prefer reading `api.nemar.org/datasets` once and caching** the JSON to disk for the session over repeated requests.
- **For background investigations**, use `Explore` / `general-purpose` agents with explicit tool budgets rather than open-ended scraping.

Keep cross-agent project rules in `AGENTS.md` so Codex, Copilot, Cursor, and other AGENTS.md-aware tools stay aligned. Append only Claude-specific plugin / skill / command / MCP guidance below.
