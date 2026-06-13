# wegofwd-llm — Watch Setup

This directory documents the **top-level-watch** mechanism for the wegofwd-llm
package. It is part of the v2.6 admission to the `project-critique` four-lens
watch set. See [`project-critique/wegofwd-llm-critique.md`](https://github.com/wegofwd2020-hub/project-critique/blob/main/wegofwd-llm-critique.md)
§8 for the cadence rationale.

## What it does

`.github/workflows/watch.yml` runs:

- **weekly** (Mondays 09:00 UTC),
- **on every push to `main`**, and
- **on manual `workflow_dispatch`**.

It compares the current `wegofwd-llm` HEAD against the last-reviewed commit
recorded in `project-critique/wegofwd-llm-last-reviewed.txt`. If anything has
changed, it opens a PR to `wegofwd2020-hub/project-critique` adding a dated
report (`wegofwd-llm-watch-YYYY-MM-DD.md`) and bumping the pointer. Quiet
weeks produce **no PR** (and no noise).

It does **NOT** re-run the test suite (the existing `ci.yml` already does
that on every push and PR). It does **NOT** auto-re-critique the code — the
report is a *prompt* for a human/Claude refresh of the critique, not a
generated critique.

## Why a cross-repo workflow

`wegofwd-llm` is load-bearing across the WeGoFwd2020 portfolio
(StudyBuddy_OnDemand, StudyBuddy_SelfLearner/Mentible, Kathai Chithiram).
A regression here is a portfolio incident, not a single-product incident.
Centralising the watch trail in `project-critique` means the same place
that holds the critique also holds the change-detection log, in PR-review
form so the operator sees every meaningful change without polling the repo.

## One-time setup

### 1. Create the cross-repo token

The workflow needs to (a) read `project-critique`, (b) create a branch +
commit + push, and (c) open a PR. The default `GITHUB_TOKEN` is scoped to
the running repo only, so we need a separate token with `repo` scope on
`wegofwd2020-hub/project-critique`.

Two acceptable forms:

**Option A — Fine-grained Personal Access Token (recommended for a solo founder).**
Go to https://github.com/settings/personal-access-tokens, **Generate new token**,
and set:

- **Resource owner:** `wegofwd2020-hub`
- **Repository access:** Only select repositories → `wegofwd2020-hub/project-critique`
- **Repository permissions:**
  - `Contents` → **Read and write** (to push the watch branch)
  - `Pull requests` → **Read and write** (to open the PR)
  - `Metadata` → Read-only (granted automatically)
- **Expiration:** pick a date you'll actually rotate at (e.g. 1 year)

Copy the generated token (it will only be shown once).

**Option B — GitHub App installation token.** More robust for an
organization with multiple maintainers; overkill for a solo founder. If you
go this route, install the App on `project-critique` with `Contents: write`
and `Pull requests: write` permissions, and grant the App's installation
token to this workflow via `actions/create-github-app-token` instead of a
raw secret. The workflow's secret name should still be
`PROJECT_CRITIQUE_PR_TOKEN` (the steps reference it by that name).

### 2. Add the secret

Go to `https://github.com/wegofwd2020-hub/wegofwd-llm/settings/secrets/actions`
and **New repository secret**:

- **Name:** `PROJECT_CRITIQUE_PR_TOKEN`
- **Value:** the token from step 1.

### 3. Bootstrap the baseline (optional)

The workflow defaults to `4823606` (v0.1.2, 2026-06-09) as the baseline
when `project-critique/wegofwd-llm-last-reviewed.txt` does not yet exist.
That is the HEAD at the moment wegofwd-llm was admitted to the critique
suite (v2.6 of `project-critique`). On the **first** workflow run after
setup, the watch will report every commit since that baseline.

If you want to start watching from a different baseline (for instance, you
want to ignore changes already covered by the v1.0 critique), create the
file manually before the first run:

```bash
cd path/to/project-critique
echo "4823606ca287cd19e658f9497691a6f2d2947708" > wegofwd-llm-last-reviewed.txt
git add wegofwd-llm-last-reviewed.txt
git commit -m "chore(watch): set wegofwd-llm baseline to v0.1.2 (4823606)"
git push
```

### 4. Verify

Trigger the workflow manually from the Actions tab
(`watch` → **Run workflow** → branch `main`). Two acceptable outcomes:

- **No delta:** the workflow run summary says *"✅ wegofwd-llm watch — no
  change since baseline"* and no PR is opened. Confirms the token + baseline
  are wired correctly.
- **Delta detected:** the workflow opens a PR to `project-critique` titled
  `watch(wegofwd-llm): delta since <baseline> (<N> commits)`. Confirms the
  full path works end-to-end.

## Manual fallback

If the cron is down, the token expires, or you want to run a watch report
locally:

```bash
# from wegofwd-llm/
BASELINE=$(cat ../project-critique/wegofwd-llm-last-reviewed.txt 2>/dev/null || echo 4823606)
CURRENT=$(git rev-parse HEAD)
echo "Baseline: $BASELINE"
echo "Current:  $CURRENT"
echo ""
git log --oneline "$BASELINE..HEAD"
echo ""
git diff --stat "$BASELINE..HEAD"
```

Copy the output into `project-critique/wegofwd-llm-watch-YYYY-MM-DD.md` and
update `wegofwd-llm-last-reviewed.txt` manually.

## Security notes

The workflow's `run:` steps that consume `git log` commit subjects and
`git diff` file paths pass those values through `env:` rather than
`${{ }}` interpolation, so a hypothetical malicious commit subject like
`oops"; rm -rf / && curl evil`  cannot execute when the watch processes
it. Commit messages and PR bodies are written to tmp files and consumed
via `git commit -F` and `gh pr create --body-file` respectively, never via
inline `-m "..."` / `--body "..."` arguments. See the inline comments in
`watch.yml` for the threat-model rationale.

The token has scope only on `project-critique` (not on every wegofwd2020-hub
repo) and only the permissions needed to open a PR. If it leaks, the blast
radius is one repo's PR feed.
