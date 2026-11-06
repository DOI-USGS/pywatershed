---
name: review
description: Project-aware code review - run pywatershed's mechanical convention checks on the diff, then the built-in code-review skill for correctness, and merge both into one findings-only report written to a .md file in the repo root. Use when the user invokes /review, optionally with a PR number, branch, or effort level (low/medium/high/max), e.g. "/review 407 high". Also supports a blind second pass and fusing two reviews into one comprehensive report.
---

# Project-aware review

Two layers over one diff, merged into one report: (A) mechanical
checks of pywatershed's conventions — run as greps/scripts so they are
identical from review to review — and (B) the harness's built-in
`code-review` skill for correctness bugs and cleanups.

## Ground rules

- Findings only. Never apply fixes as part of a review; fixes are a
  separate, discussed step afterwards.
- The review assumes the test suite and the pre-commit hooks are run
  outside the review. Do not run tests or re-run anything pre-commit
  covers; the assumed items are listed as bullets up-front in step 2.
- **Writes: the report file only.** The merged report is written to a
  markdown file (step 2 fixes the path; step 6 writes it). Invoking
  `/review` is permission for that one write. Creating, mutating, or
  destroying *anything else* — including scratch files, notes, and
  fixes — requires asking the human AND receiving explicit permission
  first. This is the hard line; hold it for sub-agents too.
- **No machine-specific absolute paths in anything committed** — not
  in the report, not in this file. The `check-security` pre-commit
  hook (`.github/scripts/check_security.py`) fails the commit on them.
  Refer to tools and environments generically (`conda env list`, then
  `<env>/bin/<tool>`) and keep real paths in chat.
- **Running code: read-only verification is expected.** Running code
  that is not redundant with the project's CI and static checkers is
  fine and preferred over guessing — `python -c "import ..."` to
  enumerate types, a quick static check, reproducing a suspected crash
  to confirm a finding. Do not run the test suite or anything
  pre-commit covers. A finding verified by execution is worth more
  than a finding reasoned about; say which one it is in the report.
- **Diff acquisition: REST first, read-only git is also allowed.**
  Prefer the public GitHub REST API for PR targets (step 1). Read-only
  git — `diff`, `log`, `status` against the target — is permitted
  without asking; invoking `/review` is that permission, and it
  extends to the agents the built-in review spawns. Any git command
  that writes, moves, or discards state is still forbidden.
- Invoking `/review` is permission for the agents the built-in
  `code-review` skill spawns.
- You cannot launch `/code-review ultra` yourself (cloud-run, billed,
  human-triggered only) — hand the human the command instead.

## Procedure

1. **Preflight: tooling and one consolidated poll.** Do this before
   any checks, so nothing is discovered mid-run.

   **Diff source, in order of preference:**
   - **Public GitHub REST API (preferred for a PR target).** Needs no
     auth and no local tooling:
     `https://api.github.com/repos/DOI-USGS/pywatershed/pulls/<N>` and
     `.../pulls/<N>/files?per_page=100`. Follow redirects (`curl -sSL`)
     — the repo was renamed from `EC-USGS` and the API answers
     `301 Moved Permanently` without them. The `files` response
     carries a per-file `patch`; it is omitted for very large files,
     which are the generated `test_data/` fixtures you want excluded
     anyway. Filter those and the `.nc` binaries out before reviewing
     and say how many lines you dropped.
   - **Local git.** `git diff <base>...HEAD` works offline and is the
     only option for a branch with no PR. Caveat worth stating in the
     report: a diff against `upstream/develop` is only as fresh as the
     last fetch of that remote-tracking ref, so a stale ref silently
     yields a stale diff. Confirm the local branch tip matches the PR
     head (read `.git/refs/heads/<branch>`, compare to the API's
     `head_sha`) before treating a branch review as a PR review.
   - **`gh` (treat as unavailable for PR data).** Its GraphQL paths
     fail against this org with `Resource protected by organization
     SAML enforcement`, and that authorization is deliberately not
     being pursued — so `gh pr diff` and `gh pr create` are out. Use
     REST for the diff and the web UI to open a PR. If you need `gh`
     for something else, probe PATH first and then the conda
     environments (`conda env list`, then `<env>/bin/gh`); if it is
     missing, ask rather than improvising. Never block on `gh` when
     REST will do.

   **Then poll once**, in a single question set, for everything that
   is the human's call:
   - the effort level (see step 5 for why it matters);
   - the report path, whenever it is ambiguous or already taken
     (see step 2);
   - the target, if the invocation left it unclear.

   **Never proceed on a default for an unanswered question.** If the
   human answers some questions and not others — including answering
   with free text that does not map to any option — treat the rest as
   unanswered and re-ask those alone. Repeat until every blocking
   question has a real selection. The defaults documented in this file
   are for non-interactive runs, not a licence to skip confirmation.

2. **Establish the target diff and the output file.** Default target:
   the current branch vs `develop` (vs `main` only for release/hotfix
   branches). A PR number or branch in the invocation overrides.

   The report file lives in the repo root, named `review_<PR#>.md` for
   a PR target or `review_<branch>.md` for a branch target. If that
   name is ambiguous (no PR number and an unwieldy branch name, say)
   or the file already exists, that is one of the questions to fold
   into the step 1 poll — settle it before running any checks, never
   at the end.

   Before proceeding, state:
   - the target, the file count, and the report path;
   - how the diff was obtained (REST / git) and what was filtered out;
   - the effort level as confirmed by the human, with a one-line
     reminder that the choice matters strongly — depth and token cost
     scale with it, and at `high` and above the built-in review fans
     out multiple internal reviewer agents (multi-agent token usage
     should be expected);
   - the assumptions, as a bulleted list of everything this review
     takes as already checked outside it:
     - the test suite (`autotest/ci_local.sh` / CI);
     - the pre-commit hooks, enumerated live from
       `.pre-commit-config.yaml` at review time so the list never
       drifts (currently: ruff check/format, blackdoc, doctoc,
       nbstripout, the security script).

3. **Codegraph context (optional).** Check whether the codegraph MCP
   server's tools are available (ToolSearch for `codegraph`). If they
   are not, tell the human the review is better with it and suggest
   installing/enabling it — https://github.com/colbymchenry/codegraph
   (registered in Claude as `codegraph serve --mcp`; new sessions see
   it, sessions older than the install do not) — then continue
   without it. If they are available, use read-only codegraph queries
   before reviewing to establish:
   - the blast radius of the diff: callers/dependents of every
     changed public symbol, including impact beyond the files the
     diff touches;
   - the type hierarchy around changed classes (bases and
     subclasses).
   Feed both into the correctness layer and summarize them briefly in
   the report. Codegraph's "no covering tests found" note on a changed
   symbol is itself a finding worth reporting.

4. **Mechanical convention checks (A).** Check only what the diff
   touches; report pass/fail per item with the offending lines quoted:
   - New public class/function: exported in `pywatershed/__init__.py`
     (and the subpackage `__init__.py`) and listed in a hand-written
     `doc/api/*.rst` autosummary. Verify each documented target
     actually exists — an autosummary entry pointing at a missing
     symbol breaks the docs build.
   - User-facing change: has a `doc/whats-new.rst` entry with
     `(:pull:`XXX`)` or the real PR number.
   - `.github/workflows/ci.yaml` changed: `autotest/ci_local.sh`
     updated to match (job/test-file correspondence — a read of both
     files, never a run of either). Check the pytest *flags* too, not
     just the file lists: `ci_local.sh` exists to reproduce CI, so a
     worker-count or marker constraint CI needed (e.g. `-n=1` to avoid
     an OOM) has to be mirrored there.
   - New domain-test CI job: carries the skeleton/full `if:` gate with
     a `ci-<token>` (an ungated job runs on every push; see
     DEVELOPER.md "CI").
   - New/changed conditional skip in a test: `--ignore`d in the broad
     CI steps or given a dedicated step whose `--control_pattern`
     avoids it (CI runs `--error-for-skips`). Enumerate every step
     that collects the file rather than spot-checking. Read the skip
     guard itself: a guard testing an option's *presence*
     (`if "flag" not in control.options`) behaves differently from one
     testing its *value*, and a control file that sets the flag to `0`
     will not skip. Before calling it a CI failure, confirm the test
     would actually fail rather than pass as redundant coverage.
   - Dependency change: `pyproject.toml` vs `environment.yml` vs
     `environment_w_jupyter.yml` kept consistent (conda names may
     differ, e.g. `epiweeks4cf`).
   - Docstrings (ruff's lint selection has no pydocstyle rules, so
     nothing else checks these): every new or signature-changed
     public class/function/method has a docstring whose documented
     parameters, returns, and raises match the final code; behavior
     changes update the docstring, not just the code. Style follows
     the surrounding module (Sphinx napoleon; Google `Args:` is the
     dominant style in the package).
   - Type hints: every new or signature-changed function/method is
     fully annotated — every parameter and the return type. This is
     an existence check, not an accuracy check (the repo has no
     mypy/static enforcement, so hint correctness is not chased);
     a missing annotation is a finding.
   - Tests added or changed: for each test, state in one line what
     regression it guards against, then confirm that regression would
     actually make it fail. A gap between the two is the finding. Flag
     it if: the exercised code is not what the name/docstring claims
     (mistargeted); the test defines the thing it asserts, e.g. its
     own `get_parameters()`, so the feared defect cannot fail it
     (tautological); the assertion holds for any input by a general
     property of a base class (vacuous); another test in the diff or
     suite already pins the behavior (redundant); or its intent is not
     obvious from the assertions and no comment names the regression
     it guards (unexplained). Recommend deletion for the first four,
     a comment for the last. Tests should be as short as the point
     allows; length beyond that needs the comment too.

5. **Correctness review (B).** Invoke the built-in `code-review`
   skill on the same target at the effort level confirmed in step 1
   (documented default `high`, but confirm it — see step 1). Note:
   at `high` and above the built-in skill fans out several parallel
   reviewer subagents and then verifies findings — one `/review`
   invocation, many agents inside. In the report, translate any
   agent-count language into plain confidence terms ("flagged
   independently by two of the parallel reviewers, so higher
   confidence"), so the human is not left wondering how many reviews
   they paid for.

   **Pass the ground rules down explicitly when you invoke it.** State
   that read-only verification by execution is welcome, that read-only
   git is permitted, and that the report file is the only permitted
   write. Without this the sub-review either over-restricts itself and
   guesses where it could have checked, or writes stray files.

6. **One merged report, written to the file from step 2.** Convention
   findings first (they are cheap to fix and objective), then
   correctness findings ranked by severity. For each finding quote the
   offending code, not a description of it. Where a reference
   implementation exists (the PRMS Fortran in the pws-prms-oracle
   repository's `prms_src/`), quote
   the corresponding lines alongside. State explicitly when a layer
   found nothing, and list what was checked and cleared — a reader
   needs to know what the review looked at and did not flag. Write
   this report to the path fixed in step 2, show it in chat as well,
   and end by stating the report's path.
   For a large or high-stakes diff (release PRs, multi-hundred-line
   features), end with the optional escalation line for the human to
   run, e.g. `/code-review ultra 407` — a separate, deeper cloud
   review; do not attempt to run it yourself.

## Second pass and fusing

A second review of the same target is worth running when the diff is
large or high-stakes. Run it **blind**: do not read the existing
report before or during the pass. A review handed the prior findings
anchors on them — it spends its budget confirming and refuting what is
already there and inherits the same blind spots. Agreement between two
genuinely independent passes is real signal; agreement with a report
you just read is not.

- The second report goes to `review_<target>_second.md` (confirm the
  name in the step 1 poll, since `review_<target>.md` is taken).
- Afterwards, offer to fuse. The fused report goes to
  `review_<target>_fused.md` and supersedes both inputs, so it must be
  self-contained — the human should not need to open the other two.

A fused report needs, beyond the merged findings:

- **A provenance tag on every finding** — found by both passes
  (highest confidence), or by one only. Note which pass, and record
  each pass's head SHA and date, since the two may have reviewed
  different commits.
- **Explicit reconciliation of disagreements.** Where the passes
  conflict, say which one is right, why, and what the residual finding
  is after correction. Do not quietly drop the loser; a reader who
  saw the first report needs to know it was superseded. Own it plainly
  when the later pass was the wrong one.
- **A coverage note** on how the passes divided — if one went wide on
  regressions in untouched code and the other deep on the class
  hierarchy, say so, because it tells the human whether a third pass
  would find more.
- **A suggested order of attack** at the end, grouping fixes that
  belong in one change (a bug and the test gap that hid it) rather
  than listing findings in severity order again.
