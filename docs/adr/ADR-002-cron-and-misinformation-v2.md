# ADR-002: v2 — Scheduled Builds, Grounded Misinformation Tagging, Source Attribution

Status: Accepted — shipped 2026-07-21
Date: 2026-07-21
Deciders: Billfordx

## Context

ADR-001 shipped v1: manual builds, satire-domain tagging only, and go-deeper
links with no indication of which original articles fed the AI summary.
Three gaps were prioritized for v2:

1. Publishing was still a manual `build` + `publish.sh` run — not actually
   "daily" without a human remembering to do it.
2. Satire tagging (a domain allowlist) only catches one kind of
   misinformation-adjacent content. It says nothing about a real news
   story that happens to repeat a claim that's actually false.
3. The rendered page showed source *names* ("bbc, npr") next to a story
   but never linked to the actual RSS articles the summary was built from.

## Decisions

### 1. Scheduling: launchd, not cron

wanderlust is this machine, running macOS — launchd is the native
scheduling mechanism there (not systemd, and more reliable across
sleep/wake than plain cron). Ships as:

- `scripts/daily_build.sh` — wrapper that runs `python -m newshelper.build`
  then `scripts/publish.sh`, logging every run to `logs/daily_build.log`
  (gitignored). Deliberately does not use `set -e`; each stage's failure is
  handled explicitly so a failed build never silently skips notification.
- `scripts/com.billford.newshelper.daily.plist` — the LaunchAgent
  definition (checked into the repo as a template), installed to
  `~/Library/LaunchAgents/` and loaded via `launchctl load`.
- **Twice daily, not once** — 9:00 AM and 6:00 PM local time
  (`StartCalendarInterval` array with two entries), since news moves
  faster than a once-a-day digest can keep up with.
- On failure, `daily_build.sh` fires a local macOS notification via
  `osascript` (free, no new credentials) in addition to the log file.

### 2. Misinformation tagging v2: Google Fact Check Claims Search API

ADR-001 explicitly flagged two candidate mechanisms and left both out of
v1: a grounded Fact Check Claims API lookup, and a soft LLM "unverified,
treat with caution" heuristic (flagged as needing careful wording so it
never reads as an independent truth verdict). **v2 builds the Fact Check
API path, not the LLM heuristic** — same rationale as book verification in
v1: a real, independently-published, real-world source beats an LLM's own
judgment of truth.

Mechanism (`src/newshelper/factcheck.py`):

- Query `factchecktools.googleapis.com/v1alpha1/claims:search` with the
  story's headline text.
- **Requires a Google Cloud API key** with the Fact Check Tools API
  enabled — unlike the books APIs, this one is not keyless. Set
  `NEWSHELPER_FACTCHECK_API_KEY` in wanderlust's local environment (never
  committed to git). With no key configured, lookups are skipped
  entirely — this must never fail a build. **This key has not been
  provisioned yet; fact-check lookups are currently a no-op in production
  until one is created and set.**
- Fact-check search is keyword-based, not story-aware — a query can return
  a claim that shares a word or two with the headline but is about
  something else entirely. To avoid mislabeling an unrelated real story as
  "disputed," a match is only accepted when the claim text's similarity to
  the headline (via the same `difflib`-based `rank.similarity()` used for
  cross-feed clustering) clears a threshold
  (`FACT_CHECK_SIMILARITY_THRESHOLD = 0.4`). Of multiple claims returned,
  only the highest-similarity one above threshold is kept.
- **Never asserts an independent verdict.** The rendered notice surfaces
  the existing published rating, publisher name, and a link to the full
  fact-check, plus an explicit caveat: "This is a claim judged similar to
  this headline, not necessarily about the exact same story — follow the
  link and judge for yourself." The page never says "this is false" in its
  own voice.
- Deliberately independent of satire tagging (`satire.py`) — one is an
  exact domain-allowlist match, free and precise; the other is a fuzzy,
  keyword-based, rate-limited external lookup. A story can be flagged by
  either, both, or neither; neither mechanism's logic references the
  other. Tests (`test_factcheck.py::test_satire_and_fact_check_are_independent_signals`)
  confirm this independence explicitly.
- Runs only on the final top-6 stories (in `enrich.py`, alongside book
  verification), not during clustering over the full ~250 raw candidates —
  a network call per candidate would be both slow and needlessly
  API-quota-hungry.

### 3. Source attribution ("go deeper" cites its own inputs)

Every `HeadlineCandidate` that fed a story's cluster is now surfaced as a
"SOURCE"-tagged go-deeper link (`enrich.build_source_citations`) — the
original RSS entry, title and link, that the model's summary was built
from. This required no new network or model call: the candidate list
already exists post-clustering, so the citation is purely structural and
cannot be hallucinated.

## Trade-offs

- Fact Check API coverage will be sparse for same-day stories — most
  published fact-checks lag the news cycle by days or weeks. This is
  expected and acceptable: the feature's job is to catch the cases where
  a fresh headline echoes an already-debunked claim, not to fact-check
  breaking news in real time.
- The similarity threshold (0.4) is a precision/recall trade-off tuned by
  spot-checking a handful of headline/claim pairs during development, not
  a rigorously validated cutoff — worth revisiting if false positives
  (unrelated claims attached to real stories) or false negatives (missed
  genuine matches) show up in practice.
- launchd is macOS-specific. If wanderlust ever changes OS, this
  scheduling mechanism doesn't port — acceptable since wanderlust's
  identity (this Mac) is now confirmed, not a future unknown.

## Consequences

- `NEWSHELPER_FACTCHECK_API_KEY` needs to be provisioned (Google Cloud
  project + Fact Check Tools API enabled + API key generated) before this
  feature does anything in production. Until then it's a correctly-behaving
  no-op, not a broken feature.
- `logs/` is now gitignored — `daily_build.log`, `launchd.out.log`,
  `launchd.err.log` all live there, machine-local, not versioned.
- 51 tests passing (14 new: 8 for `factcheck.py`, 4 for the `enrich.py`
  integration, 2 for the new render paths), pylint 10.00/10, bandit clean.

## Action Items

- [x] Twice-daily launchd job installed and loaded on wanderlust
- [x] Fact Check Claims API integration, similarity-gated, independent of satire tagging
- [x] Source citation links for every go-deeper list
- [ ] Provision `NEWSHELPER_FACTCHECK_API_KEY` (Google Cloud project + API key) — blocks the fact-check feature from doing anything live
- [ ] Revisit `FACT_CHECK_SIMILARITY_THRESHOLD` once real fact-check results start coming through
- [ ] Confirm the twice-daily launchd job actually fires reliably over the first several days (check `logs/daily_build.log`)
