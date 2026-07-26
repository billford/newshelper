# ADR-003: Current-Events Chatbot — RAG, Not Per-Build Fine-Tuning

Status: Accepted — Phase 1 (ingestion) in progress
Date: 2026-07-25
Deciders: Billfordx

## Context

The ask: a chatbot that can discuss current events, refreshed from each new
build, running entirely on local models on a new GPU cluster (RTX 5060
16GB cards, behind Olla as an Ollama-compatible load balancer/proxy). The
original framing was "dynamically train a new model on each build's
articles and books" — evaluated and rejected below.

## Decision 1: RAG, not per-build fine-tuning

Fine-tuning optimizes for patterns in training data, not queryable factual
recall — models fine-tuned on documents commonly produce fluent, wrong
answers about those documents. Running it twice a day compounds the risk
(repeated fine-tune passes risk catastrophic forgetting of both general
capability and prior news), gives no source attribution (a fact baked into
weights can't be traced to the article it came from — this is a news
chatbot, that matters specifically here), and has no cheap rollback (a bad
fine-tune cycle means retraining, not re-indexing). Twice-daily cadence is
also faster than any realistic training+eval loop can safely keep up with.

RAG addresses all of this: updating the index is fast (embed + write),
retrieval is inspectable and citable, and a bad build can be re-indexed or
rolled back without touching the model. Fine-tuning is reserved for a
separate, infrequent (weekly, never per-build), style-only LoRA track that
never sees raw article text as training data (Phase 4, not scoped yet —
see Non-goals below).

## Decision 2: content scope — index what newshelper already produces, not full article bodies

Phase 0 discovery (inspecting the actual repo, not assumption) found: RSS
gives `HeadlineCandidate` only `title`, `link`, `source`, `published`
(raw string) — no article body text anywhere in the pipeline.
`BookRecommendation` / `ArticleRecommendation` / `FactCheckResult` are
title+URL only. The only prose that exists per story is
`EnrichedStory.summary` (2-4 AI-generated sentences).

Two ways to close that gap were considered: (a) index what already exists
(title + summary + citations + recommendations, no chunking really
needed), or (b) add a new fetch-full-article-body step (scrape the
original HTML from each candidate's link). **(a) was chosen.** (b) is a
real scope increase — a new dependency, fragile against arbitrary site
structures, and a step beyond what the digest itself needs — and conflicts
with this project's standing preference for minimizing fragility. The
chatbot's factual depth is therefore capped at "2-4 sentence summary,"
same ceiling the site itself already has. If that ceiling turns out to be
the limiting factor in practice, (b) is the documented fallback, not a
surprise.

## Decision 3: vector store — LanceDB, not Chroma

Both are reasonable, embedded (no server process), locally-persisted
choices at this scale (twice-daily builds, a few thousand chunks in the
90-day retention window at most — this is not a "vector database" problem,
it's a "small local index" problem). Concretely compared by dry-run
installing both against this repo's Python 3.14 venv:

- `chromadb`: ~50 packages, including a kubernetes client, grpcio,
  opentelemetry exporters, and a full uvicorn/FastAPI server stack — all
  pulled in even though only the embedded `PersistentClient` mode would
  ever be used.
- `lancedb`: ~10 packages (`pyarrow`, `pydantic`, `lance-namespace`, a
  couple of small utilities). No server-mode cruft.

LanceDB matches this project's existing minimal-dependency posture
(`feedparser`, `jinja2`, `requests`, `kokoro-onnx`, `numpy`, `pillow` — no
heavy frameworks anywhere else in the codebase) and stores data as
Lance-format files directly on disk, which is easy to gitignore and
reason about. Chosen on dependency-weight grounds, not a capability gap —
either would technically work.

## Decision 4: id synthesis

No dataclass in `models.py` has an id field, and `Story.title` is not a
stable key (fuzzy-clustered, freeform, not guaranteed unique across
builds). Doc ids are synthesized as
`f"{build_date:%Y-%m-%d}-{slugify(story.title)}"`, scoped to the build
that produced them — collisions across the *same* build are not expected
(top stories are already deduplicated by `rank.py`'s clustering) and a
same-day re-run intentionally overwrites rather than duplicates.

## Decision 5: retention — stale flag, not a second collection

`EnrichedStory` items go in a `current` collection (news, time-windowed);
books go in a `reference` collection (append-only, no natural expiry —
confirmed in Phase 0 that books have no date field at all). Rather than
migrating aged-out `current` documents to a third `current_archive`
collection, aging-out sets a `stale=true` metadata flag in place and
default queries filter `stale=false`. Simpler to implement and query than
managing a second collection's lifecycle, and satisfies the same
requirement (never hard-delete by default; a "what happened last month"
query can still opt in to stale documents).

## Decision 6: hook point

No event/callback system exists anywhere in this codebase. The natural
in-process hook is `build.py`'s `run()`, immediately after
`output_dir = write_site(enriched)` and before `return 0` — `enriched`
(the `list[EnrichedStory]`) is already in scope there. Wrapped in the same
try/except-log-and-continue pattern already used for video generation in
that function: RAG ingestion failing must never fail the whole build.

## Model note (chat serving, Phase 2 — flagged now since it affects Phase 2 planning)

The digest pipeline's actual production model is `qwen2.5:32b` (set via
`NEWSHELPER_OLLAMA_MODEL` in `daily_build.sh`, not the `llama3.1:8b`
default in `config.py`) — confirmed in Phase 0. A 32B model does not fit
on a single 16GB RTX 5060 card even quantized. The chat-serving model
(Phase 2) will need its own, separate, smaller pick (7-8B class, Q4/Q5)
validated against actual VRAM headroom on the real cluster — this is not
a decision this ADR makes, since the cluster isn't online yet; flagged
here so Phase 2 doesn't quietly assume `qwen2.5:32b` is reusable as-is.

## Non-goals (this ADR)

- No per-build fine-tuning or retraining of any kind.
- No cloud model calls anywhere in the pipeline.
- No full-article-body fetching (see Decision 2) — revisit only if the
  summary-only ceiling proves to be the actual limiting factor.
- The weekly persona LoRA track (Phase 4) is out of scope until Phases 1-3
  are live and an actual tone gap is identified — not built speculatively.

## Consequences

- New dependencies: `lancedb`, `pyyaml` (already present transitively,
  now declared explicitly rather than relied upon implicitly).
- New config surface: `config/rag.yaml`, loaded via `yaml.safe_load` only.
- New data on disk: a LanceDB directory (gitignored, machine-local, like
  `logs/` and the Kokoro/mascot model files).

## Decision 7: Phase 2 architecture — where retrieval actually lives

The cluster came online behind Olla on a separate machine ("lampoon"),
which already runs an existing pattern for chatbot widgets
(`~/llm-chat-widget`, documented in `ADDING-A-NEW-CHATBOT.md`): a small
Node.js proxy per site (guardrails + auth + CORS), Tailscale-Funneled
publicly, forwarding to Olla's OpenAI-compatible endpoint. That proxy is a
**passthrough** — it doesn't know about retrieval. Two things had to be
decided to make this actually RAG rather than a plain chatbot:

**Where retrieval runs.** The LanceDB store is local, file-based data on
wanderlust; lampoon can't query it directly. wanderlust and lampoon turned
out to already be on the same home LAN (`lampoon` resolves to
`192.168.1.156`, no Tailscale required for this hop) — so retrieval runs
as its own small always-on service, `rag_serve.py`, bound to `0.0.0.0` on
wanderlust and reachable at its LAN address. It is **never Funneled or
otherwise exposed to the public internet** — only the final
chat-completion hop (already public on lampoon for the travel bot) needs
that; retrieval doesn't, and keeping it LAN-only means a mistake
configuring it can't become a public-internet exposure. Lampoon's
newshelper-specific proxy (`~/chatbot-newshelper/server/rag-server.js` on
lampoon; reference copy at `scripts/lampoon-newshelper-rag-server.js` in
this repo) calls `rag_serve.py`'s `/retrieve` endpoint, then injects the
results into the model prompt as `<source>` blocks with an explicit
instruction to treat them as data, never instructions — this is the
prompt-injection guardrail §5 of the original spec required.

**One Funnel port per bot, not path-based routing under one port.** The
original ask was to avoid spending a second public port on a second bot.
`tailscale serve --set-path` looked like the answer (route `/newshelper`
and `/` to different backends under lampoon's existing port 443) — but in
practice, reconfiguring Serve for a new path **silently dropped Funnel
exposure for the whole port**, briefly taking the live travel bot
offline. That's real, demonstrated fragility on shared infrastructure, not
a hypothetical one. **Decision: each bot gets its own dedicated Funnel
port** (travel bot keeps 443; newshelper is 8443), exactly as
`ADDING-A-NEW-CHATBOT.md` already documented as the default pattern for a
2nd bot — isolated by construction, a config mistake on one bot's port
cannot touch another's.

**Model**: `llama3.1:8b`, already available on the Olla cluster and
matching the model-class note in this ADR's original model note (32B
doesn't fit a single 16GB card; 7-8B does).

**Retrieval tuning note**: `retrieval.top_k` started at the original
spec's suggested default of 8, but a real test against an early (~12
chunk) store showed it pulling back nearly everything ingested so far
regardless of relevance, and the citations footer listed sources the
answer never actually drew from. Lowered to 4. Still a guess, not
rigorously tuned — revisit as the store grows past its first few days
(Phase 3).

## Action Items

- [x] Phase 0 discovery against the real repo, all `[ASSUMED]` items
      resolved or flagged
- [x] Phase 1: `newshelper-rag` ingestion module, LanceDB store, retention
      policy, full test suite
- [x] Phase 2: `newshelper-chat` serving — `rag_retrieve.py` (recency-
      weighted ranking) + `rag_serve.py` (LAN-only HTTP endpoint) on
      wanderlust; lampoon's `chatbot-newshelper` deployment (own systemd
      service, own Funnel port 8443, own `rules.json`/`prod.env`); widget
      vendored into `static/chat-widget/`, embedded on the digest page.
      Verified end-to-end through the real public URL with real
      retrieval, real citations, and auth/guardrails enforced.
- [ ] Phase 3: hardening (recency weighting tuning, load test, retrieval
      quality as the store grows past its first few days)
- [ ] Phase 4: persona LoRA — not started, not scoped, revisit later
