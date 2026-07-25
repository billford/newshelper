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
- Phase 2 (`newshelper-chat` serving) needs an actual reachable
  Olla/Ollama endpoint and a validated chat-model choice before it can be
  built for real — currently blocked on the GPU cluster coming online.

## Action Items

- [x] Phase 0 discovery against the real repo, all `[ASSUMED]` items
      resolved or flagged
- [ ] Phase 1: `newshelper-rag` ingestion module, LanceDB store, retention
      policy, full test suite
- [ ] Phase 2: `newshelper-chat` serving, once Olla/cluster is reachable
- [ ] Phase 3: hardening (recency weighting tuning, load test)
- [ ] Phase 4: persona LoRA — not started, not scoped, revisit later
