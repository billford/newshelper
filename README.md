# NewsHelper

Beyond-the-headline daily news digest. A static site that surfaces the day's
top 6 stories (by cross-feed frequency across free RSS sources) with an
AI-generated plain-language summary of each, plus a verified list of
deeper-reading follow-ups — including books, which most aggregators skip.

See [ADR-001](docs/adr/ADR-001-newshelper.md) for the full design rationale.

## Pipeline

```
fetch.py  -> rank.py -> enrich.py -> render.py
(RSS)        (cluster/   (local model  (Jinja2 ->
              top 6)      + book        dist/index.html)
                          verification)
```

1. **Fetch** — pull candidates from Google News, Google Trends, BBC, NPR RSS feeds.
2. **Rank** — cluster near-duplicate titles across feeds, score by source count, keep top 6. Plain code, not model-based.
3. **Enrich** — ask a local model (Ollama, running on "wanderlust") for a summary and book/article topic ideas per story, then verify every book suggestion against Open Library (fallback: Google Books) before it's allowed to be published.
4. **Render** — write a static `dist/index.html` (old-newspaper editorial style, no client-side JS).

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
```

To run a full build (requires Ollama serving locally):

```bash
python -m newshelper.build
```

Output lands in `dist/`.

## Publishing

`scripts/publish.sh` pushes `dist/` to the `gh-pages` branch from wherever
it's run. On wanderlust this is wired into cron, run right after
`newshelper.build`, using a locally-stored git deploy key/PAT — not GitHub
Actions secrets, since the local model is never exposed to the internet.

## Status

Enrichment currently runs against a mockable `OllamaClient` interface
(`src/newshelper/ollama_client.py`); wire up the real `wanderlust` endpoint
and model choice per the action items in ADR-001 before the first live
daily build.
