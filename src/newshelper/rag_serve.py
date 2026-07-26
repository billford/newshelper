"""Minimal HTTP service exposing rag_retrieve.retrieve() (ADR-003 Phase 2)
to other machines on the home LAN -- specifically lampoon's chatbot proxy,
which has no direct access to wanderlust's local LanceDB store (an
embedded, file-based store, not a network service).

Deliberately LAN-only: bound per config/rag.yaml's `serve` section
(0.0.0.0 by default, so it's reachable from the LAN) but never Funneled or
otherwise exposed to the public internet. Only the final chat-completion
hop (on lampoon, already Funneled for the existing travel bot) needs
public reachability; retrieval doesn't, and keeping it LAN-only means a
mistake here can't become a public-internet exposure the way a Funneled
port would.

Run:
    python -m newshelper.rag_serve
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from newshelper.rag_config import RagConfig, load_config
from newshelper.rag_embed import EmbedClientProtocol, OllamaEmbedClient
from newshelper.rag_retrieve import retrieve
from newshelper.rag_store import VectorStore

logger = logging.getLogger(__name__)


def make_handler(store: VectorStore, embed_client: EmbedClientProtocol, config: RagConfig):
    """Builds a request handler closed over the given store/client/config
    (BaseHTTPRequestHandler subclasses can't take constructor args, so the
    dependencies are injected via closure instead -- this is also what
    makes the handler testable without a real Ollama/LanceDB behind it)."""

    class Handler(BaseHTTPRequestHandler):
        """Routes GET / (health check) and POST /retrieve. Method names
        (do_GET, do_POST) are dictated by BaseHTTPRequestHandler's own
        dispatch convention, not renameable to snake_case."""

        def _reply(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # pylint: disable=invalid-name
            """GET / -> a bare health check; anything else is 404."""
            if self.path == "/":
                self._reply(200, {"status": "ok"})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:  # pylint: disable=invalid-name
            """POST /retrieve {"query": str} -> {"results": [...]}."""
            # Drain the body before responding regardless of path -- an
            # unread body on a rejected request can reset the client's
            # connection instead of delivering the 404 cleanly.
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""

            if self.path != "/retrieve":
                self.send_response(404)
                self.end_headers()
                return

            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._reply(400, {"error": "invalid JSON body"})
                return

            query = payload.get("query", "")
            if not isinstance(query, str):
                self._reply(400, {"error": "'query' must be a string"})
                return

            try:
                results = retrieve(query, store, embed_client, config)
            except Exception:
                logger.exception("retrieval failed for query %r", query)
                self._reply(502, {"error": "retrieval failed"})
                return

            self._reply(
                200,
                {
                    "results": [
                        {
                            "title": r.title,
                            "url": r.url,
                            "text": r.text,
                            "collection": r.collection,
                            "published_at": r.published_at,
                        }
                        for r in results
                    ]
                },
            )

        def log_message(self, format_str: str, *args) -> None:
            """Route BaseHTTPRequestHandler's default stderr logging
            through the standard logging module instead."""
            logger.info("%s - " + format_str, self.address_string(), *args)

    return Handler


def main() -> None:
    """Load config, open the store, and serve /retrieve forever."""
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    config = load_config()
    store = VectorStore(config.store.path)
    embed_client = OllamaEmbedClient(
        host=config.embedding.host,
        model=config.embedding.model,
        timeout_seconds=config.embedding.timeout_seconds,
    )
    handler = make_handler(store, embed_client, config)
    server = ThreadingHTTPServer((config.serve.host, config.serve.port), handler)
    logger.info("rag_serve listening on %s:%d", config.serve.host, config.serve.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
