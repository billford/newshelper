window.ChatWidgetConfig = {
  endpoint: "https://lampoon.tailf55a4b.ts.net:8443/v1/chat/completions",
  apiKey: "01bf91906884b4cf4e224aced522fd9bde7b636572cf02118398685fb299b560",
  model: "llama3.1:8b",
  title: "Ask NewsHelper",
  subtitle: "Grounded in today's digest",
  greeting: "Hi! Ask me about any story from today's NewsHelper digest -- I'll answer from the digest itself and cite what I used.",
  systemPrompt: "You are NewsHelper's current-events assistant. Answer only from the digest content you're given, and say so plainly if you don't have information on something rather than guessing.",
  avatarUrl: "static/brand/mascot.png",
  position: "right",
  footerNote: "AI-generated from NewsHelper's own digest -- always double-check anything important against the linked sources.",
  storageKey: "newshelper-chat"
};
