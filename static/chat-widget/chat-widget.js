/**
 * llm-chat-widget — a portable, dependency-free chat widget for talking to
 * any OpenAI-compatible chat/completions endpoint (Ollama, vLLM, llama.cpp
 * server, LM Studio, or a real OpenAI/Anthropic-compatible proxy).
 *
 * Usage: include chat-widget.css + chat-widget.js on any page, and define
 * window.ChatWidgetConfig before the script tag:
 *
 *   <script>
 *     window.ChatWidgetConfig = {
 *       endpoint: "https://your-tailscale-node.ts.net/v1/chat/completions",
 *       apiKey: "",              // optional bearer token
 *       model: "llama3.1",       // model name your server expects
 *       title: "Ask us anything",
 *       subtitle: "Usually replies in a minute",
 *       greeting: "Hi! What can I help you plan?",
 *       systemPrompt: "You are a helpful assistant that only discusses ...",
 *       avatarUrl: "/assets/img/avatars/avatar-neutral.png",
 *       position: "right",       // "right" | "left"
 *       footerNote: "AI-generated — for anything urgent, use Contact.",
 *       storageKey: "myproject-chat"
 *     };
 *   </script>
 *   <link rel="stylesheet" href="chat-widget.css">
 *   <script src="chat-widget.js"></script>
 *
 * The widget sends { model, messages, stream: false } and expects back
 * { choices: [ { message: { role, content } } ] } — the standard OpenAI
 * chat/completions shape that Ollama's compat endpoint, vLLM, and most
 * local inference servers all speak.
 */
(function () {
  'use strict';

  var cfg = window.ChatWidgetConfig || {};
  if (!cfg.endpoint) {
    console.warn('[llm-chat-widget] window.ChatWidgetConfig.endpoint is not set — widget will not be able to send messages.');
  }

  var opts = {
    endpoint: cfg.endpoint || '',
    apiKey: cfg.apiKey || '',
    model: cfg.model || 'default',
    title: cfg.title || 'Chat with us',
    subtitle: cfg.subtitle || '',
    greeting: cfg.greeting || "Hi! How can I help?",
    systemPrompt: cfg.systemPrompt || '',
    avatarUrl: cfg.avatarUrl || '',
    position: cfg.position === 'left' ? 'left' : 'right',
    footerNote: cfg.footerNote || '',
    storageKey: cfg.storageKey || 'llm-chat-widget:history'
  };

  var STORE_KEY = opts.storageKey;

  function loadHistory() {
    try {
      var raw = sessionStorage.getItem(STORE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  }

  function saveHistory(history) {
    try {
      sessionStorage.setItem(STORE_KEY, JSON.stringify(history));
    } catch (e) {
      /* storage unavailable — conversation just won't persist */
    }
  }

  var history = loadHistory(); // array of {role:'user'|'assistant', content:string}

  // ---- Build DOM ----
  var launcher = document.createElement('button');
  launcher.className = 'cw-launcher' + (opts.position === 'left' ? ' cw-launcher--left' : '');
  launcher.setAttribute('aria-label', opts.title);
  launcher.type = 'button';
  if (opts.avatarUrl) {
    var launcherImg = document.createElement('img');
    launcherImg.src = opts.avatarUrl;
    launcherImg.alt = '';
    launcher.appendChild(launcherImg);
  } else {
    launcher.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.5 8.5 0 0 1-4-1L3 20l1-4.5a8.38 8.38 0 0 1-1-4A8.38 8.38 0 0 1 11.5 3 8.5 8.5 0 0 1 21 11.5z"/></svg>';
  }

  var panel = document.createElement('div');
  panel.className = 'cw-panel' + (opts.position === 'left' ? ' cw-panel--left' : '');
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', opts.title);

  var header = document.createElement('div');
  header.className = 'cw-header';
  var headerHtml = '';
  if (opts.avatarUrl) {
    headerHtml += '<img src="' + opts.avatarUrl + '" alt="">';
  }
  headerHtml += '<div class="cw-header-text"><div class="cw-header-title"></div>' +
    (opts.subtitle ? '<div class="cw-header-subtitle"></div>' : '') + '</div>' +
    '<button type="button" class="cw-close" aria-label="Close chat">×</button>';
  header.innerHTML = headerHtml;
  header.querySelector('.cw-header-title').textContent = opts.title;
  if (opts.subtitle) header.querySelector('.cw-header-subtitle').textContent = opts.subtitle;

  var messagesEl = document.createElement('div');
  messagesEl.className = 'cw-messages';

  var inputRow = document.createElement('div');
  inputRow.className = 'cw-inputrow';
  inputRow.innerHTML =
    '<textarea class="cw-input" rows="1" placeholder="Type a message…"></textarea>' +
    '<button type="button" class="cw-send">Send</button>';

  panel.appendChild(header);
  panel.appendChild(messagesEl);
  if (opts.footerNote) {
    var footer = document.createElement('div');
    footer.className = 'cw-footer-note';
    footer.textContent = opts.footerNote;
    panel.appendChild(footer);
  }
  panel.appendChild(inputRow);

  document.addEventListener('DOMContentLoaded', mount);
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    mount();
  }

  function mount() {
    if (launcher.isConnected) return;
    document.body.appendChild(launcher);
    document.body.appendChild(panel);
    renderHistory();
  }

  // ---- Rendering ----
  function addBubble(role, text) {
    var el = document.createElement('div');
    el.className = 'cw-msg ' + (role === 'user' ? 'cw-msg--user' : role === 'error' ? 'cw-msg--error' : 'cw-msg--bot');
    el.textContent = text;
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return el;
  }

  function renderHistory() {
    messagesEl.innerHTML = '';
    if (history.length === 0 && opts.greeting) {
      addBubble('assistant', opts.greeting);
    } else {
      history.forEach(function (m) {
        addBubble(m.role, m.content);
      });
    }
  }

  function showTyping() {
    var el = document.createElement('div');
    el.className = 'cw-typing';
    el.innerHTML = '<span></span><span></span><span></span>';
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return el;
  }

  // ---- Networking ----
  function sendMessage(text) {
    history.push({ role: 'user', content: text });
    saveHistory(history);
    addBubble('user', text);

    var typingEl = showTyping();
    setSending(true);

    var payloadMessages = [];
    if (opts.systemPrompt) {
      payloadMessages.push({ role: 'system', content: opts.systemPrompt });
    }
    history.forEach(function (m) {
      payloadMessages.push({ role: m.role, content: m.content });
    });

    var headers = { 'Content-Type': 'application/json' };
    if (opts.apiKey) headers['Authorization'] = 'Bearer ' + opts.apiKey;

    fetch(opts.endpoint, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({
        model: opts.model,
        messages: payloadMessages,
        stream: false
      })
    })
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (data) {
        typingEl.remove();
        var reply = extractReply(data);
        if (!reply) throw new Error('No reply in response');
        history.push({ role: 'assistant', content: reply });
        saveHistory(history);
        addBubble('assistant', reply);
      })
      .catch(function (err) {
        typingEl.remove();
        console.error('[llm-chat-widget]', err);
        addBubble('error', "Sorry — I'm having trouble connecting right now. Please try again in a moment.");
      })
      .finally(function () {
        setSending(false);
      });
  }

  function extractReply(data) {
    try {
      if (data.choices && data.choices[0] && data.choices[0].message) {
        return data.choices[0].message.content;
      }
      // Fallback for servers that return Ollama's native shape.
      if (data.message && data.message.content) {
        return data.message.content;
      }
    } catch (e) {}
    return null;
  }

  var sendBtn = inputRow.querySelector('.cw-send');
  var textarea = inputRow.querySelector('.cw-input');

  function setSending(isSending) {
    sendBtn.disabled = isSending;
    textarea.disabled = isSending;
  }

  function handleSend() {
    var text = textarea.value.trim();
    if (!text) return;
    textarea.value = '';
    autoGrow();
    sendMessage(text);
  }

  function autoGrow() {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 90) + 'px';
  }

  sendBtn.addEventListener('click', handleSend);
  textarea.addEventListener('input', autoGrow);
  textarea.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  launcher.addEventListener('click', function () {
    panel.classList.toggle('cw-open');
    if (panel.classList.contains('cw-open')) {
      textarea.focus();
    }
  });

  header.querySelector('.cw-close').addEventListener('click', function () {
    panel.classList.remove('cw-open');
  });
})();
