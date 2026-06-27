/**
 * 小t Agent — SSE 聊天客户端
 */

(function () {
  "use strict";

  const DOM = {
    messages: document.getElementById("messages"),
    form: document.getElementById("chat-form"),
    input: document.getElementById("chat-input"),
    sendBtn: document.getElementById("send-btn"),
    resetBtn: document.getElementById("reset-btn"),
    statusBar: document.getElementById("status-bar"),
  };

  let isStreaming = false;
  let abortController = null;

  // ── 工具函数 ──

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function scrollToBottom() {
    const container = document.getElementById("chat-container");
    requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
    });
  }

  // ── 消息渲染 ──

  function addMessage(role, content, isStreaming = false) {
    const existingStreaming = document.querySelector(
      ".message.assistant .bubble.streaming"
    );
    if (isStreaming && existingStreaming) {
      existingStreaming.textContent = content;
      scrollToBottom();
      return;
    }

    // Remove welcome if it exists
    const welcome = document.querySelector(".welcome-msg");
    if (welcome) welcome.remove();

    const div = document.createElement("div");
    div.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "🐱" : "🤖";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (isStreaming) bubble.classList.add("streaming");
    bubble.textContent = content;

    div.appendChild(avatar);
    div.appendChild(bubble);
    DOM.messages.appendChild(div);
    scrollToBottom();
    return bubble;
  }

  function updateStreaming(content) {
    const bubble = document.querySelector(".message.assistant .bubble.streaming");
    if (bubble) {
      bubble.textContent = content;
      scrollToBottom();
    }
  }

  function finalizeStreaming() {
    const bubble = document.querySelector(".message.assistant .bubble.streaming");
    if (bubble) {
      bubble.classList.remove("streaming");
    }
  }

  function addStepMsg(content) {
    const div = document.createElement("div");
    div.className = "step-msg";
    div.innerHTML = `<div class="spinner"></div><span>${escapeHtml(content)}</span>`;
    DOM.messages.appendChild(div);
    scrollToBottom();
    return div;
  }

  function addToolCallMsg(content) {
    const div = document.createElement("div");
    div.className = "tool-call-msg";
    div.textContent = `🔧 ${content}`;
    DOM.messages.appendChild(div);
    scrollToBottom();
  }

  function addErrorMsg(content) {
    const div = document.createElement("div");
    div.className = "error-msg";
    div.textContent = `❌ ${content}`;
    DOM.messages.appendChild(div);
    scrollToBottom();
  }

  function setStatus(text) {
    if (DOM.statusBar) {
      DOM.statusBar.innerHTML = text
        ? `<span class="thinking-dots">${escapeHtml(text)}</span>`
        : "";
    }
  }

  // ── SSE 流式聊天 ──

  async function sendMessage(message) {
    if (isStreaming || !message.trim()) return;

    isStreaming = true;
    DOM.sendBtn.disabled = true;
    DOM.input.disabled = true;

    // 显示用户消息
    addMessage("user", message.trim());

    // 清空输入
    DOM.input.value = "";
    DOM.input.style.height = "auto";

    let assistantContent = "";
    const stepElements = [];
    let hasContent = false;

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message.trim() }),
      });

      if (!response.ok) {
        addErrorMsg(`请求失败 (${response.status})`);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;

          const payload = line.slice(6).trim();

          // End marker
          if (payload === "[DONE]") {
            continue;
          }

          try {
            const event = JSON.parse(payload);
            switch (event.type) {
              case "text":
                if (!hasContent) {
                  addMessage("assistant", event.content, true);
                  hasContent = true;
                } else {
                  updateStreaming(event.content);
                }
                assistantContent = event.content;
                break;

              case "step":
                const el = addStepMsg(event.content);
                stepElements.push(el);
                break;

              case "tool_call":
                addToolCallMsg(event.content);
                break;

              case "error":
                addErrorMsg(event.content);
                break;

              case "done":
                // done
                break;
            }
          } catch (e) {
            // skip invalid JSON
          }
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        addErrorMsg(`连接错误: ${err.message}`);
      }
    } finally {
      // 清理 step 元素
      stepElements.forEach((el) => el.remove());

      if (hasContent) {
        finalizeStreaming();
      } else if (!document.querySelector(".error-msg")) {
        // No content received, show empty response
      }

      isStreaming = false;
      DOM.sendBtn.disabled = false;
      DOM.input.disabled = false;
      DOM.input.focus();
      setStatus("");
      scrollToBottom();
    }
  }

  // ── 加载历史 ──

  async function loadHistory() {
    try {
      const res = await fetch("/api/history");
      const data = await res.json();
      const history = data.history || [];

      // Remove welcome
      const welcome = document.querySelector(".welcome-msg");
      if (welcome) welcome.remove();

      for (const msg of history) {
        const role = msg.role === "user" ? "user" : "assistant";
        addMessage(role, msg.content);
      }
    } catch (e) {
      // Silent fail — history is best-effort
    }
  }

  // ── 重置对话 ──

  async function resetChat() {
    if (isStreaming) return;
    try {
      await fetch("/api/reset", { method: "POST" });
      DOM.messages.innerHTML = `
        <div class="welcome-msg">
          <span class="cat-big">🐱</span>
          <h2>小t Agent</h2>
          <p>小红书内容工坊 AI 助手<br>帮你做视频、写文案、分析素材</p>
          <div class="suggestions">
            <div class="suggestion-chip" data-text="帮我做6个穿搭视频，35秒">👗 穿搭视频</div>
            <div class="suggestion-chip" data-text="帮我分析一下这些素材">📊 分析素材</div>
            <div class="suggestion-chip" data-text="换个风格，改成小清新">🌸 改风格</div>
            <div class="suggestion-chip" data-text="搜索小红书最近的爆款笔记">🔍 搜爆款</div>
          </div>
        </div>
      `;
      // Re-bind suggestion chips
      bindSuggestions();
    } catch (e) {
      addErrorMsg("重置失败");
    }
  }

  // ── 建议点击 ──

  function bindSuggestions() {
    document.querySelectorAll(".suggestion-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const text = chip.dataset.text;
        if (text) {
          DOM.input.value = text;
          DOM.input.dispatchEvent(new Event("input"));
          sendMessage(text);
        }
      });
    });
  }

  // ── 自动调整输入框 ──

  function autoResize() {
    DOM.input.style.height = "auto";
    DOM.input.style.height = Math.min(DOM.input.scrollHeight, 120) + "px";
  }

  // ── 事件绑定 ──

  DOM.form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = DOM.input.value.trim();
    if (text) sendMessage(text);
  });

  DOM.input.addEventListener("input", autoResize);

  DOM.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const text = DOM.input.value.trim();
      if (text) sendMessage(text);
    }
  });

  if (DOM.resetBtn) {
    DOM.resetBtn.addEventListener("click", resetChat);
  }

  // ── 初始化 ──

  bindSuggestions();

  // 如果浏览器不支持 EventSource 流式读取，但我们的 fetch + reader 方式兼容所有现代浏览器
  // No-op
})();
