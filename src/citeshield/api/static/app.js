// CiteShield local demo. No dependencies. Every piece of model or source text is inserted
// with textContent, never as HTML: retrieved text can be adversarial.
"use strict";

const EXAMPLES = [
  { mode: "answer", q: "Within how many hours must a controller notify a personal data breach to the supervisory authority?", tag: "GDPR Art. 33(1)" },
  { mode: "answer", q: "What conditions apply to consent, and can a data subject withdraw it?", tag: "GDPR Art. 7" },
  { mode: "compare", q: "Compare the deadlines for reporting a personal data breach under the GDPR with significant incident reporting under NIS2.", tag: "GDPR 33(1) vs NIS2 23(4)" },
  { mode: "checklist", q: "Give me a checklist of the obligations of providers of high-risk AI systems.", tag: "AI Act Art. 16" },
  { mode: "answer", q: "What is the corporate income tax rate in Ireland?", tag: "should be refused: outside the corpus" },
];

const REFUSALS = {
  insufficient_evidence: "The retrieved provisions do not support an answer.",
  out_of_scope: "The question is about law or facts outside the five regulations covered.",
  legal_advice_request: "The question asks for an assessment of a specific situation (legal advice).",
  conflicting_sources: "The retrieved provisions point in different directions.",
  injection_detected: "The request tried to change how the assistant works.",
  invalid_citations: "No answer with citations that check out against the official texts could be produced.",
  unsupported_claims: "The cited provisions do not support the statements needed to answer.",
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
};

let ready = false;
let running = false;

// ------------------------------------------------------------------ health

async function pollHealth() {
  const box = $("health");
  try {
    const h = await (await fetch("/health")).json();
    const llm = h.llm || {};
    const backend = llm.provider === "ollama"
      ? `local Ollama, ${llm.model}${llm.reachable === false ? " (not reachable!)" : llm.model_present === false ? " (model not pulled!)" : ""}`
      : `${llm.provider} API${llm.paid ? " (paid)" : ""}${llm.api_key_configured === false ? ", no key set!" : ""}`;
    box.className = "health " + (h.status === "ok" ? "ok" : h.status);
    if (h.status === "ok") {
      box.textContent = `Ready · ${backend} · corpus ${h.corpus_version}`;
      if (!ready) fillRegulations(h.regulations || []);
      ready = true;
      return;
    }
    if (h.status === "error") {
      box.textContent = `Startup failed: ${h.error}. Check the server console.`;
      return;
    }
    box.textContent = `Loading models and index… · ${backend}`;
  } catch (e) {
    box.className = "health error";
    box.textContent = "Server not reachable.";
  }
  setTimeout(pollHealth, 3000);
}

function fillRegulations(regs) {
  const box = $("regulations");
  box.replaceChildren();
  for (const r of regs) {
    const label = el("label");
    const cb = el("input");
    cb.type = "checkbox";
    cb.value = r;
    label.append(cb, " " + r);
    box.append(label);
  }
}

// ------------------------------------------------------------------ examples

function fillExamples() {
  const ul = $("examples");
  for (const ex of EXAMPLES) {
    const li = el("li");
    const b = el("button", null, ex.q);
    b.type = "button";
    b.addEventListener("click", () => {
      $("question").value = ex.q;
      document.querySelector(`input[name=mode][value=${ex.mode}]`).checked = true;
    });
    li.append(b, el("span", "tag", `${ex.mode} · ${ex.tag}`));
    ul.append(li);
  }
}

// ------------------------------------------------------------------ streaming

function step(text, warn) {
  const li = el("li", warn ? "warn" : null, text);
  $("steps").append(li);
}

function describe(kind, p) {
  switch (kind) {
    case "status":
      return p.stage === "research" ? `Researching (round ${p.round})` :
             p.stage === "draft" ? `Drafting the answer${p.attempt > 1 ? ` (attempt ${p.attempt})` : ""}` :
             p.stage === "guard" ? "Checking the question's scope" : `Stage: ${p.stage}`;
    case "guard": return `Scope check: ${p.label}`;
    case "tool_call": return `Tool: ${p.name} ${JSON.stringify(p.input)}`;
    case "tool_result":
      return p.is_error ? `Tool ${p.name} failed: ${p.error}` :
        `${p.n_chunks} provisions: ${(p.sources || []).slice(0, 5).map((s) => s.label).join(", ")}`;
    case "note": return `Note: ${p.text.slice(0, 200)}`;
    case "draft_rejected": return `Draft rejected by validation (${(p.errors || []).length} problems), repairing`;
    case "verification_failed": return `Verification failed for ${(p.failed || []).length} statement(s), revising`;
    default: return null;
  }
}

async function ask(event) {
  event.preventDefault();
  if (running) return;
  const question = $("question").value.trim();
  if (question.length < 3) return;
  const mode = document.querySelector("input[name=mode]:checked").value;
  const regulations = [...document.querySelectorAll("#regulations input:checked")].map((c) => c.value);

  running = true;
  $("submit").disabled = true;
  $("steps").replaceChildren();
  $("progress").hidden = false;
  $("result").hidden = true;
  const t0 = Date.now();
  const timer = setInterval(() => { $("elapsed").textContent = `${Math.round((Date.now() - t0) / 1000)} s`; }, 1000);
  step("Sent. Local models can take a few minutes per answer.");

  try {
    const resp = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode, regulations: regulations.length ? regulations : null }),
    });
    if (!resp.ok) {
      let detail = resp.statusText;
      try { detail = (await resp.json()).detail; } catch (_) { /* not JSON */ }
      showError(typeof detail === "string" ? detail : JSON.stringify(detail), resp.status);
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finished = false;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const kind = (block.match(/^event: (.*)$/m) || [])[1];
        const data = (block.match(/^data: (.*)$/m) || [])[1];
        if (!kind || data === undefined) continue;
        const payload = JSON.parse(data);
        if (kind === "final") { showAnswer(payload); finished = true; }
        else if (kind === "error") { showError(payload.message); finished = true; }
        else {
          const text = describe(kind, payload);
          if (text) step(text, kind === "draft_rejected" || kind === "verification_failed");
        }
      }
    }
    if (!finished) showError("The stream ended without an answer. Check the server console.");
  } catch (e) {
    showError(`Request failed: ${e.message}`);
  } finally {
    clearInterval(timer);
    running = false;
    $("submit").disabled = false;
  }
}

// ------------------------------------------------------------------ rendering

function showError(message, code) {
  const r = $("result");
  r.replaceChildren(el("h2", null, "Something went wrong"));
  r.append(el("span", "status error", code ? `HTTP ${code}` : "error"));
  r.append(el("p", null, message));
  if (code === 503) r.append(el("p", "muted", "The server is still loading or failed to start; see the status box above."));
  if (code === 429) r.append(el("p", "muted", "Only one question runs at a time on a local machine."));
  r.hidden = false;
}

function citationNode(c, sources) {
  const src = sources[c.chunk_id] || {};
  const wrap = el("span", "cite");
  const a = el("a", null, src.label || c.chunk_id);
  a.href = "#";
  a.title = c.chunk_id;
  a.addEventListener("click", (e) => { e.preventDefault(); openSource(c.chunk_id); });
  wrap.append("Source: ", a);
  if (src.url) {
    const off = el("a", null, "official text");
    off.href = src.url;
    off.target = "_blank";
    off.rel = "noopener noreferrer";
    wrap.append(" · ", off);
  }
  wrap.append(el("blockquote", null, `“${c.quote}”`));
  return wrap;
}

function showAnswer(p) {
  const a = p.answer;
  const sources = p.sources || {};
  const r = $("result");
  r.replaceChildren(el("h2", null, "Answer"));
  r.append(el("span", `status ${a.status}`, a.status));

  if (a.status === "refused") {
    const box = el("div", "refusal");
    box.append(el("p", null, a.summary));
    box.append(el("p", "muted", `Reason: ${a.refusal_reason} — ${REFUSALS[a.refusal_reason] || ""}`));
    r.append(box);
  } else {
    r.append(el("p", "summary", a.summary));
    if (a.status === "partial") {
      r.append(el("p", "muted", `Partial answer${a.refusal_reason ? ` (${a.refusal_reason})` : ""}: ` +
        "some statements could not be verified and were removed, or the sources cover only part of the question."));
    }
    if (a.claims && a.claims.length) {
      r.append(el("h3", null, "Claims"));
      const ol = el("ol", "claims");
      for (const claim of a.claims) {
        const li = el("li", null, claim.text);
        for (const c of claim.citations) li.append(citationNode(c, sources));
        ol.append(li);
      }
      r.append(ol);
    }
    if (a.checklist && a.checklist.length) {
      r.append(el("h3", null, "Checklist"));
      const ul = el("ul", "checklist");
      for (const item of a.checklist) {
        const li = el("li", null, `☐ ${item.obligation}`);
        li.append(el("div", "muted", `Applies to: ${item.applies_to}`));
        for (const c of item.citations) li.append(citationNode(c, sources));
        ul.append(li);
      }
      r.append(ul);
    }
  }
  if (a.caveats && a.caveats.length) {
    const cv = el("div", "caveats");
    cv.append(el("h3", null, "Qualifications"));
    const ul = el("ul");
    for (const c of a.caveats) ul.append(el("li", null, c));
    cv.append(ul);
    r.append(cv);
  }
  const v = p.verification || {};
  const failed = (v.claims || []).filter((c) => c.verdict !== "supported").length;
  const ms = (p.timings_ms || {}).total;
  r.append(el("p", "meta",
    `Verified claims: ${(v.claims || []).length - failed}/${(v.claims || []).length}` +
    (failed ? ` (${failed} removed)` : "") +
    ` · guard: ${p.guard_label || "n/a"}` +
    ` · LLM calls: ${(p.usage || {}).llm_calls ?? "?"}` +
    (ms ? ` · ${(ms / 1000).toFixed(0)} s` : "") +
    ` · corpus ${a.corpus_version || ""}`));
  r.hidden = false;
}

async function openSource(chunkId) {
  const dlg = $("source-dialog");
  $("source-title").textContent = chunkId;
  $("source-meta").textContent = "Loading…";
  $("source-text").textContent = "";
  $("source-link").removeAttribute("href");
  dlg.showModal();
  try {
    const resp = await fetch(`/sources/${encodeURIComponent(chunkId).replace(/%2F/g, "/")}`);
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const s = await resp.json();
    $("source-title").textContent = s.label;
    $("source-meta").textContent = `${s.regulation} · CELEX ${s.celex} · ${s.version} · ${s.chunk_id}`;
    $("source-text").textContent = s.text;
    $("source-link").href = s.url;
  } catch (e) {
    $("source-meta").textContent = `Could not load the source: ${e.message}`;
  }
}

$("source-close").addEventListener("click", () => $("source-dialog").close());
$("ask-form").addEventListener("submit", ask);
fillExamples();
pollHealth();
