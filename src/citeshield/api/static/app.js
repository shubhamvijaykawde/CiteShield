// CiteShield workbench. No dependencies, no network resources beyond this server.
// Security: every piece of model output and legal-source text is inserted with textContent
// or as a text node, never parsed as HTML, because retrieved text can be adversarial. Links
// are only rendered for https://eur-lex.europa.eu URLs.
"use strict";

const EXAMPLES = [
  { mode: "answer", q: "Within how many hours must a controller notify a personal data breach to the supervisory authority?", tag: "GDPR Art. 33(1)" },
  { mode: "answer", q: "What conditions apply to consent, and can a data subject withdraw it?", tag: "GDPR Art. 7" },
  { mode: "compare", q: "Compare the deadlines for reporting a personal data breach under the GDPR with significant incident reporting under NIS2.", tag: "GDPR 33(1) · NIS2 23(4)" },
  { mode: "checklist", q: "Give me a checklist of the obligations of providers of high-risk AI systems.", tag: "AI Act Art. 16" },
  { mode: "answer", q: "What is the corporate income tax rate in Ireland?", tag: "Out of scope: refused" },
];

const REFUSALS = {
  insufficient_evidence: ["Not enough evidence", "The provisions retrieved do not support an answer to this question."],
  out_of_scope: ["Outside the covered law", "The question concerns law or facts outside the five regulations CiteShield covers."],
  legal_advice_request: ["Request for legal advice", "The question asks for an assessment of a specific situation. CiteShield explains what the regulations require; it does not advise on individual cases."],
  conflicting_sources: ["Conflicting provisions", "The retrieved provisions point in different directions, so no single answer is given."],
  injection_detected: ["Instruction-like request", "The request tried to change how the assistant works, so it was not acted on."],
  invalid_citations: ["Citations did not check out", "No answer could be produced whose citations match the official texts."],
  unsupported_claims: ["Claims not supported", "The cited provisions do not support the statements needed to answer."],
};

const MODES = {
  answer: { title: "Answer", hint: "Ask a question about the regulations.", placeholder: "e.g. Within how many hours must a controller notify a personal data breach?" },
  compare: { title: "Comparison", hint: "Compare provisions, within one regulation or across two.", placeholder: "e.g. Compare breach notification under the GDPR with incident reporting under NIS2." },
  checklist: { title: "Compliance checklist", hint: "List the obligations that apply to a role.", placeholder: "e.g. Give me a checklist of the obligations of providers of high-risk AI systems." },
};

const ACTS = {
  GDPR: "General Data Protection Regulation",
  "AI Act": "Artificial Intelligence Act",
  DSA: "Digital Services Act",
  NIS2: "NIS 2 Directive",
  "Data Act": "Data Act",
};
const DISCLAIMER = "General information from the EU legal texts, not legal advice.";

const $ = (id) => document.getElementById(id);
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let serverState = "checking";
let running = false;
let lastRequest = null;
let timer = null;

// ------------------------------------------------------------------ safety helpers

function safeEurLex(url) {
  // Only official EUR-Lex pages over https become links; anything else is dropped.
  try {
    const u = new URL(String(url));
    return u.protocol === "https:" && u.hostname === "eur-lex.europa.eu" ? u.href : null;
  } catch (_) {
    return null;
  }
}

function regulationOf(label) {
  const m = /^(.*?)\s+(?:Art\.|Annex\b|Recital\b)/.exec(label || "");
  return m ? m[1] : label || "Source";
}

function celexAct(celex) {
  // 32016R0679 -> Regulation (EU) 2016/679, 32022L2555 -> Directive (EU) 2022/2555.
  const m = /^3(\d{4})([RL])(\d{4})$/.exec(celex || "");
  if (!m) return null;
  return `${m[2] === "R" ? "Regulation" : "Directive"} (EU) ${m[1]}/${Number(m[3])}`;
}

function actName(regulation, celex) {
  const formal = celexAct(celex);
  const short = ACTS[regulation];
  if (formal && short) return `${formal} · ${short}`;
  return formal || short || regulation;
}

const celexOf = (chunkId) => String(chunkId).split(":")[0];

// ------------------------------------------------------------------ system status

function setStatus(state, text) {
  serverState = state;
  const box = $("system-status");
  box.dataset.state = state;
  box.querySelector(".status-text").textContent = text;
}

function describeBackend(llm) {
  if (!llm) return "";
  if (llm.provider === "ollama") {
    if (llm.reachable === false) return "Ollama not reachable";
    if (llm.model_present === false) return `${llm.model} not pulled`;
    return `Local · ${llm.model}`;
  }
  const key = llm.api_key_configured === false ? ", no key" : "";
  return `${llm.provider} API${llm.paid ? " (paid)" : ""}${key}`;
}

async function pollHealth() {
  let delay = 3000;
  try {
    const resp = await fetch("/health", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const h = await resp.json();
    const backend = describeBackend(h.llm);
    const llmProblem = h.llm && h.llm.provider === "ollama" && (h.llm.reachable === false || h.llm.model_present === false);
    if (h.status === "ok" && !llmProblem) {
      setStatus("ready", `Ready · ${backend}`);
      fillRegulations(h.regulations || []);
      delay = 20000;
    } else if (h.status === "ok") {
      setStatus("error", `Models loaded, but ${backend}`);
      fillRegulations(h.regulations || []);
    } else if (h.status === "error") {
      setStatus("error", `Startup failed: ${h.error || "see the server console"}`);
      delay = 10000;
    } else {
      setStatus("loading", `Loading models and index… · ${backend}`);
    }
  } catch (_) {
    setStatus("offline", "Local server not reachable");
    delay = 5000;
  }
  setTimeout(pollHealth, delay);
}

let regulationsFilled = false;
function fillRegulations(regs) {
  if (regulationsFilled || !regs.length) return;
  regulationsFilled = true;
  const box = $("regulations");
  box.replaceChildren();
  for (const r of regs) {
    const label = el("label", "chip");
    const input = el("input");
    input.type = "checkbox";
    input.value = r;
    input.name = "regulation";
    label.append(input, el("span", null, r));
    label.title = ACTS[r] || r;
    box.append(label);
  }
}

// ------------------------------------------------------------------ composer

function currentMode() {
  return document.querySelector("input[name=mode]:checked").value;
}

function syncMode() {
  const m = MODES[currentMode()];
  $("mode-hint").textContent = m.hint;
  $("question").placeholder = m.placeholder;
}

function updateCount() {
  $("char-count").textContent = `${$("question").value.length} / 2000`;
}

function fillExamples() {
  const ul = $("examples");
  for (const ex of EXAMPLES) {
    const li = el("li");
    const b = el("button", "example");
    b.type = "button";
    const refused = ex.tag.startsWith("Out of scope");
    const tag = el("span", `tag tag-${refused ? "refusal" : ex.mode}`, refused ? "refusal" : ex.mode);
    const meta = el("span", "example-meta");
    meta.append(tag, el("span", null, ex.tag));
    b.append(el("span", "example-q", ex.q), meta);
    b.addEventListener("click", () => {
      $("question").value = ex.q;
      document.querySelector(`input[name=mode][value=${ex.mode}]`).checked = true;
      syncMode();
      updateCount();
      $("question").focus();
    });
    li.append(b);
    ul.append(li);
  }
}

function setBusy(busy) {
  running = busy;
  const btn = $("submit");
  btn.disabled = busy;
  btn.setAttribute("aria-busy", busy ? "true" : "false");
  btn.querySelector(".btn-label").textContent = busy ? "Researching…" : "Research";
  $("result").setAttribute("aria-busy", busy ? "true" : "false");
}

// ------------------------------------------------------------------ trace

const STAGES = { guard: "Checking scope", research: "Researching", draft: "Drafting the answer" };

function step(title, detail, cls) {
  for (const li of $("steps").querySelectorAll("li.current")) li.classList.remove("current");
  const li = el("li", cls || "current");
  li.append(el("span", null, title));
  if (detail) li.append(el("span", "step-detail", detail));
  $("steps").append(li);
}

function finishSteps(failed) {
  const items = $("steps").querySelectorAll("li");
  items.forEach((li) => li.classList.remove("current"));
  if (failed && items.length) items[items.length - 1].classList.add("fail");
}

function traceEvent(kind, p) {
  switch (kind) {
    case "status": {
      const extra = p.stage === "research" ? `round ${p.round}` : p.stage === "draft" && p.attempt > 1 ? `attempt ${p.attempt}` : "";
      step(STAGES[p.stage] || `Stage: ${p.stage}`, extra);
      break;
    }
    case "guard":
      step("Scope check", p.label === "in_scope" ? "in scope" : p.label, p.label === "in_scope" ? "" : "warn");
      break;
    case "tool_call":
      step(`Tool: ${p.name}`, p.input && p.input.query ? `“${p.input.query}”` : JSON.stringify(p.input));
      break;
    case "tool_result":
      if (p.is_error) step(`Tool ${p.name} failed`, p.error, "warn");
      else step(`${p.n_chunks} provisions found`, (p.sources || []).slice(0, 4).map((s) => s.label).join(" · "), "");
      break;
    case "note":
      step("Research note", String(p.text || "").slice(0, 160), "");
      break;
    case "draft_rejected":
      step("Draft rejected by validation", `${(p.errors || []).length} problem(s); repairing`, "warn");
      break;
    case "verification_failed":
      step("Verification failed", `${(p.failed || []).length} statement(s); revising`, "warn");
      break;
    default:
      break;
  }
}

// ------------------------------------------------------------------ request

async function submit(event) {
  if (event) event.preventDefault();
  if (running) return;
  const question = $("question").value.trim();
  if (question.length < 3) {
    $("question").focus();
    return;
  }
  const regulations = [...document.querySelectorAll("#regulations input:checked")].map((c) => c.value);
  lastRequest = { question, mode: currentMode(), regulations: regulations.length ? regulations : null };
  run(lastRequest);
}

async function run(req) {
  setBusy(true);
  $("workspace").hidden = false;
  $("examples-section").hidden = true;
  $("how").hidden = true;
  $("steps").replaceChildren();
  showPending(req);
  const t0 = Date.now();
  clearInterval(timer);
  timer = setInterval(() => { $("elapsed").textContent = `${Math.round((Date.now() - t0) / 1000)} s`; }, 1000);
  step("Question sent", "Local models usually take 1–5 minutes.", "");
  let finished = false;
  try {
    const resp = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!resp.ok) {
      let detail = resp.statusText;
      try { detail = (await resp.json()).detail; } catch (_) { /* not JSON */ }
      finishSteps(true);
      showError(resp.status, typeof detail === "string" ? detail : JSON.stringify(detail));
      finished = true;
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
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
        let payload;
        try { payload = JSON.parse(data); } catch (_) { continue; }
        if (kind === "final") {
          finishSteps(false);
          step("Verified answer ready", "", "");
          finishSteps(false);
          showAnswer(payload, req);
          finished = true;
        } else if (kind === "error") {
          finishSteps(true);
          showError("run", payload.message);
          finished = true;
        } else {
          traceEvent(kind, payload);
        }
      }
    }
    if (!finished) {
      finishSteps(true);
      showError("stream", "The stream ended without an answer.");
    }
  } catch (e) {
    finishSteps(true);
    showError("network", e.message);
  } finally {
    clearInterval(timer);
    setBusy(false);
  }
}

// ------------------------------------------------------------------ result states

function resultCard(extraClass) {
  const card = el("article", `card result-card ${extraClass || ""}`.trim());
  $("result").replaceChildren(card);
  return card;
}

function questionLine(req) {
  const p = el("p", "result-question");
  p.append(el("strong", null, `${MODES[req.mode].title}: `), document.createTextNode(req.question));
  return p;
}

function showPending(req) {
  const card = resultCard("state-card");
  const head = el("div", "result-head");
  head.append(el("span", "badge badge-pending", "Researching"), el("h2", null, MODES[req.mode].title));
  card.append(head, questionLine(req));
  card.append(el("p", "muted", "Searching the official texts, drafting, then verifying every claim against its source. Progress appears in the research trace."));
  const lines = el("div", "pending-lines");
  lines.setAttribute("aria-hidden", "true");
  lines.append(el("span"), el("span"), el("span"));
  card.append(lines);
  $("workspace").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
}

function showError(code, message) {
  const titles = {
    503: ["The server is not ready", "The models or the index are still loading, or startup failed. The status in the header shows which."],
    429: ["Another question is in progress", "A local machine answers one question at a time. Wait for it to finish, then retry."],
    413: ["The question is too long", "Shorten the question and try again."],
    422: ["The request was not accepted", "Check the question and the selected regulations."],
    network: ["Could not reach the local server", "Check that the CiteShield server is still running in its terminal."],
    run: ["The research run failed", "The model or a tool failed while answering. This is a technical failure, not a refusal."],
    stream: ["The answer did not arrive", "The connection closed before an answer was produced."],
  };
  const [title, explain] = titles[code] || ["Something went wrong", "The request failed."];
  const card = resultCard("state-card");
  const head = el("div", "result-head");
  head.append(el("span", "badge badge-error", typeof code === "number" ? `HTTP ${code}` : "Error"));
  card.append(head, el("h3", null, title), el("p", null, explain));
  if (message) card.append(el("p", "detail", message));
  const actions = el("div", "state-actions");
  const retry = el("button", "btn btn-primary", "Retry");
  retry.type = "button";
  retry.addEventListener("click", () => { if (lastRequest) run(lastRequest); });
  const edit = el("button", "btn btn-secondary", "Edit question");
  edit.type = "button";
  edit.addEventListener("click", () => $("question").focus());
  actions.append(retry, edit);
  card.append(actions);
  focusResult();
}

function citationBlock(c, sources) {
  const src = sources[c.chunk_id] || {};
  const quote = el("blockquote", "quote");
  const head = el("div", "quote-head");
  head.append(el("span", "quote-label", "Official text"), citeButton(c.chunk_id, src.label));
  quote.append(head, document.createTextNode(`“${c.quote}”`));
  return quote;
}

function citeButton(chunkId, label, text) {
  const b = el("button", "cite-ref", text || label || chunkId);
  b.type = "button";
  b.title = `Inspect source ${chunkId}`;
  b.setAttribute("aria-label", `Inspect source: ${label || chunkId}`);
  b.addEventListener("click", () => openSource(chunkId));
  return b;
}

function claimList(claims, sources) {
  const ol = el("ol", "claims");
  for (const claim of claims) {
    const li = el("li", "claim");
    li.append(el("p", "claim-text", claim.text));
    const ev = el("div", "evidence");
    for (const c of claim.citations || []) ev.append(citationBlock(c, sources));
    li.append(ev);
    ol.append(li);
  }
  return ol;
}

function compareView(claims, sources) {
  const groups = new Map();
  for (const claim of claims) {
    const first = (claim.citations || [])[0];
    const reg = first ? regulationOf((sources[first.chunk_id] || {}).label) : "Other";
    const celex = first ? celexOf(first.chunk_id) : "";
    if (!groups.has(reg)) groups.set(reg, { celex, claims: [] });
    groups.get(reg).claims.push(claim);
  }
  if (groups.size < 2) return claimList(claims, sources);
  const grid = el("div", "compare-grid");
  for (const [reg, g] of groups) {
    const col = el("section", "compare-col");
    col.append(el("h3", null, reg), el("p", "act", actName(reg, g.celex)), claimList(g.claims, sources));
    grid.append(col);
  }
  return grid;
}

function checklistView(items, sources) {
  const table = el("table", "checklist");
  const thead = el("thead");
  const hr = el("tr");
  for (const h of ["", "Obligation", "Applies to"]) hr.append(el("th", null, h));
  thead.append(hr);
  const tbody = el("tbody");
  items.forEach((item, i) => {
    const tr = el("tr");
    const check = el("td", "check");
    const box = el("input");
    box.type = "checkbox";
    box.id = `item-${i}`;
    box.addEventListener("change", () => tr.classList.toggle("done", box.checked));
    check.append(box);
    const ob = el("td");
    const label = el("label", "obligation", item.obligation);
    label.htmlFor = box.id;
    const ev = el("div", "evidence");
    for (const c of item.citations || []) ev.append(citationBlock(c, sources));
    ob.append(label, ev);
    tr.append(check, ob, el("td", "applies", item.applies_to));
    tbody.append(tr);
  });
  table.append(thead, tbody);
  return table;
}

function block(title, content) {
  const s = el("section", "block");
  s.append(el("h3", "section-title", title), content);
  return s;
}

function sourcesList(answer, sources) {
  const ids = [];
  for (const claim of [...(answer.claims || []), ...(answer.checklist || [])]) {
    for (const c of claim.citations || []) if (!ids.includes(c.chunk_id)) ids.push(c.chunk_id);
  }
  if (!ids.length) return null;
  const ul = el("ul", "sources");
  for (const id of ids) {
    const src = sources[id] || {};
    const li = el("li", "source-item");
    const left = el("div");
    left.append(el("div", null, src.label || id), el("div", "act", actName(regulationOf(src.label), celexOf(id))));
    li.append(left, citeButton(id, src.label, "Inspect"));
    ul.append(li);
  }
  return ul;
}

function showRefusal(answer, payload, req) {
  const card = resultCard("refusal");
  const [title, explain] = REFUSALS[answer.refusal_reason] || ["Not answered", "CiteShield could not answer from its sources."];
  const head = el("div", "result-head");
  head.append(el("span", "badge badge-refused", "Refused"), el("h2", null, MODES[req.mode].title));
  card.append(head, questionLine(req), el("h3", "refusal-title", title), el("p", "explain", explain));
  if (answer.summary) card.append(el("p", "agent-text", answer.summary));
  const next = el("ul", "next-steps");
  next.append(
    el("li", null, "Ask about obligations, definitions or deadlines in the GDPR, AI Act, DSA, NIS2 or Data Act."),
    el("li", null, "Ask what the regulation requires of a type of organisation, rather than of your own case."),
  );
  card.append(block("What you can ask instead", next));
  card.append(metaLine(payload, answer));
}

function metaLine(payload, answer) {
  const v = payload.verification || {};
  const checks = v.claims || [];
  const failed = checks.filter((c) => c.verdict !== "supported").length;
  const ms = (payload.timings_ms || {}).total;
  const meta = el("div", "result-meta");
  if (checks.length) meta.append(el("span", null, `Verified ${checks.length - failed} of ${checks.length} claim(s)${failed ? `, ${failed} removed` : ""}`));
  meta.append(el("span", null, `Scope check: ${payload.guard_label || "n/a"}`));
  if (payload.usage && payload.usage.llm_calls !== undefined) meta.append(el("span", null, `${payload.usage.llm_calls} model call(s)`));
  if (ms) meta.append(el("span", null, `${Math.round(ms / 1000)} s`));
  if (answer.corpus_version) meta.append(el("span", null, `Corpus ${answer.corpus_version}`));
  return meta;
}

function showAnswer(payload, req) {
  const answer = payload.answer || {};
  const sources = payload.sources || {};
  if (answer.status === "refused") {
    showRefusal(answer, payload, req);
    focusResult();
    return;
  }
  const card = resultCard();
  const head = el("div", "result-head");
  const label = answer.status === "partial" ? "Partial answer" : "Answered";
  head.append(el("span", `badge badge-${answer.status === "partial" ? "partial" : "answered"}`, label), el("h2", null, MODES[req.mode].title));
  card.append(head, questionLine(req), el("p", "summary", answer.summary));

  if (answer.status === "partial") {
    card.append(el("p", "notice", "Some statements could not be verified against the cited texts and were removed, or the sources cover only part of the question. What remains is verified."));
  }
  const claims = answer.claims || [];
  const items = answer.checklist || [];
  if (items.length) card.append(block("Checklist", checklistView(items, sources)));
  if (claims.length) {
    const title = req.mode === "compare" ? "Comparison" : items.length ? "Supporting claims" : "Claims and evidence";
    card.append(block(title, req.mode === "compare" ? compareView(claims, sources) : claimList(claims, sources)));
  }
  const quals = (answer.caveats || []).filter((c) => c !== DISCLAIMER);
  if (quals.length) {
    const ul = el("ul", "qualifications");
    for (const q of quals) ul.append(el("li", null, q));
    card.append(block("Qualifications", ul));
  }
  const srcs = sourcesList(answer, sources);
  if (srcs) card.append(block("Sources cited", srcs));
  card.append(metaLine(payload, answer));
  focusResult();
}

function focusResult() {
  const r = $("result");
  r.focus({ preventScroll: true });
  r.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
}

// ------------------------------------------------------------------ source dialog

function metaRow(dl, term, value, code) {
  dl.append(el("dt", null, term));
  const dd = el("dd");
  if (code) dd.append(el("code", null, value));
  else dd.textContent = value;
  dl.append(dd);
}

async function openSource(chunkId) {
  const dlg = $("source-dialog");
  $("source-act").textContent = "Source";
  $("source-title").textContent = chunkId;
  $("source-meta").replaceChildren();
  $("source-text").textContent = "";
  $("source-status").textContent = "Loading…";
  $("source-link").hidden = true;
  $("source-link").removeAttribute("href");
  if (!dlg.open) dlg.showModal();
  try {
    const resp = await fetch(`/sources/${encodeURIComponent(chunkId).replace(/%2F/g, "/")}`);
    if (!resp.ok) {
      let detail = resp.statusText;
      try { detail = (await resp.json()).detail; } catch (_) { /* not JSON */ }
      throw new Error(detail);
    }
    const s = await resp.json();
    $("source-act").textContent = s.regulation;
    $("source-title").textContent = s.label;
    const dl = $("source-meta");
    metaRow(dl, "Legal act", actName(s.regulation, s.celex));
    metaRow(dl, "Provision", (s.heading_path || []).slice(1).join(" › ") || s.label);
    metaRow(dl, "CELEX", s.celex, true);
    metaRow(dl, "Version", s.version, true);
    metaRow(dl, "Citation id", s.chunk_id, true);
    $("source-text").textContent = s.text;
    const href = safeEurLex(s.url);
    if (href) {
      $("source-link").href = href;
      $("source-link").hidden = false;
      $("source-status").textContent = "";
    } else {
      $("source-status").textContent = "No verified EUR-Lex link for this source.";
    }
  } catch (e) {
    $("source-status").textContent = `Could not load the source: ${e.message}`;
  }
}

function closeSource() { $("source-dialog").close(); }

// ------------------------------------------------------------------ wiring

$("source-close").addEventListener("click", closeSource);
$("source-dialog").addEventListener("click", (e) => { if (e.target === e.currentTarget) closeSource(); });
$("composer").addEventListener("submit", submit);
$("question").addEventListener("input", updateCount);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(e);
});
for (const r of document.querySelectorAll("input[name=mode]")) r.addEventListener("change", syncMode);
fillExamples();
syncMode();
updateCount();
pollHealth();
