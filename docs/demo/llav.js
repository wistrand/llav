// llav in the browser: the same typed questions, prompt and readout as the llav server, on llama.cpp compiled
// to WebAssembly (wllama). A port of src/llav/prompt.py and src/llav/questions.py; keep the two in step.
//
// What differs from the server: every question re-reads the text (no shared state), there is no native
// helper, and the model is whatever GGUF the page loads. The prompt and the readout are the same: the chat
// template with thinking off, one forward pass, the log-probabilities of the answer letters A-Z softmaxed
// over the declared options, and the candidate mass from the same vocabulary-normalized log-probabilities.

export const WLLAMA_VERSION = "3.6.1";
const WLLAMA_BASE = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_VERSION}/esm`;

export const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
export const SYSTEM =
  "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. " +
  "Respond with only its uppercase letter, with no explanation or reasoning.";
const MAX_LEVELS = 10;
// How many of the most likely next tokens to ask for. A declared label outside them gets the smallest
// returned log-probability, an upper bound, as on the server.
const TOP_LOGPROBS = 128;

export class ValidationError extends Error {
  constructor(loc, message) {
    super(`${loc.join(".")}: ${message}`);
    this.loc = loc;
  }
}

// Python's json.dumps(value, ensure_ascii=False): ", " and ": " separators, which JSON.stringify omits. The
// prompt must match the server's byte for byte. Known gaps: a float with no fraction (1.0) prints as 1, and
// JSON.parse puts integer-like object keys first in ascending order, so a state or criteria object keyed
// "2", "1" reaches the model in a different order than on the server, where dicts keep the caller's order.
export function pyJson(value) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(pyJson).join(", ") + "]";
  return "{" + Object.entries(value).map(([k, v]) => JSON.stringify(k) + ": " + pyJson(v)).join(", ") + "}";
}

export function messages(state, criterion, descriptions) {
  if (descriptions.length < 2 || descriptions.length > LABELS.length) {
    throw new Error(`A decision needs 2-${LABELS.length} options`);
  }
  const payload = {
    evidence: state,
    criterion,
    options: descriptions.map((description, index) => ({ letter: LABELS[index], description })),
  };
  return [{ role: "system", content: SYSTEM }, { role: "user", content: pyJson(payload) }];
}

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const nonEmpty = (v) =>
  typeof v === "string" ? v.trim().length > 0 : (Array.isArray(v) ? v.length > 0 : isObject(v) && Object.keys(v).length > 0);

function foldNoul(instructions, criteria) {
  const answers = { true: "Yes", false: "No" };
  const rules = ["true", "false"].filter((side) => nonEmpty(criteria[side])).map((side) => [side, criteria[side]]);
  if (!rules.length) return instructions;
  if (typeof instructions === "string" && rules.every(([, v]) => typeof v === "string")) {
    return [instructions.trim(), ...rules.map(([side, v]) => `Answer ${answers[side]} when: ${v.trim()}`)].join("\n\n");
  }
  const folded = { criterion: instructions };
  for (const [side, v] of rules) folded[`answer_${answers[side].toLowerCase()}_when`] = v;
  return folded;
}

function describe(label, value) {
  if (value === null || value === undefined) return label;
  if (typeof value === "string") return value.trim() ? `${label}: ${value}` : label;
  return { option: label, description: value };
}

function alignLetters(keys) {
  const slots = new Array(keys.length).fill(null);
  const placed = new Set();
  for (const exact of [true, false]) {
    for (const key of keys) {
      if (key.length !== 1 || placed.has(key) || (exact && !LABELS.includes(key))) continue;
      const index = LABELS.indexOf(key.toUpperCase());
      if (index >= 0 && index < keys.length && slots[index] === null) { slots[index] = key; placed.add(key); }
    }
  }
  const rest = keys.filter((key) => !placed.has(key));
  return slots.map((key) => (key !== null ? key : rest.shift()));
}

function checkValue(value, loc, allowNull = false) {
  if (value === null && allowNull) return;
  if (!(typeof value === "string" || Array.isArray(value) || isObject(value))) {
    throw new ValidationError(loc, "must be a string, object, or array");
  }
}

export function parseQuestion(key, raw) {
  const loc = ["questions", key];
  if (!isObject(raw)) throw new ValidationError(loc, "question must be an object");
  const kind = raw.type;
  if (!["noul", "choice", "score"].includes(kind)) throw new ValidationError([...loc, "type"], "type must be 'noul', 'choice', or 'score'");
  const unknown = Object.keys(raw).filter((k) => !["type", "instructions", "criteria"].includes(k)).sort();
  if (unknown.length) throw new ValidationError([...loc, unknown[0]], "unknown field");
  const instructions = raw.instructions;
  if (!nonEmpty(instructions)) throw new ValidationError([...loc, "instructions"], "instructions must be a nonempty string, object, or array");
  let criteria = raw.criteria;
  if (kind === "noul") {
    if (criteria === undefined || criteria === null) criteria = {};
    if (!isObject(criteria) || Object.keys(criteria).some((k) => k !== "true" && k !== "false")) {
      throw new ValidationError([...loc, "criteria"], "noul criteria may only contain 'true' and 'false'");
    }
    for (const side of ["true", "false"]) if (side in criteria) checkValue(criteria[side], [...loc, "criteria", side], true);
    return { key, type: kind, instructions: foldNoul(instructions, criteria), optionIds: ["true", "false"],
      descriptions: [describe("Yes", criteria.true ?? null), describe("No", criteria.false ?? null)] };
  }
  if (kind === "choice") {
    if (!isObject(criteria) || Object.keys(criteria).length < 2) {
      throw new ValidationError([...loc, "criteria"], "choice criteria must map at least two options to descriptions");
    }
    const keys = Object.keys(criteria);
    if (keys.length > LABELS.length) throw new ValidationError([...loc, "criteria"], `at most ${LABELS.length} options per choice`);
    for (const k of keys) {
      if (!k) throw new ValidationError([...loc, "criteria"], "option keys must be nonempty");
      checkValue(criteria[k], [...loc, "criteria", k], true);
    }
    const order = alignLetters(keys);
    return { key, type: kind, instructions, optionIds: order, descriptions: order.map((k) => describe(k, criteria[k])),
      declared: order.join("\u0000") !== keys.join("\u0000") ? keys : null };
  }
  if (!Array.isArray(criteria) || criteria.length < 2 || criteria.length > MAX_LEVELS) {
    throw new ValidationError([...loc, "criteria"], `score criteria must be an array of 2-${MAX_LEVELS} levels`);
  }
  criteria.forEach((level, i) => {
    checkValue(level, [...loc, "criteria", i]);
    if (!nonEmpty(level)) throw new ValidationError([...loc, "criteria", i], "levels must be nonempty");
  });
  return { key, type: kind, instructions, optionIds: criteria.map((_, i) => String(i)), descriptions: criteria,
    legend: criteria.map((level) => (typeof level === "string" ? level : pyJson(level))) };
}

export function parseRequest(body) {
  if (!isObject(body)) throw new ValidationError(["body"], "request body must be a JSON object");
  const unknown = Object.keys(body).filter((k) => !["state", "model", "questions"].includes(k)).sort();
  if (unknown.length) throw new ValidationError([unknown[0]], "unknown field");
  if (!nonEmpty(body.state)) throw new ValidationError(["state"], "state must be a nonempty string, object, or array");
  if (!isObject(body.questions) || !Object.keys(body.questions).length) {
    throw new ValidationError(["questions"], "questions must be a nonempty object");
  }
  return { state: body.state, questions: Object.entries(body.questions).map(([k, q]) => parseQuestion(k, q)) };
}

export function confidence(probabilities) {
  const entropy = -probabilities.reduce((sum, p) => (p > 0 ? sum + p * Math.log(p) : sum), 0);
  return Math.max(0, Math.min(1, 1 - entropy / Math.log(probabilities.length)));
}

export function buildAnswer(question, probabilities) {
  if (question.type === "noul") return { type: "noul", noul: probabilities[0] };
  const byId = Object.fromEntries(question.optionIds.map((id, i) => [id, probabilities[i]]));
  if (question.type === "choice") {
    const best = probabilities.indexOf(Math.max(...probabilities));
    const ordered = question.declared ? Object.fromEntries(question.declared.map((k) => [k, byId[k]])) : byId;
    return { type: "choice", choice: question.optionIds[best], probabilities: ordered, confidence: confidence(probabilities) };
  }
  return {
    type: "score",
    score: probabilities.reduce((sum, p, i) => sum + i * p, 0),
    legend: Object.fromEntries(question.optionIds.map((id, i) => [id, question.legend[i]])),
    probabilities: byId,
    confidence: confidence(probabilities),
  };
}

function softmax(values) {
  const top = Math.max(...values);
  const weights = values.map((v) => Math.exp(v - top));
  const total = weights.reduce((a, b) => a + b, 0);
  return weights.map((w) => w / total);
}

export class LlavBrowser {
  constructor(logger) {
    this.wllama = null;
    this.logger = logger;
    this.gpu = null;  // after load: whether the model runs on WebGPU
  }

  // Whether the browser exposes WebGPU at all. An adapter can still be refused (headless, blocked, no
  // driver), in which case wllama runs on the CPU without an error; `gpu` after `load` says what happened.
  static webgpu() {
    return typeof navigator !== "undefined" && !!navigator.gpu;
  }

  // onStage(stage, detail) reports "runtime", "cache", "download" (detail {loaded, total}), "cached", "memory"
  // and, while the model loads, "log" with each llama.cpp log line, the only sign of progress in that phase.
  async load(url, onStage = () => {}) {
    onStage("runtime");
    const { Wllama, LoggerWithoutDebug } = await import(`${WLLAMA_BASE}/index.js`);
    if (this.wllama) await this.wllama.exit();
    this.gpu = LlavBrowser.webgpu() ? !!(await navigator.gpu.requestAdapter().catch(() => null)) : false;
    const base = this.logger ?? LoggerWithoutDebug;
    const seen = (line) => {
      if (line.includes("ggml_webgpu") && /fail|error|not available/i.test(line)) this.gpu = false;
      onStage("log", line);
    };
    const forward = (level) => (...args) => { base[level](...args); seen(args.join(" ")); };
    // llama.cpp's info lines arrive as debug: shown on the page, kept out of the console.
    const logger = { debug: (...args) => { base.debug(...args); seen(args.join(" ")); },
      log: forward("log"), warn: forward("warn"), error: forward("error") };
    this.wllama = new Wllama({ default: `${WLLAMA_BASE}/wasm/wllama.wasm` }, { logger });
    onStage("cache");
    // Keep one model cached. A download that outgrows the browser's storage quota is cut short without an
    // error, and wllama would load the truncated file; freeing the other models first and checking the size
    // after turns that into a message.
    const manager = this.wllama.modelManager;
    const cached = await manager.getModels({ includeInvalid: true });
    for (const model of cached) if (model.url !== url || model.validate() !== "valid") await model.remove();
    const hit = cached.find((m) => m.url === url && m.validate() === "valid");
    const model = hit ?? await manager.downloadModel({ url }, { progressCallback: (p) => onStage("download", p) });
    if (hit) onStage("cached", { loaded: hit.size, total: hit.size });
    if (model.validate() !== "valid") {
      await model.remove();
      const { quota } = await navigator.storage.estimate();
      throw new Error(`The browser stopped saving the model part way. It allows this page about ` +
        `${(quota / 1e9).toFixed(1)} GB of storage; free disk space or pick the smaller model.`);
    }
    onStage("memory");
    await this.wllama.loadModel(model, { n_ctx: 4096 });
  }

  // One forward pass: the next token's log-probabilities, read for the declared letters.
  async readout(turns, count) {
    const response = await this.wllama.createChatCompletion({
      messages: turns,
      max_tokens: 1,
      temperature: -1,  // a plain softmax over the raw logits, as on the server
      logprobs: true,
      top_logprobs: TOP_LOGPROBS,
      chat_template_kwargs: { enable_thinking: false },
      cache_prompt: false,  // a recurrent model cannot roll back a partial match; see agent_docs/gotchas.md
    });
    const entries = response.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs ?? [];
    // Two token ids can decode to the same text; keep the likelier one, as the server's id-based readout does.
    const top = new Map();
    for (const e of entries) if (!top.has(e.token) || e.logprob > top.get(e.token)) top.set(e.token, e.logprob);
    const found = [...LABELS.slice(0, count)].map((label) => top.get(label));
    if (!found.some((v) => v !== undefined)) {
      const likely = entries.slice(0, 3).map((e) => JSON.stringify(e.token)).join(", ");
      throw new Error(`No answer letter among the model's likely next tokens (it wanted ${likely}).`);
    }
    const floor = Math.min(...top.values());
    const logprobs = found.map((v) => (v === undefined ? floor : v));
    const mass = Math.min(1, logprobs.reduce((sum, v) => sum + Math.exp(v), 0));
    return { probabilities: softmax(logprobs), mass, tokens: response.usage?.prompt_tokens ?? 0 };
  }

  // The System One call: {state, questions} in, {model, answers, usage} out, plus the candidate mass.
  // onAnswer(key, answer, mass) runs as each question is answered, for pages that show answers as they come.
  async ask(body, modelId = "llav-browser", onAnswer = null) {
    const { state, questions } = parseRequest(body);
    const started = performance.now();
    const answers = {};
    const candidateMass = {};
    let inputTokens = 0;
    for (const question of questions) {
      const turns = messages(state, question.instructions, question.descriptions);
      const { probabilities, mass, tokens } = await this.readout(turns, question.descriptions.length);
      answers[question.key] = buildAnswer(question, probabilities);
      candidateMass[question.key] = mass;
      inputTokens += tokens;
      if (onAnswer) onAnswer(question.key, answers[question.key], mass);
    }
    return {
      response: { model: modelId, answers, usage: { input_tokens: inputTokens, output_tokens: 0 } },
      candidateMass,
      seconds: (performance.now() - started) / 1000,
    };
  }
}
