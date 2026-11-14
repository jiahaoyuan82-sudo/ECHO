import { EMOTIONS, buildEdges, loadEnsemble } from "./echo.js";

const TF_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
const EMOTION_MODEL = "SamLowe/roberta-base-go_emotions-onnx";   // ONNX export of the model used in the paper
const SENTENCE_MODEL = "Xenova/all-MiniLM-L6-v2";                // ONNX export of sentence-transformers/all-MiniLM-L6-v2
const TRANSLATE_MODEL = "Xenova/opus-mt-zh-en";

const ZH = { anger: "愤怒", disgust: "厌恶", fear: "恐惧", joy: "喜悦", sadness: "悲伤", surprise: "惊讶", neutral: "中性" };
const COLORS = { anger: "#ff6b6b", disgust: "#8bd17c", fear: "#b088f9", joy: "#ffd23f", sadness: "#5ab0f6", surprise: "#ff9f43", neutral: "#c9c9c9" };
const EKMAN = {
  anger: ["anger", "annoyance", "disapproval"], disgust: ["disgust"], fear: ["fear", "nervousness"],
  joy: ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
  sadness: ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
  surprise: ["surprise", "realization", "confusion", "curiosity"], neutral: ["neutral"],
};
const UNCERTAINTY = new Set(["reportedly", "unconfirmed", "allegedly", "rumour", "rumor", "rumours", "rumors", "apparently",
  "possibly", "perhaps", "maybe", "uncertain", "unknown", "claimed", "claims"]);
const BRADY = ("abandon* abuse* agitat* attack* bad brutal* care caring cheat* compassion* cruel* damag* damn* destroy* devil* devot* " +
  "disgust* envy* evil* faith* fault* fight* forbid* good goodness greed* gross* hate heaven* hell hero* honest* honor* hurt* ideal* " +
  "immoral* kill* liar* loyal* murder* offend* pain peace* protest punish* rebel* respect revenge* ruin* safe* save secur* shame* sin " +
  "sinister sins slut* spite* steal* victim* vile virtue* war warring wars whore* wicked* wrong* benefit* harm* suffer* value*").split(" ");
const URL_RE = /https?:\/\/\S+|www\.\S+/gi;
const MENTION_RE = /@\w+/g;
const CJK_RE = /[㐀-鿿豈-﫿]/;
const GAP_MINUTES = 5;

const EXAMPLES = {
  rumour: {
    source: { text: "BREAKING: reports say the central bridge has collapsed and many people are trapped. Unconfirmed, please share!", followers: 1200, account_age_days: 400 },
    replies: [
      { text: "Oh no, this is terrible. Praying for everyone there.", parent: 0, t: 3, followers: 85 },
      { text: "Is this real? I can't find any news source confirming it.", parent: 0, t: 5, followers: 300 },
      { text: "My cousin lives nearby and says there are sirens everywhere. I'm so scared.", parent: 2, t: 9, followers: 40 },
      { text: "Sirens don't mean the bridge collapsed. Stop spreading panic.", parent: 3, t: 15, followers: 900 },
      { text: "This is so sad, those poor families.", parent: 0, t: 20, followers: 15 },
      { text: "Heartbreaking. I hope they get everyone out.", parent: 5, t: 40, followers: 60 },
    ],
  },
  news: {
    source: { text: "City council confirms the central bridge will close for scheduled maintenance this weekend. Details: https://example.org/notice", followers: 250000, account_age_days: 3000, verified: true },
    replies: [
      { text: "Thanks for the heads up, I'll take the train.", parent: 0, t: 4 },
      { text: "Finally! That bridge needed repairs.", parent: 0, t: 9 },
      { text: "Agreed, the road surface was awful.", parent: 2, t: 15 },
      { text: "Will buses still run on Saturday?", parent: 0, t: 30 },
    ],
  },
};

const $ = s => document.querySelector(s);
let uidCounter = 0;
let lastResult = null;

// ------------------------------------------------------------------ form
function replyRows() { return [...document.querySelectorAll("#replies .post.reply")]; }

function refreshLabels() {
  const rows = replyRows();
  rows.forEach((row, k) => {
    row.querySelector(".num").textContent = `回复 ${k + 1}`;
    const sel = row.querySelector(".parent"), keep = sel.value || "src";
    sel.innerHTML = "";
    sel.add(new Option("原帖", "src"));
    rows.slice(0, k).forEach((r, j) => sel.add(new Option(`回复 ${j + 1}`, r.dataset.uid)));
    sel.value = [...sel.options].some(o => o.value === keep) ? keep : "src";
    row.querySelector(".minutes").placeholder = `自动(${(k + 1) * GAP_MINUTES})`;
  });
}

function addReply(data = {}) {
  const node = $("#reply-tpl").content.firstElementChild.cloneNode(true);
  node.dataset.uid = "u" + (++uidCounter);
  node.querySelector(".text").value = data.text || "";
  if (data.t !== undefined && data.t !== null && data.t !== "") node.querySelector(".minutes").value = data.t;
  node._extra = { followers: data.followers, verified: data.verified, account_age_days: data.account_age_days, has_media: data.has_media, has_url: data.has_url };
  node.querySelector(".del").onclick = () => {
    const parentOfDeleted = node.querySelector(".parent").value;
    replyRows().forEach(r => { const s = r.querySelector(".parent"); if (s.value === node.dataset.uid) { s.value = parentOfDeleted; s.dataset.want = parentOfDeleted; } });
    node.remove(); refreshLabels();
  };
  $("#replies").appendChild(node);
  refreshLabels();
  if (data.parentUid) node.querySelector(".parent").value = data.parentUid;
  return node;
}

function clearForm() {
  $("#source-text").value = ""; $("#src-followers").value = ""; $("#src-age").value = "";
  $("#src-verified").checked = false; $("#src-media").checked = false;
  $("#replies").innerHTML = "";
}

function fillForm(ex) {
  clearForm();
  $("#source-text").value = ex.source.text || "";
  if (ex.source.followers) $("#src-followers").value = ex.source.followers;
  if (ex.source.account_age_days) $("#src-age").value = ex.source.account_age_days;
  $("#src-verified").checked = !!ex.source.verified; $("#src-media").checked = !!ex.source.has_media;
  const nodes = [];
  ex.replies.forEach(r => {
    const parentUid = r.parent > 0 && nodes[r.parent - 1] ? nodes[r.parent - 1].dataset.uid : "src";
    nodes.push(addReply({ ...r, parentUid }));
  });
  if (!ex.replies.length) addReply();
}

function parseTime(v) {
  if (v === undefined || v === null || v === "") return null;
  if (typeof v === "number") return v;
  const n = Number(v); if (!Number.isNaN(n)) return n;
  const d = Date.parse(String(v)); return Number.isNaN(d) ? null : d / 60000;
}

function importJson(obj) {            // same format as the desktop tool (predict.py)
  const thread = Array.isArray(obj) ? obj[0] : obj;
  const posts = (thread && thread.posts) || [];
  const roots = posts.filter(p => p.parent_id === null || p.parent_id === undefined || p.parent_id === "" || p.parent_id === "null");
  if (roots.length !== 1) throw new Error("JSON 里必须有且只有一条 parent_id 为 null 的原帖");
  const root = roots[0], t0 = parseTime(root.time) ?? 0;
  const rest = posts.filter(p => p !== root).map(p => ({ p, t: parseTime(p.time) }));
  rest.sort((a, b) => (a.t ?? Infinity) - (b.t ?? Infinity));
  const order = new Map([[String(root.id), 0]]); rest.forEach((r, k) => order.set(String(r.p.id), k + 1));
  const ex = {
    source: { text: root.text, followers: root.followers, account_age_days: root.account_age_days, verified: root.verified, has_media: root.has_media },
    replies: rest.map(({ p, t }, k) => {
      let parent = order.get(String(p.parent_id)); if (parent === undefined || parent > k) parent = 0;
      return { text: p.text, parent, t: t === null ? "" : Math.max(0, Math.round((t - t0) * 100) / 100), followers: p.followers,
        verified: p.verified, account_age_days: p.account_age_days, has_media: p.has_media, has_url: p.has_url };
    }),
  };
  fillForm(ex);
  return Array.isArray(obj) && obj.length > 1 ? `文件里有 ${obj.length} 个对话，已导入第一个。` : "已导入。";
}

function collectPosts() {
  const num = v => (v === undefined || v === null || v === "" ? 0 : Number(v) || 0);
  const posts = [{ id: "src", label: "原帖", parent: null, text: $("#source-text").value.trim(), t: 0,
    followers: num($("#src-followers").value), account_age_days: num($("#src-age").value),
    verified: $("#src-verified").checked ? 1 : 0, has_media: $("#src-media").checked ? 1 : 0, has_url: null }];
  if (!posts[0].text) throw new Error("请先填写原帖内容。");
  replyRows().forEach((row, k) => {
    const text = row.querySelector(".text").value.trim();
    if (!text) return;
    const m = row.querySelector(".minutes").value, x = row._extra || {};
    posts.push({ id: row.dataset.uid, label: `回复 ${k + 1}`, parent: row.querySelector(".parent").value, text,
      t: m === "" ? (k + 1) * GAP_MINUTES : Math.max(0, Number(m)), followers: num(x.followers), account_age_days: num(x.account_age_days),
      verified: x.verified ? 1 : 0, has_media: x.has_media ? 1 : 0, has_url: x.has_url ?? null });
  });
  const ids = new Set(posts.map(p => p.id));
  posts.forEach(p => { if (p.parent !== null && !ids.has(p.parent)) p.parent = "src"; });   // reply to an empty row → source
  const [root, ...rest] = posts;
  rest.sort((a, b) => a.t - b.t);                                                           // same order rule as predict.py
  return [root, ...rest];
}

// ------------------------------------------------------------------ models
let tf = null, emo = null, sent = null, translator = null, ensemble = null;
const fileProgress = new Map();
function onProgress(info) {
  if (info.status === "progress" && info.total) fileProgress.set(info.file + info.name, [info.loaded, info.total]);
  let a = 0, b = 0; for (const [l, t] of fileProgress.values()) { a += l; b += t; }
  if (b) { setProgress(a / b); setStatus(`正在下载模型 ${(a / 1e6).toFixed(0)} / ${(b / 1e6).toFixed(0)} MB（只有第一次需要）`); }
}
async function ensureModels(needTranslate) {
  if (!tf) { setStatus("正在加载运行环境..."); tf = await import(TF_URL); tf.env.allowLocalModels = false; }
  const jobs = [];
  if (!ensemble) jobs.push(loadEnsemble("model/").then(m => { ensemble = m; }));
  if (!emo) jobs.push(Promise.all([
    tf.AutoTokenizer.from_pretrained(EMOTION_MODEL),
    tf.AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL, { dtype: "q8", progress_callback: onProgress }),
  ]).then(([tok, model]) => {
    const l2i = {}; Object.entries(model.config.id2label).forEach(([i, l]) => { l2i[l] = Number(i); });
    emo = { tok, model, groups: EMOTIONS.map(n => EKMAN[n].map(l => l2i[l])) };
  }));
  if (!sent) jobs.push(tf.pipeline("feature-extraction", SENTENCE_MODEL, { dtype: "fp32", progress_callback: onProgress }).then(p => { sent = p; }));
  if (needTranslate && !translator) jobs.push(tf.pipeline("translation", TRANSLATE_MODEL, { dtype: "q8", progress_callback: onProgress }).then(p => { translator = p; }));
  await Promise.all(jobs);
}

// ------------------------------------------------------------------ features (same as predict.py)
async function featurise(posts) {
  const texts = posts.map(p => p.en.replace(URL_RE, "URL").replace(MENTION_RE, "@USER"));
  const emb = await sent(texts, { pooling: "mean", normalize: true });
  const inputs = emo.tok(texts, { padding: true, truncation: true, max_length: 128 });
  const logits = (await emo.model(inputs)).logits, L = logits.dims[1], ld = logits.data;
  const n = posts.length, D = 384 + 7 + 7, x = new Float32Array(n * D), e = new Float32Array(n * 7);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 384; j++) x[i * D + j] = emb.data[i * 384 + j];
    const g = emo.groups.map(ids => ids.reduce((s, id) => s + 1 / (1 + Math.exp(-ld[i * L + id])), 0));
    const tot = Math.max(g.reduce((a, b) => a + b, 0), 1e-12);
    g.forEach((v, k) => { e[i * 7 + k] = v / tot; x[i * D + 384 + k] = v / tot; });
    const words = texts[i].toLowerCase().match(/[a-z]+/g) || [];
    const moral = words.length ? words.filter(w => BRADY.some(b => (b.endsWith("*") ? w.startsWith(b.slice(0, -1)) : w === b))).length / words.length : 0;
    const p = posts[i];
    const hasUrl = p.has_url === null || p.has_url === undefined ? (new RegExp(URL_RE.source, "i").test(p.text) ? 1 : 0) : (p.has_url ? 1 : 0);
    const num = [moral, words.some(w => UNCERTAINTY.has(w)) ? 1 : 0, hasUrl, p.has_media, Math.log1p(p.followers), p.verified, Math.log1p(p.account_age_days)];
    num.forEach((v, k) => { x[i * D + 391 + k] = v; });
  }
  return { x, e };
}

// ------------------------------------------------------------------ run
async function run() {
  const btn = $("#run-btn");
  let posts;
  try { posts = collectPosts(); } catch (err) { setStatus(err.message, true); return; }
  const wantZh = $("#is-chinese").checked;
  const hasCjk = posts.some(p => CJK_RE.test(p.text));
  btn.disabled = true; $("#progress").hidden = false; setProgress(0);
  try {
    await ensureModels(wantZh);
    if (hasCjk && !wantZh) setStatus("检测到中文内容，建议勾选“内容是中文”，否则结果不可靠。", true);
    for (const [k, p] of posts.entries()) {
      p.en = p.text; p.translated = false;
      if (wantZh && CJK_RE.test(p.text)) {
        setStatus(`正在翻译第 ${k + 1} / ${posts.length} 条...`); setProgress(k / posts.length);
        p.en = (await translator(p.text))[0].translation_text; p.translated = true;
      }
    }
    setStatus("正在识别情绪..."); setProgress(0.9);
    const { x, e } = await featurise(posts);
    const index = new Map(posts.map((p, k) => [p.id, k]));
    const parents = posts.map(p => (p.parent === null ? -1 : index.get(p.parent)));
    const t = Float32Array.from(posts.map(p => p.t));
    const edges = buildEdges(parents);
    setStatus("ECHO 模型计算中...");
    const outs = ensemble.map(m => ({ m, ...m.forward(x, e, t, parents, edges) }));
    const probs = outs.map(o => o.prob), votes = outs.filter(o => o.prob >= o.m.threshold).length;
    const alpha = edges.td.map((_, k) => outs.reduce((s, o) => s + o.alphaTd[k], 0) / outs.length);
    const strongest = {}, parentAlpha = {};
    edges.td.forEach(([s, g], k) => {
      if (s === g) return;
      if (!strongest[g] || alpha[k] > strongest[g][1]) strongest[g] = [s, alpha[k]];
      if (parents[g] === s) parentAlpha[g] = alpha[k];
    });
    const mean = probs.reduce((a, b) => a + b, 0) / probs.length;
    lastResult = { posts, e, parents, probs, votes, mean, strongest, parentAlpha, models: ensemble };
    render(lastResult);
    setStatus("完成 ✓"); setProgress(1);
    $("#result-card").hidden = false; $("#result-card").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    console.error(err);
    setStatus("出错了：" + (err.message || err) + "（请检查网络能否访问 huggingface.co 和 cdn.jsdelivr.net）", true);
  } finally { btn.disabled = false; setTimeout(() => { $("#progress").hidden = true; }, 600); }
}

// ------------------------------------------------------------------ output
const dominant = (e, i) => { let b = 0; for (let k = 1; k < 7; k++) if (e[i * 7 + k] > e[i * 7 + b]) b = k; return EMOTIONS[b]; };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function render(r) {
  const n = r.models.length, rumour = r.votes > n / 2, pct = Math.round(r.mean * 100);
  $("#prob").textContent = pct + "%";
  const arc = $("#gauge-arc"); arc.style.stroke = rumour ? "#e03131" : "#37b24d";
  requestAnimationFrame(() => { arc.style.strokeDashoffset = String(314.16 * (1 - r.mean)); });
  const d = $("#decision"); d.textContent = rumour ? "较像谣言" : "不太像谣言"; d.className = "decision " + (rumour ? "hi" : "lo");
  $("#votes").innerHTML = `15 个 ECHO 模型中有 <b>${r.votes}</b> 个判断为谣言（各模型概率 ${Math.round(Math.min(...r.probs) * 100)}%–${Math.round(Math.max(...r.probs) * 100)}%）。`;
  const se = dominant(r.e, 0);
  $("#src-emotion").innerHTML = `原帖的主要情绪：<span class="chip" style="background:${COLORS[se]}">${ZH[se]}</span>`;
  $("#legend").innerHTML = EMOTIONS.map(m => `<span><i class="dot" style="background:${COLORS[m]}"></i>${ZH[m]}</span>`).join("");
  const tbody = $("#post-table tbody");
  tbody.innerHTML = r.posts.map((p, i) => {
    const dom = dominant(r.e, i);
    const dist = EMOTIONS.map((m, k) => `<i title="${ZH[m]} ${(r.e[i * 7 + k] * 100).toFixed(0)}%" style="width:${r.e[i * 7 + k] * 100}%;background:${COLORS[m]}"></i>`).join("");
    const inf = i === 0 ? "—" : `${r.posts[r.strongest[i][0]].label}<br><small>权重 ${r.strongest[i][1].toFixed(2)}</small>`;
    return `<tr><td class="nw"><b>${p.label}</b><br><small>${p.t} 分钟</small></td><td>${esc(p.text)}${p.translated ? `<div class="en">英文：${esc(p.en)}</div>` : ""}</td>` +
      `<td><span class="chip" style="background:${COLORS[dom]}">${ZH[dom]}</span></td><td><div class="dist">${dist}</div></td><td>${inf}</td></tr>`;
  }).join("");
  drawTree(r);
}

function drawTree(r) {
  const n = r.posts.length, children = Array.from({ length: n }, () => []);
  r.parents.forEach((p, i) => { if (p >= 0) children[p].push(i); });
  const pos = [], leaf = { v: 0 }; let maxDepth = 0;
  (function place(i, depth) {
    maxDepth = Math.max(maxDepth, depth);
    if (!children[i].length) { pos[i] = [leaf.v++, depth]; return; }
    children[i].forEach(c => place(c, depth + 1));
    pos[i] = [children[i].reduce((s, c) => s + pos[c][0], 0) / children[i].length, depth];
  })(0, 0);
  const DX = 120, DY = 110, PAD = 60, W = Math.max(560, (leaf.v - 1) * DX + 2 * PAD), H = maxDepth * DY + 2 * PAD;
  const off = (W - (leaf.v - 1) * DX) / 2, X = i => off + pos[i][0] * DX, Y = i => PAD + pos[i][1] * DY;
  let s = "";
  for (let i = 1; i < n; i++) {
    const p = r.parents[i], a = r.parentAlpha[i] ?? 0;
    s += `<line x1="${X(p)}" y1="${Y(p)}" x2="${X(i)}" y2="${Y(i)}" stroke="#4a5263" stroke-linecap="round" stroke-width="${(1 + 9 * a).toFixed(1)}" opacity=".75"/>`;
    s += `<g><rect x="${(X(p) + X(i)) / 2 - 17}" y="${(Y(p) + Y(i)) / 2 - 12}" width="34" height="18" rx="5" fill="#fff"/><text x="${(X(p) + X(i)) / 2}" y="${(Y(p) + Y(i)) / 2 + 1}" font-size="11" text-anchor="middle" fill="#2c3a5a">${a.toFixed(2)}</text></g>`;
  }
  for (let i = 0; i < n; i++) {
    const dom = dominant(r.e, i), rad = i === 0 ? 24 : 18;
    s += `<g><title>${esc(r.posts[i].label + "：" + r.posts[i].text)}</title><circle cx="${X(i)}" cy="${Y(i)}" r="${rad}" fill="${COLORS[dom]}" stroke="#1d1d28" stroke-width="1.3"/>` +
      `<text x="${X(i)}" y="${Y(i) + 4}" font-size="${i === 0 ? 12 : 11}" font-weight="700" text-anchor="middle">${i === 0 ? "原帖" : r.posts[i].label.replace("回复 ", "")}</text>` +
      `<text x="${X(i) + rad + 5}" y="${Y(i) + 4}" font-size="11" fill="#475467">${ZH[dom]}</text></g>`;
  }
  const svg = $("#tree"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H); svg.innerHTML = s;
}

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function rowsOf(r) {
  return r.posts.map((p, i) => {
    const dom = dominant(r.e, i), row = {
      "编号": p.label, "回复给": i ? r.posts[r.parents[i]].label : "", "原帖后分钟": p.t, "内容": p.text, "英文(翻译后)": p.translated ? p.en : "",
      "主要情绪": ZH[dom] };
    EMOTIONS.forEach((m, k) => { row[ZH[m]] = +r.e[i * 7 + k].toFixed(4); });
    row["对上一条的注意力"] = i ? +(r.parentAlpha[i] ?? 0).toFixed(4) : "";
    row["最受谁影响"] = i ? r.posts[r.strongest[i][0]].label : "";
    row["影响权重"] = i ? +r.strongest[i][1].toFixed(4) : "";
    return row;
  });
}
function toCsv(r) {
  const rows = rowsOf(r), cols = Object.keys(rows[0]), q = v => `"${String(v).replace(/"/g, '""')}"`;
  return "﻿" + [`谣言可能性,${r.mean.toFixed(4)},判为谣言的模型数,${r.votes}/${r.models.length}`, cols.map(q).join(","),
    ...rows.map(row => cols.map(c => q(row[c])).join(","))].join("\r\n");
}

// ------------------------------------------------------------------ ui helpers
function setStatus(msg, err = false) { const s = $("#status"); s.textContent = msg; s.className = "status" + (err ? " err" : ""); }
function setProgress(f) { $("#progress-bar").style.width = Math.round(Math.min(1, Math.max(0, f)) * 100) + "%"; }

$("#add-reply").onclick = () => addReply().querySelector(".text").focus();
$("#clear-btn").onclick = () => { clearForm(); addReply(); setStatus(""); };
document.querySelectorAll("[data-example]").forEach(b => { b.onclick = () => { fillForm(EXAMPLES[b.dataset.example]); $("#is-chinese").checked = false; setStatus("已填入示例，点“开始分析”。"); }; });
$("#json-file").onchange = async ev => {
  const f = ev.target.files[0]; if (!f) return;
  try { setStatus(importJson(JSON.parse(await f.text()))); } catch (err) { setStatus("导入失败：" + err.message, true); }
  ev.target.value = "";
};
$("#run-btn").onclick = run;
$("#csv-btn").onclick = () => lastResult && download("ECHO_结果.csv", toCsv(lastResult), "text/csv;charset=utf-8");
$("#json-btn").onclick = () => lastResult && download("ECHO_结果.json", JSON.stringify({
  "谣言可能性": +lastResult.mean.toFixed(4), "判为谣言的模型数": `${lastResult.votes}/${lastResult.models.length}`,
  "结论": lastResult.votes > lastResult.models.length / 2 ? "较像谣言" : "不太像谣言",
  "各模型概率": Object.fromEntries(lastResult.models.map((m, k) => [m.name, +lastResult.probs[k].toFixed(4)])),
  "帖子": rowsOf(lastResult) }, null, 2), "application/json");
addReply();
