// ECHO forward pass in plain JavaScript (same maths as predict.py / src/models/echo.py, inference mode).
export const EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"];

function linear(x, n, inDim, W, b, outDim) {          // x: n*inDim, W: outDim*inDim (torch layout)
  const y = new Float32Array(n * outDim);
  for (let i = 0; i < n; i++) {
    const xo = i * inDim;
    for (let o = 0; o < outDim; o++) {
      let s = b ? b[o] : 0; const wo = o * inDim;
      for (let k = 0; k < inDim; k++) s += x[xo + k] * W[wo + k];
      y[i * outDim + o] = s;
    }
  }
  return y;
}
const relu = a => { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; return a; };
const softplus = v => (v > 20 ? v : Math.log1p(Math.exp(v)));

export function buildEdges(parents) {               // parents[i] = parent index or -1 (node 0 = source)
  const n = parents.length, td = [], bu = [], children = Array.from({ length: n }, () => []);
  for (let i = 1; i < n; i++) children[parents[i]].push(i);
  for (let i = 0; i < n; i++) {
    td.push([i, i]);
    const seen = new Set(); let c = parents[i];
    while (c >= 0 && !seen.has(c)) { seen.add(c); td.push([c, i]); c = parents[c]; }
    bu.push([i, i]);
    for (const ch of children[i]) bu.push([ch, i]);
  }
  return { td, bu };
}

export class EchoModel {
  constructor(spec, buffer) {
    this.name = spec.name; this.threshold = spec.threshold; this.p = {};
    for (const [k, v] of Object.entries(spec.tensors))
      this.p[k] = { data: new Float32Array(buffer, v.offset * 4, v.shape.reduce((a, b) => a * b, 1)), shape: v.shape };
    this.hidden = this.p["encoder.3.weight"].shape[0]; this.inDim = this.p["encoder.0.weight"].shape[1];
  }
  W(k) { return this.p[k].data; }
  layer(pre, h, x, e, t, edges, n, beta, delta) {
    const H = this.hidden, D = this.inDim, C = this.W("C"), bself = this.W("b_self")[0];
    const proj = linear(x, n, D, this.W(pre + ".semantic.weight"), this.W(pre + ".semantic.bias"), H);
    const sw = this.W(pre + ".score.weight");
    const scores = new Float64Array(edges.length);
    edges.forEach(([s, g], k) => {
      let sem = 0;
      for (let j = 0; j < H; j++) sem += sw[j] * proj[g * H + j] + sw[H + j] * proj[s * H + j];
      let extra;
      if (s === g) extra = bself;
      else {
        let emo = 0;
        for (let a = 0; a < 7; a++) { let v = 0; for (let b = 0; b < 7; b++) v += e[s * 7 + b] * C[b * 7 + a]; emo += v * e[g * 7 + a]; }
        const lag = Math.log1p(Math.abs(t[g] - t[s]));
        extra = emo + beta * Math.exp(-delta * lag);
      }
      scores[k] = Math.fround(sem + extra);
    });
    const mx = new Float64Array(n).fill(-Infinity), den = new Float64Array(n);
    edges.forEach(([, g], k) => { if (scores[k] > mx[g]) mx[g] = scores[k]; });
    const ex = scores.map((v, k) => Math.exp(v - mx[edges[k][1]]));
    edges.forEach(([, g], k) => { den[g] += ex[k]; });
    const alpha = ex.map((v, k) => v / Math.max(den[edges[k][1]], 1e-16));
    const msg = linear(h, n, H, this.W(pre + ".message.weight"), this.W(pre + ".message.bias"), H);
    const out = Float32Array.from(h);
    edges.forEach(([s, g], k) => { for (let j = 0; j < H; j++) out[g * H + j] += msg[s * H + j] * alpha[k]; });
    return { h: relu(out), alpha };
  }
  forward(x, e, t, parents, edges = buildEdges(parents)) {
    const n = parents.length, H = this.hidden, D = this.inDim;
    const beta = softplus(this.W("beta_raw")[0]), delta = softplus(this.W("delta_raw")[0]);
    const h0 = linear(relu(linear(x, n, D, this.W("encoder.0.weight"), this.W("encoder.0.bias"), H)), n, H,
                      this.W("encoder.3.weight"), this.W("encoder.3.bias"), H);
    let td = h0, bu = h0, alphaTd = null;
    for (const l of [0, 1]) { const r = this.layer(`td_layers.${l}`, td, x, e, t, edges.td, n, beta, delta); td = r.h; alphaTd = r.alpha; }
    for (const l of [0, 1]) { bu = this.layer(`bu_layers.${l}`, bu, x, e, t, edges.bu, n, beta, delta).h; }
    const ro = new Float32Array(4 * H);
    for (let j = 0; j < H; j++) {
      ro[j] = td[j]; ro[2 * H + j] = bu[j];
      let a = 0, b = 0; for (let i = 0; i < n; i++) { a += td[i * H + j]; b += bu[i * H + j]; }
      ro[H + j] = a / n; ro[3 * H + j] = b / n;
    }
    const z = relu(linear(ro, 1, 4 * H, this.W("classifier.0.weight"), this.W("classifier.0.bias"), H));
    const lg = linear(z, 1, H, this.W("classifier.3.weight"), this.W("classifier.3.bias"), 2);
    const prob = 1 / (1 + Math.exp(lg[0] - lg[1]));
    return { prob, alphaTd, edgesTd: edges.td };
  }
}

export async function loadEnsemble(base = "model/") {
  const [spec, buf] = await Promise.all([
    fetch(base + "echo_weights.json").then(r => r.json()),
    fetch(base + "echo_weights.bin").then(r => r.arrayBuffer())]);
  return spec.models.map(m => new EchoModel(m, buf));
}
