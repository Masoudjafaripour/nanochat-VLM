// Web Worker: owns the wasm module and the model so generation never blocks the UI.
import init, { Model } from "./pkg/vlm_wasm.js";

let model = null;
const ready = init();

async function fetchOk(url, kind) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return kind === "bytes" ? new Uint8Array(await r.arrayBuffer()) : r.text();
}

self.onmessage = async ({ data }) => {
  try {
    await ready;
    if (data.type === "load") {
      const base = data.base.endsWith("/") ? data.base : data.base + "/";
      const t = performance.now();
      const [config, weights, tokenizer] = await Promise.all([
        fetchOk(base + "config.json", "text"),
        fetchOk(base + "model.safetensors", "bytes"),
        fetchOk(base + "tokenizer.json", "text"),
      ]);
      model = new Model(config, weights, tokenizer);
      const mb = (weights.byteLength / 1048576).toFixed(1);
      self.postMessage({ type: "loaded", info: `loaded ${mb} MiB in ${(performance.now() - t).toFixed(0)} ms` +
                                              ` · projector: ${model.has_projector ? "yes" : "no"}` });
    } else if (data.type === "generate") {
      if (!model) throw new Error("load a model first");
      const stats = model.generate(data.prompt, data.image ?? undefined, data.maxNewTokens, data.temperature,
                                   undefined, 42n, (text) => { self.postMessage({ type: "token", text }); return true; });
      self.postMessage({ type: "done", stats: JSON.parse(stats) });
    }
  } catch (e) {
    self.postMessage({ type: "error", message: String(e?.message ?? e) });
  }
};
