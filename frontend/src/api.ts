const API_BASE = "http://127.0.0.1:8000/api";

export async function api(path, opts = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

export const datasets = () => api("/datasets");
export const models = () => api("/models");
export const train = (body) => api("/train", { method: "POST", body: JSON.stringify(body) });
export const assess = (body) => api("/assess", { method: "POST", body: JSON.stringify(body) });
export const history = () => api("/history");
export const review = (body) => api("/review", { method: "POST", body: JSON.stringify(body) });
export const report = (runId, fmt) => api(`/report/${encodeURIComponent(runId)}/${fmt}`);
