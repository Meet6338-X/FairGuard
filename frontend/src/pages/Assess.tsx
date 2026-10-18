import { useEffect, useState } from "react";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import { assess, datasets, models } from "../api.ts";
import { BarChart3, Play, FileText, AlertTriangle, ArrowRight } from "lucide-react";

export default function Assess() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [datasetsList, setDatasets] = useState([]);
  const [modelsList, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const [form, setForm] = useState({
    dataset: searchParams.get("dataset") || "demo_loans.csv",
    model: "",
    label: "approved",
    sensitive: "gender",
    threshold: "0.5",
    useReference: true,
  });

  useEffect(() => {
    datasets().then((d) => setDatasets(d.datasets || []));
    models().then((m) => setModels(m.models || []));
  }, []);

  // Suggest sensitive attribute based on dataset name
  const suggestedSensitive = (ds: string) => {
    const lower = ds.toLowerCase();
    if (lower.includes("adult") || lower.includes("german")) return "sex";
    if (lower.includes("taiwan")) return "gender";
    if (lower.includes("compas")) return "race";
    if (lower.includes("bank")) return "age";
    if (lower.includes("student")) return "sex";
    if (lower.includes("heart")) return "sex";
    if (lower.includes("communities")) return "majority_white";
    if (lower.includes("australian")) return "gender_like";
    if (lower.includes("demo")) return "gender";
    return "";
  };

  useEffect(() => {
    if (form.dataset && !form.sensitive) {
      setForm({ ...form, sensitive: suggestedSensitive(form.dataset) });
    }
    // Suggest label based on dataset
    const labelMap: Record<string, string> = {
      adult: "income_gt_50k",
      german: "credit_good",
      taiwan_credit: "default_next_month",
      compas_two_years: "two_year_recid",
      bank_marketing: "subscribed",
      student_performance: "G3_pass",
      heart: "disease",
      communities: "high_violent_crime",
      australian: "credit_approved",
      demo_loans: "approved",
      demo_loans_new: "approved",
    };
    const key = Object.keys(labelMap).find((k) => form.dataset.includes(k));
    if (key && !form.label) {
      setForm({ ...form, label: labelMap[key] });
    }
  }, [form.dataset]);

  // Filter models that match the selected dataset
  const matchingModels = modelsList.filter((m) => {
    const base = form.dataset.replace(".csv", "");
    return m.startsWith(base) || m === "logistic_regression.pkl" || m === "random_forest.pkl" || m === "gradient_boosting.pkl";
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const resp = await assess({
        dataset: form.dataset,
        model: form.model,
        label: form.label,
        sensitive: form.sensitive,
        threshold: parseFloat(form.threshold),
        use_reference: form.useReference,
      });
      setResult(resp);
      navigate(`/results/${resp.run_id}`);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="flex flex-col items-center justify-center py-20">
          <div className="animate-spin h-8 w-8 border-4 border-blue-200 border-t-blue-600 rounded-full mb-4" />
          <p className="text-slate-500">Running assessment… this may take a moment.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container space-y-8">
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold mb-1">Assess Model</h1>
          <p className="text-slate-500 text-sm">Run a full responsible-AI assessment: fairness, explainability, privacy, drift, and risk.</p>
        </div>
        <Link to="/" className="btn-ghost">
          <ArrowRight /> Back to dashboard
        </Link>
      </div>

      <form onSubmit={handleSubmit} className="card space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div>
            <label className="form-label">Dataset</label>
            <select
              value={form.dataset}
              onChange={(e) => setForm({ ...form, dataset: e.target.value, model: "", label: "", sensitive: "" })}
              className="form-select"
            >
              {datasetsList.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="form-label">Model</label>
            <select
              value={form.model}
              onChange={(e) => setForm({ ...form, model: e.target.value })}
              className="form-select"
            >
              <option value="">Select a model</option>
              {matchingModels.length > 0 ? (
                matchingModels.map((m) => (
                  <option key={m} value={m}>
                    {m} {matchingModels.length < modelsList.length ? "(matches dataset)" : ""}
                  </option>
                ))
              ) : (
                modelsList.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))
              )}
            </select>
            {matchingModels.length === 0 && modelsList.length > 0 && (
              <p className="text-xs text-slate-400 mt-1">No exact match for {form.dataset}. Showing all models.</p>
            )}
          </div>
          <div>
            <label className="form-label">Label column</label>
            <input
              value={form.label}
              onChange={(e) => setForm({ ...form, label: e.target.value })}
              className="form-input"
              placeholder="e.g. approved, income_gt_50k"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div>
            <label className="form-label">Sensitive attribute</label>
            <input
              value={form.sensitive}
              onChange={(e) => setForm({ ...form, sensitive: e.target.value })}
              className="form-input"
              placeholder="e.g. gender, sex, race"
            />
          </div>
          <div>
            <label className="form-label">Decision threshold</label>
            <input
              type="number"
              min="0.05"
              max="0.95"
              step="0.05"
              value={form.threshold}
              onChange={(e) => setForm({ ...form, threshold: e.target.value })}
              className="form-input"
            />
          </div>
          <div className="flex items-end">
            <label className="flex items-center gap-2 cursor-pointer p-3 border rounded-lg hover:bg-slate-50 transition-colors">
              <input
                type="checkbox"
                checked={form.useReference}
                onChange={(e) => setForm({ ...form, useReference: e.target.checked })}
                className="rounded text-blue-600 focus:ring-blue-500"
              />
              Use as drift reference
            </label>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !form.model || !form.label}
          className={`btn-primary ${loading || !form.model || !form.label ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          {loading ? "Assessing…" : <><Play className="h-4 w-4 mr-2" />Run assessment</>}
        </button>
      </form>

      {error && (
        <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">
          {error}
        </div>
      )}

      {result && (
        <section className="card border-l-4 border-green-500 bg-green-50/50">
          <div className="flex items-center gap-2 mb-4">
            <FileText className="h-5 w-5 text-green-600" />
            <h2 className="text-lg font-semibold">Assessment complete</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
            <div className="p-3 rounded-lg border border-slate-100 bg-white">
              <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Risk score</div>
              <div className="font-bold text-slate-900">{result.assessment.risk.score.toFixed(3)}</div>
            </div>
            <div className="p-3 rounded-lg border border-slate-100 bg-white">
              <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Risk level</div>
              <div className={`font-bold capitalize ${result.assessment.risk.level === "High" ? "text-red-600" : result.assessment.risk.level === "Medium" ? "text-amber-600" : "text-green-600"}`}>
                {result.assessment.risk.level}
              </div>
            </div>
            <div className="p-3 rounded-lg border border-slate-100 bg-white">
              <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Decision</div>
              <div className="font-bold text-slate-900">{result.assessment.decision}</div>
            </div>
          </div>
          <p className="mt-3 text-sm text-slate-500">
            Run ID: <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">{result.run_id}</code>
          </p>
          <Link
            to={`/results/${result.run_id}`}
            className="text-blue-600 hover:underline font-medium flex items-center gap-2 inline-block mt-2"
          >
            <BarChart3 className="h-4 w-4" /> View full results
          </Link>
        </section>
      )}
    </div>
  );
}
