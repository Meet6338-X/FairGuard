import { useEffect, useState } from "react";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import { train, datasets, models } from "../api.ts";
import { Bot, Play, FileText, Cpu, AlertTriangle, ArrowRight } from "lucide-react";

export default function Train() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [datasetsList, setDatasets] = useState([]);
  const [modelsList, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [form, setForm] = useState({
    dataset: searchParams.get("dataset") || "",
    label: "approved",
    sensitive: "",
    modelTypes: ["logistic_regression", "random_forest", "gradient_boosting"],
  });

  useEffect(() => {
    datasets().then((d) => setDatasets(d.datasets || []));
    models().then((m) => setModels(m.models || []));
  }, []);

  const currentDs = datasetsList.find((d) => d === form.dataset);
  const columns = currentDs ? [] : []; // columns fetched dynamically via backend if needed

  const MODEL_TYPES = [
    { key: "logistic_regression", label: "Logistic Regression", desc: "Linear model, fast and interpretable" },
    { key: "random_forest", label: "Random Forest", desc: "Ensemble of trees, robust to outliers" },
    { key: "gradient_boosting", label: "Gradient Boosting", desc: "Sequential trees, often highest accuracy" },
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const { results } = await train({
        dataset: form.dataset || undefined,
        label: form.label,
        sensitive: form.sensitive || undefined,
        model_types: form.modelTypes,
      });
      setResult(results);
      navigate(`/train?dataset=${encodeURIComponent(form.dataset || "")}`);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  return (
    <div className="page-container space-y-8">
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold mb-1">Train Model</h1>
          <p className="text-slate-500 text-sm">Train baseline models on any dataset. FairGuard auto-selects numeric features and keeps the sensitive attribute out of model features.</p>
        </div>
        <Link to="/" className="btn-ghost">
          <ArrowRight className="h-4 w-4 mr-2" /> Back to dashboard
        </Link>
      </div>

      <form onSubmit={handleSubmit} className="card space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div>
            <label className="form-label">Dataset</label>
            <select
              value={form.dataset}
              onChange={(e) => setForm({ ...form, dataset: e.target.value, sensitive: "" })}
              className="form-select"
            >
              <option value="">Select a dataset</option>
              {datasetsList.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="form-label">Label column</label>
            <input
              value={form.label}
              onChange={(e) => setForm({ ...form, label: e.target.value })}
              className="form-input"
              placeholder="e.g. approved, credit_good, income_gt_50k"
            />
          </div>
          <div>
            <label className="form-label">Sensitive attribute</label>
            <select
              value={form.sensitive}
              onChange={(e) => setForm({ ...form, sensitive: e.target.value })}
              className="form-select"
            >
              <option value="">None (fairness still measured)</option>
              {currentDs ? [
                "sex", "gender", "race", "majority_white", "gender_like", "age_group", "age"
              ].filter(c => c !== form.label).map((c) => (
                <option key={c} value={c}>{c}</option>
              )) : null}
            </select>
          </div>
        </div>

        <div className="space-y-3">
          <label className="form-label">Model types</label>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {MODEL_TYPES.map((t) => (
              <label key={t.key} className="flex items-center gap-2 cursor-pointer p-3 border rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors group">
                <input
                  type="checkbox"
                  checked={form.modelTypes.includes(t.key)}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      modelTypes: e.target.checked
                        ? [...form.modelTypes, t.key].filter(Boolean)
                        : form.modelTypes.filter((x) => x !== t.key),
                    })
                  }
                  className="rounded text-blue-600 focus:ring-blue-500"
                />
                <div>
                  <div className="text-sm font-medium text-slate-800 group-hover:text-blue-800">{t.label}</div>
                  <div className="text-xs text-slate-500">{t.desc}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !form.dataset}
          className={`btn-primary ${loading || !form.dataset ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          {loading ? "Training…" : <><Play className="h-4 w-4 mr-2" />Train models</>}
        </button>
      </form>

      {error && (
        <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">
          {error}
        </div>
      )}

      {result && (
        <section className="card">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="h-5 w-5 text-indigo-600" />
            <h2 className="text-lg font-semibold">Training results</h2>
          </div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Type</th>
                  <th className="text-right">Test accuracy</th>
                  <th>Features used</th>
                </tr>
              </thead>
              <tbody>
                {result.map((r) => (
                  <tr key={r.model} className="hover:bg-slate-50">
                    <td className="font-mono text-xs text-slate-700">{r.model}</td>
                    <td className="text-gray-700">{r.model_type.replace(/_/g, " ")}</td>
                    <td className="text-right font-bold text-slate-900">{r.test_accuracy?.toFixed(3)}</td>
                    <td className="text-slate-500 text-xs truncate max-w-xs">{r.features?.join(", ") ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-sm text-slate-500">
            Models saved to <code className="font-mono bg-slate-100 px-1.5 py-0.5 rounded">models/</code>.
            Go to <Link to="/assess" className="text-blue-600 hover:underline font-medium">Assess Model</Link>
            to evaluate them.
          </p>
        </section>
      )}
    </div>
  );
}
