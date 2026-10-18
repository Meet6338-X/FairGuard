import { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import { report } from "../api.ts";
import {
  BarChart3,
  ShieldCheck,
  FileText,
  Clock,
  AlertTriangle,
  Download,
  ArrowRight,
  PieChart as PieChartIcon,
  Activity,
} from "lucide-react";
import { BarChart, Bar, PieChart, Pie, Cell } from "recharts";

const LEVEL_COLORS: Record<string, string> = {
  Low: "#22c55e",
  Medium: "#f59e0b",
  High: "#ef4444",
  Critical: "#7f1d1d",
};

const COLOR_PALETTE = ["#0ea5e9", "#0284c7", "#3b82f6", "#6366f1", "#a855f7", "#ec4899"];

export default function Results() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [assessment, setAssessment] = useState<any>(null);
  const [rawJson, setRawJson] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    report(runId, "json")
      .then((r) => {
        setRawJson(r.content);
        setAssessment(JSON.parse(r.content));
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [runId]);

  if (loading) {
    return (
      <div className="page-container">
        <div className="flex flex-col items-center justify-center py-20">
          <div className="animate-spin h-8 w-8 border-4 border-blue-200 border-t-blue-600 rounded-full mb-4" />
          <p className="text-slate-500">Loading results…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <div className="text-center py-20">
          <AlertTriangle className="h-10 w-10 text-red-500 mx-auto mb-3" />
          <h2 className="text-lg font-semibold text-red-700">Error loading results</h2>
          <p className="text-gray-500">{error}</p>
          <Link to="/" className="mt-4 inline-block text-blue-600 hover:underline">
            Back to dashboard
          </Link>
        </div>
      </div>
    );
  }

  const a = assessment;
  const risk = a.risk || {};
  const perf = a.performance || {};
  const fair = a.fairness || {};
  const priv = a.privacy || {};
  const drift = a.drift || {};
  const expl = a.explainability || {};

  const levelColor = LEVEL_COLORS[risk.level] || "#666";
  const badgeClass =
    risk.level === "High"
      ? "badge-high"
      : risk.level === "Medium"
        ? "badge-medium"
        : risk.level === "Low"
          ? "badge-low"
          : "badge-critical";

  const perfData = [
    { name: "Accuracy", value: perf.accuracy ?? 0 },
    { name: "Precision", value: perf.precision ?? 0 },
    { name: "Recall", value: perf.recall ?? 0 },
    { name: "F1", value: perf.f1 ?? 0 },
    { name: "ROC-AUC", value: perf.roc_auc ?? 0 },
  ];

  const fairnessData = fair.by_group
    ? Object.entries(fair.by_group).flatMap(([metric, groups]) =>
        Object.entries(groups as Record<string, number>).map(([group, val]) => ({
          metric,
          group,
          value: val,
        }))
      )
    : [];

  return (
    <div className="page-container space-y-8">
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold mb-1">Assessment Results</h1>
          <p className="text-slate-500 text-sm">
            Run <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">{runId}</code>
            &middot; {a.assessed_at}
          </p>
        </div>
        <Link to="/" className="btn-ghost">
          <ArrowRight className="h-4 w-4 mr-2" /> Back to dashboard
        </Link>
      </div>

      {/* Risk summary */}
      <section className="card">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-6">
          <div className="flex-1">
            <div className="flex items-baseline gap-2">
              <span className="text-5xl font-bold" style={{ color: levelColor }}>
                {risk.score?.toFixed(2)}
              </span>
              <span className={`text-xl font-bold capitalize ${badgeClass}`}>
                {risk.level}
              </span>
            </div>
            <p className="text-slate-500 text-sm mt-1">
              Suggested decision: <span className="font-semibold">{a.decision}</span>
            </p>
          </div>
          <div className="text-right md:text-left">
            <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Model</div>
            <div className="font-mono text-sm">{a.model_id}</div>
            <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold mt-2">Dataset</div>
            <div className="font-mono text-sm">{a.dataset_id}</div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { label: "Fairness", key: "fairness", icon: <ShieldCheck className="h-4 w-4" /> },
            { label: "Privacy", key: "privacy", icon: <FileText className="h-4 w-4" /> },
            { label: "Drift", key: "drift", icon: <Clock className="h-4 w-4" /> },
          ].map((d) => {
            const v = risk.contributions?.[d.key] ?? 0;
            return (
              <div key={d.key} className="p-4 rounded-lg border border-slate-100 bg-slate-50">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {d.icon}
                    <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">{d.label}</div>
                  </div>
                  <span className={`text-xs font-bold ${v > 0.5 ? "text-red-600" : v > 0.25 ? "text-amber-600" : "text-green-600"}`}>
                    {v.toFixed(3)}
                  </span>
                </div>
                <div className="risk-meter">
                  <div
                    className="risk-meter-fill"
                    style={{
                      width: `${Math.min(v * 100, 100)}%`,
                      backgroundColor: v > 0.5 ? "#ef4444" : v > 0.25 ? "#f59e0b" : "#22c55e",
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-6">
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Recommendations</div>
            <span className="text-xs text-slate-400">
              {(a.recommendations || []).length} item(s)
            </span>
          </div>
          <ul className="list-disc list-inside text-sm text-slate-700 space-y-2">
            {(a.recommendations || []).map((r: string, i: number) => (
              <li key={i} className="flex gap-2">
                <span className="text-blue-600">•</span>
                {r}
              </li>
            ))}
            {(a.recommendations || []).length === 0 && (
              <li className="text-slate-400">No specific recommendations — keep monitoring.</li>
            )}
          </ul>
        </div>
      </section>

      {/* Charts & metrics */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Performance chart */}
        <div className="lg:col-span-2 card">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Performance metrics</h3>
          <div className="h-64">
            <BarChart width={600} height={300} data={perfData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <Bar dataKey="value" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
            </BarChart>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm mt-4">
            {perfData.map((d) => (
              <div key={d.name} className="flex justify-between border-b border-slate-100 pb-1 last:border-0">
                <span className="text-slate-500">{d.name}</span>
                <span className="font-mono">{d.value == null ? "—" : d.value.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Risk pie */}
        <div className="card">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Risk breakdown</h3>
          <div className="h-64 flex items-center justify-center">
            <PieChart width={300} height={300}>
              <Pie
                data={[
                  { name: "Fairness", value: risk.contributions?.fairness ?? 0 },
                  { name: "Privacy", value: risk.contributions?.privacy ?? 0 },
                  { name: "Drift", value: risk.contributions?.drift ?? 0 },
                  { name: "Performance", value: risk.contributions?.performance ?? 0 },
                  { name: "Text", value: risk.contributions?.text ?? 0 },
                ].filter((d) => d.value > 0)}
                dataKey="value"
                nameKey="name"
                cx={150}
                cy={150}
                innerRadius={60}
                outerRadius={140}
                fill="#8884d8"
                paddingAngle={2}
              >
                {COLOR_PALETTE.map((color, i) => (
                  <Cell key={`cell-${i}`} fill={color} />
                ))}
              </Pie>
            </PieChart>
          </div>
          <div className="mt-4 space-y-2 text-sm">
            {Object.entries(risk.contributions || {}).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between">
                <span className="text-slate-500">{k}</span>
                <span className="font-mono">{v.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Detail panels */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <section className="card lg:col-span-2">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Fairness — by group</h3>
          {fairnessData.length === 0 ? (
            <p className="text-slate-500 text-sm">No group fairness data available.</p>
          ) : (
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Group</th>
                    <th className="text-right">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {fairnessData.map((d: any, i: number) => (
                    <tr key={i} className="hover:bg-slate-50">
                      <td className="font-mono text-xs">{d.metric}</td>
                      <td className="text-sm">{d.group}</td>
                      <td className="text-right font-mono text-xs">{d.value.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="card">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Privacy findings</h3>
          {priv.findings && priv.findings.length > 0 ? (
            <div className="space-y-2 text-sm">
              {priv.findings.map((f: any, i: number) => (
                <div key={i} className="p-3 rounded-lg border border-slate-100 bg-white">
                  <div className="flex justify-between mb-1">
                    <span className="font-medium text-slate-800">{f.column}</span>
                    <span className="text-xs font-bold uppercase tracking-wider">{f.severity}</span>
                  </div>
                  <div className="text-xs text-slate-500">{f.type}</div>
                  <div className="mt-1 text-xs font-mono text-slate-400">Matches: {f.matches}</div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-slate-400 text-sm">No PII-like columns detected.</p>
          )}
        </section>

        <section className="card">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Drift (PSI)</h3>
          {Object.keys(drift).length === 0 ? (
            <p className="text-slate-400 text-sm">No drift assessment (no reference dataset).</p>
          ) : (
            <div className="space-y-2 text-sm">
              {Object.entries(drift).map(([k, v]: any) => (
                <div key={k} className="p-2.5 rounded-lg border border-slate-100 bg-white">
                  <div className="flex justify-between mb-1">
                    <span className="font-medium text-slate-800 text-sm">{k}</span>
                    <span className={`text-xs font-bold ${v.psi_level === "stable" ? "text-green-600" : v.psi_level === "moderate" ? "text-amber-600" : "text-red-600"}`}>
                      {v.psi_level}
                    </span>
                  </div>
                  <div className="text-xs text-slate-500">PSI: {v.psi?.toFixed(3)}</div>
                  <div className="text-xs text-slate-500">KS p-value: {v.ks_pvalue?.toFixed(3)}</div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="card lg:col-span-2">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Global feature importance (SHAP)</h3>
          {Object.keys(expl.global_importance || {}).length === 0 ? (
            <p className="text-slate-400 text-sm">No explainability computed.</p>
          ) : (
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th className="text-right">Mean |SHAP|</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(expl.global_importance)
                    .slice(0, 10)
                    .map(([k, v], i) => (
                      <tr key={i} className="hover:bg-slate-50">
                        <td className="font-medium text-slate-700">{k}</td>
                        <td className="text-right font-mono text-xs">{v.toFixed(4)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </section>

      {/* Exports */}
      <section className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-slate-700">Export report</h3>
          <div className="flex gap-2">
            <button
              onClick={() => exportFile("json", "assessment.json", "application/json")}
              className="btn-ghost"
            >
              <Download className="h-4 w-4 mr-2" /> JSON
            </button>
            <button
              onClick={() => exportFile("md", "assessment.md", "text/markdown")}
              className="btn-ghost"
            >
              <Download className="h-4 w-4 mr-2" /> Markdown
            </button>
            <button
              onClick={() => exportFile("card", "model_card.md", "text/markdown")}
              className="btn-ghost"
            >
              <Download className="h-4 w-4 mr-2" /> Model card
            </button>
          </div>
        </div>
      </section>

      <details className="card">
        <summary className="cursor-pointer text-sm font-medium text-slate-600 py-2">Raw assessment JSON</summary>
        <pre className="mt-2 text-xs font-mono text-slate-600 overflow-x-auto bg-slate-50 p-3 rounded border border-slate-100">
          {rawJson}
        </pre>
      </details>
    </div>
  );

  function exportFile(fmt: string, filename: string, mime: string) {
    report(runId, fmt)
      .then((r: any) => {
        const blob = new Blob([r.content], { type: mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      })
      .catch((err: any) => alert("Export failed: " + err.message));
  }
}
