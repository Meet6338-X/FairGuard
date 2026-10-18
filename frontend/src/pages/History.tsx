import { useEffect, useState } from "react";
import { history } from "../api.ts";
import { Clock, FileText, ShieldCheck, AlertTriangle } from "lucide-react";
import { Link } from "react-router-dom";

const LEVEL_COLORS: Record<string, string> = {
  Low: "#22c55e",
  Medium: "#f59e0b",
  High: "#ef4444",
  Critical: "#7f1d1d",
};

export default function HistoryPage() {
  const [runs, setRuns] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    history()
      .then((h) => {
        setRuns(h.runs || []);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="page-container">
        <div className="flex flex-col items-center justify-center py-20">
          <div className="animate-spin h-8 w-8 border-4 border-blue-200 border-t-blue-600 rounded-full mb-4" />
          <p className="text-slate-500">Loading history…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <div className="text-center py-20">
          <AlertTriangle className="h-10 w-10 text-red-500 mx-auto mb-3" />
          <h2 className="text-lg font-semibold text-red-700">Error loading history</h2>
          <p className="text-gray-500">{error}</p>
          <Link to="/" className="mt-4 inline-block text-blue-600 hover:underline">
            Back to dashboard
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container space-y-6">
      <div className="flex items-center gap-2">
        <Clock className="h-5 w-5 text-blue-600" />
        <h1 className="text-2xl font-bold">Assessment History</h1>
      </div>

      {runs.length === 0 && (
        <div className="text-center py-20">
          <ShieldCheck className="h-12 w-12 mx-auto mb-4 opacity-30" />
          <p className="text-slate-500 text-sm">No assessments yet. Run one from <span className="font-medium">Assess Model</span>.</p>
          <Link to="/assess" className="mt-4 inline-block text-blue-600 hover:underline font-medium">
            Go to Assess Model →
          </Link>
        </div>
      )}

      <div className="card">
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Model</th>
                <th>Dataset</th>
                <th className="text-right">Risk score</th>
                <th>Level</th>
                <th>Decision</th>
                <th>Assessed at</th>
                <th className="text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {runs.map((r) => (
                <tr key={r.run_id} className="hover:bg-slate-50 transition-colors">
                  <td className="px-4 py-3 font-mono text-xs text-slate-600">{r.run_id}</td>
                  <td className="px-4 py-3 text-sm">{r.model_id || "—"}</td>
                  <td className="px-4 py-3 text-sm text-slate-600">{r.dataset_id || "—"}</td>
                  <td className="px-4 py-3 text-right font-mono text-xs">
                    {r.risk?.score != null ? r.risk.score.toFixed(3) : "—"}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${r.risk?.level === "High" ? "bg-red-100 text-red-700" : r.risk?.level === "Medium" ? "bg-amber-100 text-amber-700" : "bg-green-100 text-green-700"}`}
                    >
                      {r.risk?.level || "—"}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-600">{r.decision || "—"}</td>
                  <td className="px-4 py-3 text-sm text-slate-500 font-mono">{r.assessed_at || "—"}</td>
                  <td className="px-4 py-3 text-right">
                    <Link to={`/results/${r.run_id}`} className="text-blue-600 hover:underline text-sm font-medium">
                      View
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
