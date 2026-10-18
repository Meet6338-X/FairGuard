import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { datasets, models } from "../api.ts";
import { Database, Cpu, ShieldCheck, Clock, BarChart3, FileText } from "lucide-react";

const Stat = ({ icon: Icon, label, value, color = "blue" }) => (
  <div className="stat-card">
    <div className={`p-2.5 rounded-lg bg-${color}-50 text-${color}-600`}>
      <Icon className="h-5 w-5" />
    </div>
    <div>
      <div className="text-xs text-slate-500 uppercase tracking-wider font-semibold">{label}</div>
      <div className="text-lg font-bold text-slate-900">{value}</div>
    </div>
  </div>
);

export default function Dashboard() {
  const [ds, setDs] = useState([]);
  const [md, setMd] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([datasets(), models()])
      .then(([d, m]) => { setDs(d.datasets || []); setMd(m.models || []); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="page-container">
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin h-8 w-8 border-4 border-blue-200 border-t-blue-600 rounded-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="page-container space-y-8">
      <div>
        <h1 className="text-2xl font-bold mb-1">Dashboard</h1>
        <p className="text-slate-500 text-sm">Datasets and trained models available for assessment.</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <Stat icon={Database} label="Datasets" value={ds.length} color="blue" />
        <Stat icon={Cpu} label="Trained Models" value={md.length} color="indigo" />
        <Stat icon={ShieldCheck} label="Accuracy floor" value="≥ 0.70" color="green" />
        <Stat icon={Clock} label="Tracking" value="MLflow local" color="purple" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="card">
          <div className="flex items-center gap-3 mb-4">
            <BarChart3 className="h-5 w-5 text-blue-600" />
            <h2 className="text-lg font-semibold">Datasets</h2>
          </div>
          {ds.length === 0 ? (
            <p className="text-slate-500 text-sm">No datasets found. Run <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">python scripts/download_datasets.py</code> to fetch open datasets from UCI and ProPublica.</p>
          ) : (
            <div className="space-y-1">
              {ds.map((d) => (
                <div key={d} className="flex items-center justify-between p-2.5 rounded-lg hover:bg-slate-50 cursor-pointer group transition-colors">
                  <Link
                    to={{ pathname: "/assess", search: `?dataset=${encodeURIComponent(d)}` }}
                    className="text-slate-700 font-medium hover:text-blue-600 transition-colors"
                  >
                    {d}
                  </Link>
                  <span className="text-xs text-slate-400 group-hover:text-blue-600">Open →</span>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="card">
          <div className="flex items-center gap-3 mb-4">
            <Cpu className="h-5 w-5 text-indigo-600" />
            <h2 className="text-lg font-semibold">Trained Models</h2>
          </div>
          {md.length === 0 ? (
            <div className="text-center py-8 text-slate-500">
              <p>No models yet. Train your first model to get started.</p>
              <Link to="/train" className="mt-3 inline-block text-blue-600 hover:underline font-medium">Train model →</Link>
            </div>
          ) : (
            <div className="space-y-1">
              {md.map((m) => (
                <div key={m} className="flex items-center justify-between p-2.5 rounded-lg hover:bg-slate-50 transition-colors">
                  <span className="text-slate-700 font-medium text-sm truncate max-w-[200px]">{m}</span>
                  <span className="text-xs text-slate-400">Ready</span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      <section className="card">
        <div className="flex items-center gap-3 mb-4">
          <FileText className="h-5 w-5 text-green-600" />
          <h2 className="text-lg font-semibold">What FairGuard checks</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
          {[
            ["Performance", "Accuracy, precision, recall, F1, ROC-AUC, confusion matrix", BarChart3],
            ["Fairness", "Demographic parity & equalized odds by group", ShieldCheck],
            ["Explainability", "SHAP global importance & local contributions", BarChart3],
            ["Privacy", "PII scan (email, phone, Aadhaar, card, IP)", ShieldCheck],
            ["Drift", "PSI + KS test vs reference data", Clock],
            ["Risk", "Weighted score → Low / Medium / High", ShieldCheck],
          ].map(([t, d, Icon]) => (
            <div key={t} className="p-4 rounded-lg border border-slate-100 bg-slate-50">
              <div className="flex items-center gap-2 mb-1">
                <Icon className="h-4 w-4 text-blue-500" />
                <div className="font-semibold text-slate-800">{t}</div>
              </div>
              <div className="text-slate-500 mt-1 text-xs">{d}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
