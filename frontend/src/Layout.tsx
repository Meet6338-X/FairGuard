import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ShieldCheck, BarChart3, Bot, Play, Clock, FileText } from "lucide-react";

const NAV = [
  { to: "/", label: "Dashboard", icon: <BarChart3 className="h-4 w-4" /> },
  { to: "/train", label: "Train Model", icon: <Bot className="h-4 w-4" /> },
  { to: "/assess", label: "Assess Model", icon: <Play className="h-4 w-4" /> },
  { to: "/history", label: "History", icon: <Clock className="h-4 w-4" /> },
];

export default function Layout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <div className="flex h-screen w-screen bg-slate-50 text-slate-900">
      <aside
        className={`w-64 bg-white border-r border-slate-200 flex flex-col transition-all duration-300 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="p-5 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-blue-600" />
            <span className="text-xl font-bold tracking-tight">FairGuard AI</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">Responsible AI Workbench</p>
        </div>
        <nav className="p-2 space-y-1 flex-1">
          {NAV.map((item) => {
            const active = location.pathname === item.to;
            return (
              <button
                key={item.to}
                className={`sidebar-link w-full ${active ? "sidebar-link-active" : "sidebar-link-inactive"} ${!active && "border-transparent"}`}
                onClick={() => navigate(item.to)}
              >
                {item.icon}
                <span className="ml-1">{item.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="p-4 border-t border-slate-200 text-xs text-slate-500 bg-slate-50">
          <div className="flex items-center gap-1 mb-1">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span>Local only · no paid APIs</span>
          </div>
        </div>
      </aside>

      <button
        className="fixed top-4 left-4 z-50 p-2 bg-white border border-slate-200 rounded-lg text-slate-700 hover:bg-slate-100 transition-colors md:hidden"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label="Toggle sidebar"
      >
        <span className="sr-only">Open sidebar</span>
        <span className="text-lg">☰</span>
      </button>

      <main className="flex-1 overflow-y-auto p-6 md:pl-0">
        <div className="page-container">{children}</div>
      </main>
    </div>
  );
}
