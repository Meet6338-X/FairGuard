import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Layout from "./Layout.tsx";
import Dashboard from "./pages/Dashboard.tsx";
import Train from "./pages/Train.tsx";
import Assess from "./pages/Assess.tsx";
import Results from "./pages/Results.tsx";
import History from "./pages/History.tsx";

const API_BASE = "http://127.0.0.1:8000/api";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: "train", element: <Train /> },
      { path: "assess", element: <Assess /> },
      { path: "results/:runId", element: <Results /> },
      { path: "history", element: <History /> },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
