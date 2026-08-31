import { NavLink, Route, Routes } from "react-router-dom";
import Overview from "./pages/Overview";
import Customers from "./pages/Customers";
import RunOne from "./pages/RunOne";
import Negotiation from "./pages/Negotiation";
import Setup from "./pages/Setup";
import RequireProfile from "./components/RequireProfile";

const tabs = [
  { to: "/", label: "Overview", end: true },
  { to: "/customers", label: "Customers" },
  { to: "/run-one", label: "Run-one" },
  { to: "/negotiation", label: "Negotiation" },
];

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-ink-600 bg-ink-800/60 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center gap-6">
          <span className="text-lg font-bold">
            <span className="text-magenta">Magenta</span> Retain
          </span>
          <nav className="flex gap-1">
            {tabs.map((t) => (
              <NavLink
                key={t.to}
                to={t.to}
                end={t.end}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-lg text-sm ${
                    isActive
                      ? "bg-magenta text-white"
                      : "text-gray-300 hover:bg-ink-700"
                  }`
                }
              >
                {t.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-6 py-6">
        <Routes>
          <Route path="/setup" element={<Setup />} />
          <Route element={<RequireProfile />}>
            <Route path="/" element={<Overview />} />
            <Route path="/customers" element={<Customers />} />
            <Route path="/run-one" element={<RunOne />} />
            <Route path="/negotiation" element={<Negotiation />} />
          </Route>
        </Routes>
      </main>
    </div>
  );
}
