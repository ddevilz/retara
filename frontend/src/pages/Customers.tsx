import { useEffect, useState } from "react";
import CustomerDrawer from "../components/CustomerDrawer";
import { api } from "../api/client";
import type { CustomerSummary } from "../api/types";

export default function Customers() {
  const [rows, setRows] = useState<CustomerSummary[]>([]);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    const t = setTimeout(() => {
      api
        .customers(50, search)
        .then(setRows)
        .finally(() => setLoading(false));
    }, 200); // debounce
    return () => clearTimeout(t);
  }, [search]);

  return (
    <div className="space-y-4">
      <input
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search customer id…"
        className="w-full max-w-sm bg-ink-800 border border-ink-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-magenta"
      />

      <div className="tile p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-ink-700 text-gray-400 text-xs uppercase">
            <tr>
              <th className="text-left px-4 py-2">Customer</th>
              <th className="text-right px-4 py-2">Tenure</th>
              <th className="text-left px-4 py-2">Contract</th>
              <th className="text-right px-4 py-2">Monthly</th>
              <th className="text-right px-4 py-2">CLV</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.customer_id}
                onClick={() => setSelected(r.customer_id)}
                className="border-t border-ink-700 hover:bg-ink-700/50 cursor-pointer"
              >
                <td className="px-4 py-2 font-medium">{r.customer_id}</td>
                <td className="px-4 py-2 text-right">{r.tenure_months} mo</td>
                <td className="px-4 py-2">{r.contract}</td>
                <td className="px-4 py-2 text-right">
                  €{r.monthly_charges.toFixed(2)}
                </td>
                <td className="px-4 py-2 text-right">€{r.clv.toFixed(0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {loading && <div className="px-4 py-3 text-gray-400 text-sm">Loading…</div>}
        {!loading && rows.length === 0 && (
          <div className="px-4 py-3 text-gray-500 text-sm">No customers.</div>
        )}
      </div>

      {selected && (
        <CustomerDrawer customerId={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}
