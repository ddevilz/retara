import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Customer360 } from "../api/types";

export default function CustomerDrawer({
  customerId,
  onClose,
}: {
  customerId: string;
  onClose: () => void;
}) {
  const [data, setData] = useState<Customer360 | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    api.customer(customerId).then(setData).catch((e) => setErr(String(e)));
  }, [customerId]);

  return (
    <div className="fixed inset-0 z-20 flex">
      <div className="flex-1 bg-black/50" onClick={onClose} />
      <aside className="w-[28rem] max-w-full bg-ink-800 border-l border-ink-600 p-5 overflow-y-auto">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold">{customerId}</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-sm"
          >
            ✕ close
          </button>
        </div>

        {err && <div className="mt-4 text-red-400">{err}</div>}
        {!data && !err && <div className="mt-4 text-gray-400">Loading…</div>}

        {data && (
          <>
            <section className="mt-4 grid grid-cols-2 gap-2 text-sm">
              <Field k="Tenure" v={`${data.customer.tenure_months} mo`} />
              <Field k="Contract" v={data.customer.contract} />
              <Field k="Monthly" v={`€${data.customer.monthly_charges.toFixed(2)}`} />
              <Field k="Total" v={`€${data.customer.total_charges.toFixed(0)}`} />
              <Field k="Data util" v={data.customer.data_util_ratio.toFixed(2)} />
              <Field k="Dropped calls" v={data.customer.dropped_call_rate.toFixed(3)} />
              <Field k="NPS" v={data.customer.nps == null ? "—" : String(data.customer.nps)} />
              <Field k="Tickets" v={String(data.customer.support_tickets)} />
              <Field k="Contract end" v={`${data.customer.contract_end_days} d`} />
              <Field k="CLV" v={`€${data.customer.clv.toFixed(0)}`} />
            </section>

            <h3 className="mt-6 mb-2 text-sm font-semibold text-gray-300">
              Audit timeline
            </h3>
            {data.audit.length === 0 ? (
              <div className="text-xs text-gray-500">
                No audit rows — run this customer through the pipeline.
              </div>
            ) : (
              <ol className="relative border-l border-ink-600 pl-4 space-y-4">
                {data.audit.map((row) => (
                  <li key={row.id}>
                    <span className="absolute -left-1.5 w-3 h-3 rounded-full bg-magenta" />
                    <div className="flex items-center gap-2">
                      <span className="badge bg-magenta/20 text-magenta-300">
                        {row.node}
                      </span>
                      {row.holdout && (
                        <span className="badge bg-ink-600 text-gray-300">holdout</span>
                      )}
                      <span className="text-[11px] text-gray-500">{row.ts}</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-300">{row.rationale}</p>
                    <pre className="mt-1 text-[10px] text-gray-400 bg-ink-900 rounded p-2 overflow-x-auto">
                      {JSON.stringify(row.decision, null, 2)}
                    </pre>
                  </li>
                ))}
              </ol>
            )}
          </>
        )}
      </aside>
    </div>
  );
}

function Field({ k, v }: { k: string; v: string }) {
  return (
    <div className="bg-ink-900 rounded p-2">
      <div className="text-[10px] uppercase text-gray-500">{k}</div>
      <div className="text-gray-100">{v}</div>
    </div>
  );
}
