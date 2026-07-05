import { useEffect, useRef, useState } from "react";
import { api, endpoints } from "../api/client";
import { postSSE } from "../api/sse";
import type { CustomerSummary, NodeEvent } from "../api/types";

export default function RunOne() {
  const [customers, setCustomers] = useState<CustomerSummary[]>([]);
  const [cid, setCid] = useState("");
  const [nodes, setNodes] = useState<NodeEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    api.customers(50, "").then((rows) => {
      setCustomers(rows);
      if (rows[0]) setCid(rows[0].customer_id);
    });
  }, []);

  async function run() {
    if (!cid) return;
    setNodes([]);
    setErr(null);
    setRunning(true);
    abort.current?.abort();
    abort.current = new AbortController();
    try {
      await postSSE(
        endpoints.runOne,
        { customer_id: cid },
        {
          signal: abort.current.signal,
          onEvent: (event, data) => {
            if (event === "node") setNodes((n) => [...n, data as NodeEvent]);
            if (event === "error")
              setErr((data as { message: string }).message);
            if (event === "done") setRunning(false);
          },
          onError: (e) => {
            setErr(String(e));
            setRunning(false);
          },
        },
      );
    } catch (e) {
      setErr(String(e));
      setRunning(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <select
          value={cid}
          onChange={(e) => setCid(e.target.value)}
          className="bg-ink-800 border border-ink-600 rounded-lg px-3 py-2 text-sm"
        >
          {customers.map((c) => (
            <option key={c.customer_id} value={c.customer_id}>
              {c.customer_id}
            </option>
          ))}
        </select>
        <button
          onClick={run}
          disabled={running || !cid}
          className="bg-magenta hover:bg-magenta-600 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-semibold"
        >
          {running ? "Running…" : "Run pipeline"}
        </button>
      </div>

      {err && <div className="tile text-red-400">{err}</div>}

      <ol className="space-y-3">
        {nodes.map((n, i) => (
          <li key={i} className="tile">
            <div className="flex items-center gap-2">
              <span className="badge bg-magenta text-white">{i + 1}</span>
              <span className="font-semibold">{n.node}</span>
            </div>
            <pre className="mt-2 text-[11px] text-gray-300 bg-ink-900 rounded p-2 overflow-x-auto">
              {JSON.stringify(n.payload, null, 2)}
            </pre>
          </li>
        ))}
        {running && (
          <li className="text-gray-400 text-sm animate-pulse">…next node</li>
        )}
      </ol>
    </div>
  );
}
