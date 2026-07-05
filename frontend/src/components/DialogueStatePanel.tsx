import type { DialogueState } from "../api/types";

const STATUS_STYLE: Record<string, string> = {
  ACTIVE: "bg-blue-500/20 text-blue-300",
  ACCEPTED: "bg-emerald-500/20 text-emerald-300",
  REJECTED: "bg-red-500/20 text-red-300",
  ESCALATED: "bg-amber-500/20 text-amber-300",
  HANDOFF: "bg-purple-500/20 text-purple-300",
};

export default function DialogueStatePanel({
  state,
}: {
  state: DialogueState | null;
}) {
  const status = state?.status ?? "—";
  const sentiment = state?.sentiment ?? 0; // -1..1
  const pct = ((sentiment + 1) / 2) * 100;

  return (
    <aside className="tile w-72 shrink-0 space-y-4 self-start sticky top-20">
      <div>
        <div className="text-xs uppercase text-gray-400">Status</div>
        <span
          className={`badge mt-1 ${STATUS_STYLE[status] ?? "bg-ink-600 text-gray-300"}`}
        >
          {status}
        </span>
      </div>

      <div>
        <div className="text-xs uppercase text-gray-400 mb-1">Sentiment</div>
        <div className="h-2 rounded-full bg-ink-600 overflow-hidden">
          <div
            className="h-full bg-magenta transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="text-[11px] text-gray-400 mt-1">
          {sentiment.toFixed(2)}
        </div>
      </div>

      <div>
        <div className="text-xs uppercase text-gray-400 mb-1">Intent stack</div>
        <div className="flex flex-col gap-1">
          {(state?.intent_stack ?? []).length === 0 ? (
            <span className="text-[11px] text-gray-500">empty</span>
          ) : (
            (state?.intent_stack ?? []).map((it, i) => (
              <span key={i} className="badge bg-ink-600 text-gray-200">
                {it}
              </span>
            ))
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Stat k="Ladder pos" v={String(state?.ladder_position ?? "—")} />
        <Stat
          k="Authority cap"
          v={state?.authority_cap != null ? `€${state.authority_cap}` : "—"}
        />
      </div>
    </aside>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="bg-ink-900 rounded p-2">
      <div className="text-[10px] uppercase text-gray-500">{k}</div>
      <div className="text-gray-100 text-sm">{v}</div>
    </div>
  );
}
