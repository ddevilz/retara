import { useReducer, useRef, useState } from "react";
import DialogueStatePanel from "../components/DialogueStatePanel";
import { api, chatReducer, endpoints, initialChatState } from "../api/client";
import { postSSE } from "../api/sse";
import type { ChatMode, ChatReply, DialogueState } from "../api/types";

const ARCHETYPES = [
  "BILL_SHOCK",
  "CONFUSED",
  "PRICE_HAGGLER",
  "NETWORK_COMPLAINER",
  "COMPETITOR_BLUFFER",
  "SLEEPING_DOG",
];

export default function Negotiation() {
  const [mode, setMode] = useState<ChatMode>("persona");
  const [archetype, setArchetype] = useState(ARCHETYPES[0]);
  const [sid, setSid] = useState<string | null>(null);
  const [ui, dispatch] = useReducer(chatReducer, initialChatState);
  const [dstate, setDstate] = useState<DialogueState | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);

  async function start() {
    const res = await api.chatStart({
      mode,
      archetype: mode === "persona" ? archetype : undefined,
    });
    setSid(res.session_id);
    setDstate(null);
  }

  async function send() {
    if (!sid || !draft.trim()) return;
    const text = draft.trim();
    setDraft("");
    dispatch({ type: "send", text });
    setBusy(true);
    abort.current = new AbortController();
    try {
      await postSSE(
        endpoints.chatTurn(sid),
        { text },
        {
          signal: abort.current.signal,
          onEvent: (event, data) => {
            if (event === "reply") {
              const reply = data as ChatReply;
              dispatch({ type: "reply", reply });
              setDstate(reply.state);
            }
            if (event === "done") setBusy(false);
          },
          onError: () => setBusy(false),
        },
      );
    } catch {
      setBusy(false);
    }
  }

  return (
    <div className="flex gap-4">
      <div className="flex-1 space-y-3">
        {!sid ? (
          <div className="tile flex items-center gap-3">
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as ChatMode)}
              className="bg-ink-900 border border-ink-600 rounded px-2 py-1 text-sm"
            >
              <option value="persona">Persona</option>
              <option value="human">Human</option>
            </select>
            {mode === "persona" && (
              <select
                value={archetype}
                onChange={(e) => setArchetype(e.target.value)}
                className="bg-ink-900 border border-ink-600 rounded px-2 py-1 text-sm"
              >
                {ARCHETYPES.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            )}
            <button
              onClick={start}
              className="bg-magenta hover:bg-magenta-600 px-4 py-1.5 rounded-lg text-sm font-semibold"
            >
              Start negotiation
            </button>
          </div>
        ) : (
          <>
            <div className="tile h-[60vh] overflow-y-auto flex flex-col gap-2">
              {ui.messages.map((m, i) => (
                <div
                  key={i}
                  className={`max-w-[75%] rounded-2xl px-3 py-2 text-sm ${
                    m.role === "user"
                      ? "self-end bg-ink-600 text-gray-100"
                      : "self-start bg-magenta text-white"
                  }`}
                >
                  {m.text}
                </div>
              ))}
              {busy && (
                <div className="self-start text-gray-400 text-xs animate-pulse">
                  agent typing…
                </div>
              )}
            </div>
            <div className="flex gap-2">
              <input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && send()}
                placeholder="Type a message…"
                className="flex-1 bg-ink-800 border border-ink-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-magenta"
              />
              <button
                onClick={send}
                disabled={busy}
                className="bg-magenta hover:bg-magenta-600 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-semibold"
              >
                Send
              </button>
            </div>
          </>
        )}
      </div>

      {sid && <DialogueStatePanel state={dstate} />}
    </div>
  );
}
