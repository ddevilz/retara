import type {
  ChatMessage,
  ChatReply,
  ChatStartResponse,
  Customer360,
  CustomerSummary,
  Policy,
  Scorecards,
} from "./types";

const BASE = ""; // dev proxy handles /api; same-origin in prod

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const api = {
  health: () => getJSON<{ status: string }>("/api/health"),
  scorecards: () => getJSON<Scorecards>("/api/scorecards"),
  customers: (limit = 50, search = "") =>
    getJSON<CustomerSummary[]>(
      `/api/customers?limit=${limit}&search=${encodeURIComponent(search)}`,
    ),
  customer: (id: string) => getJSON<Customer360>(`/api/customers/${id}`),
  chatStart: (body: {
    mode: "persona" | "human";
    archetype?: string;
    customer_id?: string;
  }) => postJSON<ChatStartResponse>("/api/chat/start", body),
};

export const endpoints = {
  runOne: "/api/run-one",
  experiment: "/api/experiment",
  chatTurn: (sid: string) => `/api/chat/${sid}/turn`,
};

// Brief drift: backend's Rung/ExperimentRequest Literal (schemas.py) has
// FIVE rungs, including "agent_s1" between "risk_rules" and "agent" —
// confirmed against routes_stream.py's _DEPS_REQUIRED_POLICIES set, which
// treats risk_rules/agent_s1/agent as the three rungs needing a real
// GraphDeps. Listed here in ladder order so ExperimentTile (11.3+) can map
// over the real set of rungs instead of a stale four-value list.
export const POLICIES: Policy[] = [
  "noaction",
  "rules",
  "risk_rules",
  "agent_s1",
  "agent",
];

// ---- pure chat reducer (unit-tested) ----
export type ChatAction =
  | { type: "send"; text: string }
  | { type: "reply"; reply: ChatReply };

export interface ChatUIState {
  messages: ChatMessage[];
  status: string;
}

export const initialChatState: ChatUIState = { messages: [], status: "IDLE" };

export function chatReducer(
  state: ChatUIState,
  action: ChatAction,
): ChatUIState {
  switch (action.type) {
    case "send":
      return {
        ...state,
        messages: [...state.messages, { role: "user", text: action.text }],
      };
    case "reply":
      return {
        status: action.reply.state?.status ?? state.status,
        messages: [
          ...state.messages,
          { role: "agent", text: action.reply.text },
        ],
      };
    default:
      return state;
  }
}
