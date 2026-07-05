// 1:1 mirror of magenta.api.schemas (backend/src/magenta/api/schemas.py).
// Keep in sync by hand — there is no codegen step for this API surface.

// NOTE: brief drift — backend's Rung/ExperimentRequest Literal has FIVE
// policy values ("noaction" | "rules" | "risk_rules" | "agent_s1" | "agent"),
// not four. "agent_s1" is a real ladder rung (single-shot agent, no repair
// loop) distinct from "agent" — verified against schemas.py and
// routes_stream.py's _DEPS_REQUIRED_POLICIES set. Included here so the type
// doesn't silently reject a real backend payload.
export type Policy = "noaction" | "rules" | "risk_rules" | "agent_s1" | "agent";

export interface ScorecardData {
  churn_treatment: number;
  churn_holdout: number;
  ate: number;
  ci_low: number;
  ci_high: number;
  wasted_offer_rate: number;
  sleeping_dogs_contacted: number;
  euros_retained: number;
  offer_spend: number;
  acceptance_rate: number;
  n_treatment: number;
  n_holdout: number;
  offers_made: number;
}

export interface Rung {
  policy: Policy;
  scorecard: ScorecardData;
}

export interface Scorecards {
  rungs: Rung[];
}

export interface CustomerSummary {
  customer_id: string;
  tenure_months: number;
  contract: string;
  monthly_charges: number;
  total_charges: number;
  data_util_ratio: number;
  dropped_call_rate: number;
  nps: number | null;
  support_tickets: number;
  contract_end_days: number;
  clv: number;
  gross_margin: number;
}

export interface AuditRow {
  id: number;
  ts: string;
  customer_id: string;
  node: string;
  decision: Record<string, unknown>;
  rationale: string;
  holdout: boolean;
}

export interface Customer360 {
  customer: CustomerSummary;
  audit: AuditRow[];
}

export interface NodeEvent {
  node: string;
  payload: Record<string, unknown>;
}

export interface DialogueState {
  status?: string;
  sentiment?: number;
  ladder_position?: number;
  authority_cap?: number;
  intent_stack?: string[];
  [k: string]: unknown;
}

export interface ChatReply {
  text: string;
  act: string | null;
  offer: Record<string, unknown> | null;
  state: DialogueState | null;
}

export interface ChatStartResponse {
  session_id: string;
  mode: string;
  customer_id: string;
  archetype: string | null;
}

export type ChatMode = "persona" | "human";

export interface ChatMessage {
  role: "user" | "agent";
  text: string;
}
