import { useEffect, useState } from "react";
import AblationLadder from "../components/AblationLadder";
import ScorecardTile from "../components/ScorecardTile";
import { api } from "../api/client";
import type { Rung, Scorecards } from "../api/types";

const eur = (n: number) =>
  new Intl.NumberFormat("en-EU", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(n);
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

export default function Overview() {
  const [data, setData] = useState<Scorecards | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.scorecards().then(setData).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="tile text-red-400">Failed to load: {err}</div>;
  if (!data) return <div className="text-gray-400">Loading…</div>;
  if (data.rungs.length === 0)
    return (
      <div className="tile text-gray-300">
        No scorecards yet — run <code>magenta ablation</code> to populate{" "}
        <code>data/scorecards.json</code>.
      </div>
    );

  const agent: Rung =
    data.rungs.find((r) => r.policy === "agent") ?? data.rungs[data.rungs.length - 1];
  const sc = agent.scorecard;
  const totalSleepingDogs = data.rungs.reduce(
    (a, r) => a + r.scorecard.sleeping_dogs_contacted,
    0,
  );

  return (
    <div className="space-y-6">
      {totalSleepingDogs === 0 ? (
        <div className="tile border-emerald-600/50 bg-emerald-950/30 text-emerald-300">
          ✓ 0 guardrail violations — no sleeping-dogs contacted across all rungs.
        </div>
      ) : (
        <div className="tile border-red-600/50 bg-red-950/30 text-red-300">
          ⚠ {totalSleepingDogs} sleeping-dog contacts detected.
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <ScorecardTile
          label="ATE (churn reduction)"
          value={sc.ate.toFixed(3)}
          sub={`95% CI [${sc.ci_low.toFixed(3)}, ${sc.ci_high.toFixed(3)}]`}
          accent
        />
        <ScorecardTile label="€ retained (net)" value={eur(sc.euros_retained)} />
        <ScorecardTile label="Wasted-offer rate" value={pct(sc.wasted_offer_rate)} />
        <ScorecardTile label="Acceptance rate" value={pct(sc.acceptance_rate)} />
      </div>

      <AblationLadder rungs={data.rungs} />
    </div>
  );
}
