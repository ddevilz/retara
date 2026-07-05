import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Rung } from "../api/types";

const COLORS: Record<string, string> = {
  noaction: "#6B7280",
  rules: "#9CA3AF",
  risk_rules: "#FF4FA8",
  agent_s1: "#C2005F",
  agent: "#E20074",
};

export default function AblationLadder({ rungs }: { rungs: Rung[] }) {
  // ATE is negative (churn reduction). Plot magnitude so bars grow with effect.
  const data = rungs.map((r) => ({
    policy: r.policy,
    ate: r.scorecard.ate,
    // ErrorBar wants [lowDelta, highDelta] distances from the value.
    err: [
      Math.abs(r.scorecard.ate - r.scorecard.ci_low),
      Math.abs(r.scorecard.ci_high - r.scorecard.ate),
    ] as [number, number],
  }));

  return (
    <div className="tile h-80">
      <div className="text-sm font-semibold mb-2">
        Ablation ladder — ATE (churn reduction) with 95% CI
      </div>
      <ResponsiveContainer width="100%" height="90%">
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2A2A38" />
          <XAxis dataKey="policy" stroke="#9CA3AF" fontSize={12} />
          <YAxis stroke="#9CA3AF" fontSize={12} />
          <Tooltip
            contentStyle={{ background: "#14141B", border: "1px solid #2A2A38" }}
            formatter={(v) => (typeof v === "number" ? v.toFixed(4) : String(v))}
          />
          <Bar dataKey="ate" radius={[4, 4, 0, 0]}>
            {data.map((d) => (
              <Cell key={d.policy} fill={COLORS[d.policy] ?? "#E20074"} />
            ))}
            <ErrorBar
              dataKey="err"
              width={6}
              strokeWidth={2}
              stroke="#FF4FA8"
              direction="y"
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
