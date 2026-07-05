interface Props {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}

export default function ScorecardTile({ label, value, sub, accent }: Props) {
  return (
    <div className="tile">
      <div className="text-xs uppercase tracking-wide text-gray-400">{label}</div>
      <div
        className={`mt-1 text-2xl font-bold ${accent ? "text-magenta" : "text-white"}`}
      >
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-gray-400">{sub}</div>}
    </div>
  );
}
