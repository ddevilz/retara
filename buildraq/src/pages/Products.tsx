import { useEffect, useState } from "react";
import { BadgeRow, RollText, ArrowCircle } from "../components/ui";
import Nav from "../components/Nav";

// Minimal mirror of magenta.api.schemas — only the fields this page reads.
// Verified against frontend/src/api/types.ts (ScorecardData / Rung / Scorecards).
interface ScorecardData {
  ate: number;
  euros_retained: number;
  wasted_offer_rate: number;
}

interface Rung {
  policy: string;
  scorecard: ScorecardData;
}

interface Scorecards {
  rungs: Rung[];
}

/** Last rung with policy "agent"; falls back to whatever rungs exist. */
function pickRung(rungs: Rung[]): Rung | null {
  if (rungs.length === 0) return null;
  const agentRungs = rungs.filter((r) => r.policy === "agent");
  return agentRungs.length > 0 ? agentRungs[agentRungs.length - 1] : rungs[rungs.length - 1];
}

function formatAte(ate: number): string {
  return `${ate >= 0 ? "+" : ""}${ate.toFixed(4)}`;
}

function formatEuros(v: number): string {
  return `€${Math.round(v).toLocaleString()}`;
}

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-gray-50 p-4">
      <p className="text-[12px] text-gray-500">{label}</p>
      <p className="mt-1 text-[20px] font-semibold text-gray-900">{value}</p>
    </div>
  );
}

export default function Products() {
  const [live, setLive] = useState(false);
  const [rung, setRung] = useState<Rung | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/health");
        if (!res.ok) throw new Error("offline");
        setLive(true);
      } catch {
        setLive(false);
        return;
      }
      try {
        const res = await fetch("/api/scorecards");
        if (res.ok) {
          const data: Scorecards = await res.json();
          setRung(pickRung(data.rungs));
        }
      } catch {
        // health was fine but scorecards fetch failed — leave rung null, stay "Live"
      }
    })();
  }, []);

  return (
    <>
      <div className="bg-white">
        <Nav />
      </div>

      <section className="bg-[#F5F5F5] pb-16 pt-8 sm:pb-20 sm:pt-10 lg:pb-28 lg:pt-14">
        <div className="mx-auto max-w-[1440px]">
          <BadgeRow number="1" label="Our products" borderClass="border-gray-300" />

          <h2 className="mb-10 px-5 text-[clamp(1.75rem,7vw,4.2rem)] font-medium leading-[1.08] tracking-[-0.03em] text-gray-900 sm:mb-14 sm:px-8 sm:text-[clamp(2.5rem,5vw,4.2rem)] lg:mb-16 lg:px-12">
            Products built by BuildRaq
          </h2>

          <div className="px-5 sm:px-8 lg:px-12">
            <div className="rounded-2xl bg-white p-6 sm:p-10 lg:p-12">
              <h3 className="text-[22px] font-semibold text-gray-900 sm:text-[26px]">
                Magenta Retain
              </h3>
              <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-gray-600 sm:text-[16px]">
                A causal AI retention agent for telecoms: predicts whom we can save,
                negotiates the offer, and proves the lift against an untouched holdout.
              </p>

              {/* Live data strip */}
              <div className="mt-8 rounded-xl border border-gray-200 p-5 sm:p-6">
                <div className="flex items-center gap-1.5 text-[13px] text-gray-600">
                  <span
                    className={`h-2 w-2 rounded-full ${live ? "bg-green-500" : "bg-gray-400"}`}
                  />
                  {live ? "Live" : "Offline"}
                </div>

                {rung ? (
                  <>
                    <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3">
                      <StatTile label="ATE" value={formatAte(rung.scorecard.ate)} />
                      <StatTile
                        label="€ retained (net)"
                        value={formatEuros(rung.scorecard.euros_retained)}
                      />
                      <StatTile
                        label="wasted-offer rate"
                        value={formatPct(rung.scorecard.wasted_offer_rate)}
                      />
                    </div>
                    <p className="mt-3 text-[12px] text-gray-500">
                      simulated cohort, causal vs holdout
                    </p>
                  </>
                ) : (
                  <p className="mt-4 text-[13px] text-gray-500">
                    {live
                      ? "No scorecard data yet."
                      : "Backend offline — start `magenta serve` to see live cohort stats."}
                  </p>
                )}
              </div>

              <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:gap-5">
                <a
                  href="http://localhost:5173"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group flex w-fit items-center gap-3 rounded-full bg-[#F26522] py-2 pl-5 pr-2 text-[13px] font-medium text-white transition-colors hover:bg-[#e05a1a] sm:pl-6 sm:text-[14px]"
                >
                  <RollText text="Open live dashboard" />
                  <ArrowCircle sizeClass="h-7 w-7 sm:h-8 sm:w-8" iconClassName="text-[#F26522]" iconSize={16} />
                </a>
                <a
                  href="#"
                  className="group flex w-fit items-center gap-3 rounded-full bg-gray-900 py-2 pl-5 pr-2 text-[13px] font-medium text-white"
                >
                  <RollText text="How it works" />
                  <ArrowCircle sizeClass="h-7 w-7 sm:h-8 sm:w-8" iconClassName="text-gray-900" iconSize={16} />
                </a>
              </div>
            </div>

            <p className="mt-4 text-[13px] text-gray-500">
              LangGraph · LightGBM · Thompson sampling · FastAPI · React
            </p>
          </div>
        </div>
      </section>
    </>
  );
}
