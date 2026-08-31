import { useAuth, useClerk, useOrganization } from "@clerk/clerk-react";
import { useEffect, useState, type FormEvent } from "react";

const INDUSTRIES = [
  { value: "telecom", label: "Telecom", available: true },
  { value: "retail", label: "Retail", available: false },
  { value: "insurance", label: "Insurance", available: false },
];

export default function Setup() {
  const { getToken } = useAuth();
  const { organization } = useOrganization();
  const { signOut } = useClerk();
  const [name, setName] = useState("");

  useEffect(() => {
    if (organization?.name) setName(organization.name);
  }, [organization?.name]);
  const [industry, setIndustry] = useState("telecom");
  const [budget, setBudget] = useState("");
  const [contact, setContact] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const token = await getToken();
      const res = await fetch("/api/org/profile", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          name,
          industry,
          monthly_token_budget: budget ? Number(budget) : null,
          admin_contact_email: contact || null,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
      }
      setSubmitted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setSubmitting(false);
    }
  }

  if (submitted) {
    return (
      <div className="max-w-lg mx-auto text-center py-16">
        <h1 className="text-2xl font-bold tracking-tight mb-2 text-ink-50">
          You&apos;re all set up.
        </h1>
        <p className="text-gray-400">The full app is coming soon.</p>
      </div>
    );
  }

  return (
    <div className="max-w-lg mx-auto">
      <h1 className="text-2xl font-bold tracking-tight mb-1">Company profile</h1>
      <p className="text-gray-400 mb-6">
        Tell us about your organization to get started.
      </p>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        {error && (
          <div className="text-sm text-red-400 bg-red-950/40 border border-red-900 rounded-lg px-3 py-2">
            {error}
          </div>
        )}
        <label className="flex flex-col gap-1">
          <span className="text-sm text-gray-300">Company name</span>
          <input
            className="bg-ink-800 border border-ink-600 rounded-lg px-3 py-2"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm text-gray-300">Industry</span>
          <select
            className="bg-ink-800 border border-ink-600 rounded-lg px-3 py-2"
            value={industry}
            onChange={(e) => setIndustry(e.target.value)}
          >
            {INDUSTRIES.map((i) => (
              <option key={i.value} value={i.value} disabled={!i.available}>
                {i.label}
                {!i.available ? " (coming soon)" : ""}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm text-gray-300">
            Monthly LLM token budget (blank = unlimited)
          </span>
          <input
            type="number"
            min="1"
            className="bg-ink-800 border border-ink-600 rounded-lg px-3 py-2 font-mono"
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm text-gray-300">Admin contact email (optional)</span>
          <input
            type="email"
            className="bg-ink-800 border border-ink-600 rounded-lg px-3 py-2"
            value={contact}
            onChange={(e) => setContact(e.target.value)}
          />
        </label>
        <button
          type="submit"
          disabled={submitting}
          className="mt-2 bg-magenta hover:bg-magenta-600 disabled:opacity-50 text-white rounded-lg px-4 py-2 font-semibold active:translate-y-[1px]"
        >
          {submitting ? "Saving…" : "Save and continue"}
        </button>
      </form>
      <button
        type="button"
        onClick={() => signOut()}
        className="mt-4 text-sm text-gray-500 hover:text-gray-300 underline"
      >
        Sign out
      </button>
    </div>
  );
}
