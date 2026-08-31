import { useAuth } from "@clerk/clerk-react";
import { useEffect, useState } from "react";
import { Navigate, Outlet } from "react-router-dom";

type Status = "loading" | "complete" | "incomplete";

export default function RequireProfile() {
  const { getToken } = useAuth();
  const [status, setStatus] = useState<Status>("loading");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const token = await getToken();
        const res = await fetch("/api/org/profile", {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (cancelled) return;
        if (!res.ok) {
          setStatus("incomplete");
          return;
        }
        const body = await res.json();
        setStatus(body.industry ? "complete" : "incomplete");
      } catch {
        if (cancelled) return;
        setStatus("incomplete");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [getToken]);

  if (status === "loading") {
    return <div className="p-6 text-gray-400">Loading…</div>;
  }
  if (status === "incomplete") {
    return <Navigate to="/setup" replace />;
  }
  return <Outlet />;
}
