import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import RequireProfile from "./RequireProfile";

vi.mock("@clerk/clerk-react", () => ({
  useAuth: () => ({ getToken: async () => "fake-token" }),
}));

function renderGuarded() {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route path="/setup" element={<div>Setup page</div>} />
        <Route element={<RequireProfile />}>
          <Route path="/" element={<div>Protected home</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe("RequireProfile", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("redirects to /setup when the profile is incomplete", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => ({ industry: null }) }),
    );
    renderGuarded();
    await waitFor(() => expect(screen.getByText("Setup page")).toBeInTheDocument());
  });

  it("renders the protected route when the profile is complete", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => ({ industry: "telecom" }) }),
    );
    renderGuarded();
    await waitFor(() =>
      expect(screen.getByText("Protected home")).toBeInTheDocument(),
    );
  });
});
