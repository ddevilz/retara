import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("@clerk/clerk-react", () => ({
  useAuth: () => ({ getToken: async () => "fake-token" }),
}));
vi.stubGlobal(
  "fetch",
  vi.fn().mockResolvedValue({ ok: true, json: async () => ({ industry: "telecom" }) }),
);

describe("App", () => {
  it("renders the nav shell with all four tabs", () => {
    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    );

    expect(screen.getByText("Magenta")).toBeInTheDocument();
    expect(screen.getByText(/Retain/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Overview" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Customers" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Run-one" })).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Negotiation" }),
    ).toBeInTheDocument();
  });
});
