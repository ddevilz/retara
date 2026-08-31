import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import Setup from "./Setup";

vi.mock("@clerk/clerk-react", () => ({
  useAuth: () => ({ getToken: async () => "fake-token" }),
  useOrganization: () => ({ organization: { name: "Acme Telecom" } }),
}));

describe("Setup", () => {
  it("pre-fills Company name from the Clerk organization", () => {
    vi.stubGlobal("fetch", vi.fn());
    render(
      <MemoryRouter>
        <Setup />
      </MemoryRouter>,
    );
    expect(screen.getByLabelText("Company name")).toHaveValue("Acme Telecom");
    vi.unstubAllGlobals();
  });

  it("submits the form and PUTs the profile", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <MemoryRouter>
        <Setup />
      </MemoryRouter>,
    );
    fireEvent.change(screen.getByLabelText("Company name"), {
      target: { value: "Acme Telecom" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save/i }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/org/profile",
        expect.objectContaining({ method: "PUT" }),
      ),
    );
    vi.unstubAllGlobals();
  });

  it("shows an inline error and keeps entered values on failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({ detail: "unsupported industry" }),
      }),
    );

    render(
      <MemoryRouter>
        <Setup />
      </MemoryRouter>,
    );
    fireEvent.change(screen.getByLabelText("Company name"), {
      target: { value: "Acme Telecom" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save/i }));

    await waitFor(() =>
      expect(screen.getByText("unsupported industry")).toBeInTheDocument(),
    );
    expect(screen.getByLabelText("Company name")).toHaveValue("Acme Telecom");
    vi.unstubAllGlobals();
  });
});
