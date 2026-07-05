import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import App from "./App";

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
