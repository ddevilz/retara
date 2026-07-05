import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ScorecardTile from "./ScorecardTile";

describe("ScorecardTile", () => {
  it("renders label, value, and sub", () => {
    render(<ScorecardTile label="ATE" value="-0.052" sub="95% CI [-0.08, -0.02]" />);
    expect(screen.getByText("ATE")).toBeInTheDocument();
    expect(screen.getByText("-0.052")).toBeInTheDocument();
    expect(screen.getByText(/95% CI/)).toBeInTheDocument();
  });
});
