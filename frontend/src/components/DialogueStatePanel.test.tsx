import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import DialogueStatePanel from "./DialogueStatePanel";

describe("DialogueStatePanel", () => {
  it("renders placeholders when state is null", () => {
    render(<DialogueStatePanel state={null} />);
    expect(screen.getAllByText("—")).toHaveLength(3); // status, ladder pos, authority cap
    expect(screen.getByText("0.00")).toBeInTheDocument();
    expect(screen.getByText("empty")).toBeInTheDocument();
  });

  it("renders status badge, sentiment, intent stack, and stats from state", () => {
    render(
      <DialogueStatePanel
        state={{
          status: "ESCALATED",
          sentiment: -0.5,
          intent_stack: ["RETAIN", "DISCOUNT"],
          ladder_position: 2,
          authority_cap: 15,
        }}
      />,
    );

    expect(screen.getByText("ESCALATED")).toBeInTheDocument();
    expect(screen.getByText("-0.50")).toBeInTheDocument();
    expect(screen.getByText("RETAIN")).toBeInTheDocument();
    expect(screen.getByText("DISCOUNT")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("€15")).toBeInTheDocument();
  });

  it("falls back to a neutral badge style for an unrecognized status", () => {
    render(<DialogueStatePanel state={{ status: "WEIRD" }} />);
    const badge = screen.getByText("WEIRD");
    expect(badge.className).toContain("bg-ink-600");
  });
});
