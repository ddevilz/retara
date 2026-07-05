import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CustomerDrawer from "./CustomerDrawer";
import { api } from "../api/client";
import type { AuditRow, Customer360 } from "../api/types";

vi.mock("../api/client", () => ({
  api: {
    customer: vi.fn(),
  },
}));

const auditRow: AuditRow = {
  id: 1,
  ts: "2026-06-30T12:00:00Z",
  customer_id: "C-1001",
  node: "DIAGNOSE",
  decision: { risk_band: "HIGH" },
  rationale: "High churn risk driven by dropped calls and low NPS.",
  holdout: false,
};

const customer360: Customer360 = {
  customer: {
    customer_id: "C-1001",
    tenure_months: 24,
    contract: "MONTH_TO_MONTH",
    monthly_charges: 59.99,
    total_charges: 1439.76,
    data_util_ratio: 0.72,
    dropped_call_rate: 0.012,
    nps: 6,
    support_tickets: 2,
    contract_end_days: 0,
    clv: 820,
    gross_margin: 0.35,
  },
  audit: [auditRow],
};

describe("CustomerDrawer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows a loading state, then renders 360 fields and audit timeline", async () => {
    vi.mocked(api.customer).mockResolvedValue(customer360);
    render(<CustomerDrawer customerId="C-1001" onClose={() => {}} />);

    expect(screen.getByText("Loading…")).toBeInTheDocument();

    expect(await screen.findByText("24 mo")).toBeInTheDocument();
    expect(screen.getByText("€59.99")).toBeInTheDocument();
    expect(screen.getByText("DIAGNOSE")).toBeInTheDocument();
    expect(
      screen.getByText("High churn risk driven by dropped calls and low NPS."),
    ).toBeInTheDocument();
    expect(screen.getByText(/"risk_band": "HIGH"/)).toBeInTheDocument();
    expect(api.customer).toHaveBeenCalledWith("C-1001");
  });

  it("renders a fallback dash for a null NPS", async () => {
    vi.mocked(api.customer).mockResolvedValue({
      ...customer360,
      customer: { ...customer360.customer, nps: null },
    });
    render(<CustomerDrawer customerId="C-1001" onClose={() => {}} />);

    expect(await screen.findByText("—")).toBeInTheDocument();
  });

  it("renders an empty-audit message when there are no audit rows", async () => {
    vi.mocked(api.customer).mockResolvedValue({ ...customer360, audit: [] });
    render(<CustomerDrawer customerId="C-1001" onClose={() => {}} />);

    expect(
      await screen.findByText("No audit rows — run this customer through the pipeline."),
    ).toBeInTheDocument();
  });

  it("shows an error message when the fetch fails", async () => {
    vi.mocked(api.customer).mockRejectedValue(new Error("500 Internal Server Error"));
    render(<CustomerDrawer customerId="C-1001" onClose={() => {}} />);

    expect(await screen.findByText(/500 Internal Server Error/)).toBeInTheDocument();
  });

  it("calls onClose when the close button or backdrop is clicked", async () => {
    vi.mocked(api.customer).mockResolvedValue(customer360);
    const onClose = vi.fn();
    render(<CustomerDrawer customerId="C-1001" onClose={onClose} />);

    fireEvent.click(screen.getByText("✕ close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
