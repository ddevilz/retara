import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Customers from "./Customers";
import { api } from "../api/client";
import type { Customer360, CustomerSummary } from "../api/types";

vi.mock("../api/client", () => ({
  api: {
    customers: vi.fn(),
    customer: vi.fn(),
  },
}));

const rows: CustomerSummary[] = [
  {
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
];

const customer360: Customer360 = {
  customer: rows[0],
  audit: [],
};

describe("Customers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.customers).mockResolvedValue(rows);
    vi.mocked(api.customer).mockResolvedValue(customer360);
  });

  it("loads and renders customer rows from api.customers", async () => {
    render(<Customers />);

    expect(await screen.findByText("C-1001")).toBeInTheDocument();
    expect(screen.getByText("24 mo")).toBeInTheDocument();
    expect(screen.getByText("MONTH_TO_MONTH")).toBeInTheDocument();
    expect(api.customers).toHaveBeenCalledWith(50, "");
  });

  it("debounces search input before calling api.customers again", async () => {
    vi.useFakeTimers();
    render(<Customers />);

    await vi.waitFor(() => expect(api.customers).toHaveBeenCalledTimes(1));

    const input = screen.getByPlaceholderText("Search customer id…");
    fireEvent.change(input, { target: { value: "C-1001" } });

    // debounce window (200ms) not yet elapsed
    expect(api.customers).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(200);

    expect(api.customers).toHaveBeenCalledWith(50, "C-1001");
    vi.useRealTimers();
  });

  it("opens the drawer with 360 detail on row click", async () => {
    render(<Customers />);

    const row = await screen.findByText("C-1001");
    fireEvent.click(row);

    await waitFor(() => expect(api.customer).toHaveBeenCalledWith("C-1001"));
    expect(await screen.findByText("No audit rows — run this customer through the pipeline.")).toBeInTheDocument();
  });

  it("shows an empty state when no customers are returned", async () => {
    vi.mocked(api.customers).mockResolvedValue([]);
    render(<Customers />);

    expect(await screen.findByText("No customers.")).toBeInTheDocument();
  });
});
