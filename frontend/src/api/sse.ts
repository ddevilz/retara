// POST-SSE via @microsoft/fetch-event-source — the browser's native
// EventSource can only issue GET requests, but our stream endpoints
// (/api/run-one, /api/experiment, /api/chat/{sid}/turn) require a JSON body.
import { fetchEventSource } from "@microsoft/fetch-event-source";

export interface SSEHandlers {
  onEvent: (event: string, data: unknown) => void;
  onError?: (err: unknown) => void;
  onClose?: () => void;
  signal?: AbortSignal;
}

export async function postSSE(
  url: string,
  body: unknown,
  handlers: SSEHandlers,
): Promise<void> {
  await fetchEventSource(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: handlers.signal,
    openWhenHidden: true,
    onmessage(msg) {
      // Backend's sse_event() always JSON-encodes payload into msg.data
      // (see magenta/api/sse.py:to_json) — but fall back to the raw string
      // if a future event ever isn't JSON, rather than throwing mid-stream.
      let parsed: unknown = msg.data;
      try {
        parsed = JSON.parse(msg.data);
      } catch {
        /* leave as string */
      }
      handlers.onEvent(msg.event || "message", parsed);
    },
    onerror(err) {
      handlers.onError?.(err);
      throw err; // stop retry loop; we handle errors ourselves
    },
    onclose() {
      handlers.onClose?.();
    },
  });
}
