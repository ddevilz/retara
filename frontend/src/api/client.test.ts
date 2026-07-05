import { describe, expect, it } from "vitest";
import { chatReducer, initialChatState } from "./client";
import type { ChatReply } from "./types";

describe("chatReducer", () => {
  it("appends a user message on send", () => {
    const s = chatReducer(initialChatState, { type: "send", text: "hi" });
    expect(s.messages).toHaveLength(1);
    expect(s.messages[0]).toEqual({ role: "user", text: "hi" });
  });

  it("appends agent reply and updates status from dialogue state", () => {
    const reply: ChatReply = {
      text: "Let me help with that.",
      act: "EMPATHIZE",
      offer: null,
      state: { status: "ACTIVE", sentiment: -0.1 },
    };
    const sent = chatReducer(initialChatState, { type: "send", text: "angry" });
    const got = chatReducer(sent, { type: "reply", reply });
    expect(got.messages).toHaveLength(2);
    expect(got.messages[1]).toEqual({ role: "agent", text: reply.text });
    expect(got.status).toBe("ACTIVE");
  });
});
