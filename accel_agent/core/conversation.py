"""Message history management for the ReAct loop."""

import json


class Conversation:
    """Stores user, assistant, and tool result messages for LLM context."""

    def __init__(self):
        self._messages: list[dict] = []
        self._tool_call_count = 0

    @property
    def messages(self) -> list[dict]:
        return self._messages

    @property
    def tool_call_count(self) -> int:
        return self._tool_call_count

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self, content: str | None, tool_calls: list | None = None
    ) -> None:
        msg: dict = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input) if isinstance(tc.input, dict) else tc.input,
                    },
                }
                for tc in tool_calls
            ]
        self._messages.append(msg)
        if tool_calls:
            self._tool_call_count += len(tool_calls)

    def add_tool_results(self, results: list[dict]) -> None:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": r["call_id"],
                "content": json.dumps(r["result"]) if not isinstance(r["result"], str) else r["result"],
            }
            for r in results
        ]
        self._messages.append({"role": "user", "content": content})
