"""Core think→act→observe ReAct cycle."""

from dataclasses import dataclass, field

from accel_agent.llm.base import LLMResponse


@dataclass
class ReActState:
    steps: list = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 30


class ReActLoop:
    """
    Core think→act→observe cycle. Provider-agnostic — takes an LLM
    interface and a tool registry, loops until the LLM produces a
    final text response or hits the iteration limit.
    """

    def __init__(self, llm, tool_registry, context_manager):
        self.llm = llm
        self.tools = tool_registry
        self.context = context_manager

    async def run(self, user_message: str, conversation) -> str:
        state = ReActState()
        conversation.add_user_message(user_message)
        system_prompt = self.context.build_system_prompt()

        while state.iteration < state.max_iterations:
            state.iteration += 1

            response: LLMResponse = await self.llm.complete(
                system=system_prompt,
                messages=conversation.messages,
                tools=self.tools.schemas(),
            )

            if not response.tool_calls:
                conversation.add_assistant_message(response.text)
                return response.text or ""

            tool_results = []
            for call in response.tool_calls:
                tool = self.tools.get(call.name)
                result = await tool.execute(call.input)
                tool_results.append({"call_id": call.id, "result": result})

            conversation.add_assistant_message(response.text, response.tool_calls)
            conversation.add_tool_results(tool_results)

        return "Reached maximum iterations. Please refine your request."
