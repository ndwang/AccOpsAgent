"""Terminal UI (prompt_toolkit / rich)."""

import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown

console = Console()


class AgentCLI:
    def __init__(self, agent):
        self.agent = agent
        self.session = PromptSession(history=FileHistory(".agent_history"))

    async def run(self):
        console.print(
            "[bold green]Accelerator Agent[/] ready. " "Type /help for commands.\n"
        )

        while True:
            try:
                user_input = await asyncio.to_thread(
                    self.session.prompt, "operator> "
                )
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                await self._handle_command(user_input)
                continue

            with console.status("[bold cyan]Thinking...[/]"):
                response = await self.agent.run(user_input)

            console.print(Markdown(response))
            console.print()

    async def _handle_command(self, cmd):
        parts = cmd.strip().split()
        match parts[0]:
            case "/help":
                console.print(
                    "/help      — Show this message\n"
                    "/context   — Show loaded AGENT.md\n"
                    "/history   — Show conversation summary\n"
                    "/provider  — Switch LLM model (e.g., /provider gpt4)\n"
                    "/clear     — Clear conversation\n"
                    "/quit      — Exit"
                )
            case "/provider":
                if len(parts) > 1:
                    self.agent.switch_provider(parts[1])
                    console.print(f"Switched to [bold]{parts[1]}[/]")
            case "/context":
                path = self.agent.context.agent_md_path
                if path.exists():
                    console.print(Markdown(path.read_text()))
                else:
                    console.print("[dim]No AGENT.md loaded.[/]")
            case "/clear":
                self.agent.clear_conversation()
                console.print("Conversation cleared.")
            case "/quit":
                raise SystemExit()
