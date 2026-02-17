# accel-agent

CLI-based ReAct agent for accelerator operators. See `accelerator_agent_architecture.md` for the full design.

## Setup with uv

Create the virtual environment and install dependencies:

```bash
uv sync
```

Activate the venv (Windows):

```powershell
.venv\Scripts\activate
```

Configure your LLM (LiteLLM): set API keys in the environment for the providers you use, e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. Edit `configs/default.toml` to set `[llm] model` or named `[llm.models]` (e.g. `claude`, `gpt4`). Use `/provider gpt4` in the CLI to switch models.

Run the agent:

```bash
uv run accel-agent
# or, with venv activated:
accel-agent
```

Run tests:

```bash
uv run pytest
```
