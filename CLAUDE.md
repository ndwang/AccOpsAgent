# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AccOps Agent is an AI-powered agent for particle accelerator operations. It uses LangGraph to orchestrate a human-in-the-loop optimization workflow that reads diagnostics, reasons about machine state, proposes parameter adjustments, and executes changes with safety validation.

## Development Commands

```bash
# Install dependencies (requires uv package manager)
uv sync

# Run tests
uv run pytest

# Run single test file
uv run pytest tests/unit/test_llm.py

# Run single test by name
uv run pytest -k "test_generate_success"

# Run agent CLI (requires config file and OpenAI API key)
uv run python scripts/run_agent.py --config configs/accelerators/example_linac.yaml --backend mock

# Run with specific model and intent
uv run python scripts/run_agent.py \
    --config configs/accelerators/example_linac.yaml \
    --backend mock \
    --model gpt-4 \
    --intent "Minimize horizontal beam size"
```

## Architecture

### LangGraph Workflow (src/accops_agent/graph/)

The agent uses a StateGraph with these nodes executed in sequence:
1. **ingest_diagnostics** - Reads diagnostics and parameters from backend
2. **interpret_diagnostics** - LLM analyzes diagnostic data
3. **reasoning_planning** - LLM generates strategy to achieve user intent
4. **generate_actions** - LLM proposes parameter changes (validated against safety constraints)
5. **human_approval** - Graph interrupts here for human review (approve/reject/modify)
6. **execute_action** - Applies approved parameter changes via backend
7. **verify_results** - LLM assesses effectiveness of changes
8. **decide_continuation** - Determines if goal achieved or more iterations needed

State flows through `AgentState` TypedDict defined in `state.py`. Routing functions in `builder.py` handle conditional edges.

### Backend Abstraction (src/accops_agent/diagnostic_control/)

`AcceleratorBackend` is the abstract interface combining `DiagnosticProvider` and `ControlProvider`:
- **MockBackend** - In-memory simulation for testing
- **TaoBackend** (src/accops_agent/backends/pytao/) - Connects to Bmad/Tao particle physics simulations

Backends read/write via `DiagnosticSnapshot` and `ParameterValue` data models.

### Safety System (src/accops_agent/safety/)

`ConstraintChecker` validates all proposed actions against:
- Parameter limits (min/max from knob config)
- Rate limits (max change per step)
- Global rate limits (max changes per time window)
- Interlocks (halt if diagnostic exceeds threshold)
- Simultaneous change limits

Violations are returned as `Violation` objects with severity levels. In strict mode, errors are raised via `ConstraintViolationError`.

### Configuration (src/accops_agent/config/)

Accelerator configs are YAML files with Pydantic validation (`AcceleratorConfig`):
- `knobs` - Controllable parameters with limits, units, rate limits
- `diagnostics` - Measurements with nominal values, tolerances, alarm thresholds
- `constraints` - Safety rules (interlocks, rate limits)

See `configs/accelerators/example_linac.yaml` for structure.

### LLM Integration (src/accops_agent/llm/)

- `LLMClient` wraps OpenAI SDK, supports any OpenAI-compatible endpoint
- Prompt templates loaded from `configs/prompts/*.yaml`
- Parsers in `parsers.py` extract structured data (JSON actions, issues lists) from LLM output

### CLI (src/accops_agent/cli/)

Human-in-the-loop interface with colored output, action display, and approval prompts.

## Key Patterns

- Configuration via Pydantic models with field validators
- All LLM calls go through `config["configurable"]["llm_client"]` passed to graph nodes
- Backend instance passed via `config["configurable"]["backend"]`
- Temperature settings in `utils/constants.py` vary by task type (lower for action generation)
- Tests mock OpenAI client at `accops_agent.llm.client.OpenAI`

## Environment Variables

- `OPENAI_API_KEY` - API key for LLM
- `OPENAI_BASE_URL` - Custom endpoint for OpenAI-compatible APIs (optional)

## MCP Server (src/accops_agent/mcp_server/)

The PyTao MCP server exposes accelerator controls and diagnostics as tools for AI assistants via the Model Context Protocol.

### Running the MCP Server

```bash
# Using the console script
uv run pytao-mcp

# Or directly with Python
uv run python scripts/run_mcp_server.py
```

### Claude Desktop Configuration

Add to `~/.config/claude/claude_desktop_config.json` (Linux/Mac) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "pytao": {
      "command": "uv",
      "args": ["run", "pytao-mcp"],
      "cwd": "/path/to/AccOpsAgent"
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `tao_connect` | Connect to Tao with a config file |
| `tao_disconnect` | Disconnect from Tao |
| `tao_status` | Get connection status |
| `tao_read_diagnostic` | Read a single diagnostic |
| `tao_read_all_diagnostics` | Read all diagnostics |
| `tao_get_parameter` | Get a parameter value |
| `tao_get_all_parameters` | Get all parameters |
| `tao_set_parameter` | Set a parameter value |
| `tao_run_calculation` | Run beam propagation |
| `tao_execute_command` | Execute raw Tao command |
| `tao_list_knobs` | List available knobs |
| `tao_list_diagnostics` | List available diagnostics |

### Available Resources

When connected, the server exposes these resources:
- `tao://config` - Current accelerator configuration
- `tao://knobs` - List of controllable parameters
- `tao://diagnostics` - List of diagnostic measurements
- `tao://constraints` - Safety constraints and interlocks
