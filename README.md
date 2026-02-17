# accel-agent

CLI-based ReAct agent for accelerator operators. See `accelerator_agent_architecture.md` for the full design.

## Setup

Create the virtual environment and install dependencies:

```bash
uv sync
```

Run the agent:

```bash
uv run accel-agent
```

Run tests:

```bash
uv run pytest
```

## Configuration

All configuration lives in `configs/default.toml`. Override per facility or environment as needed.

### LLM

The agent uses [LiteLLM](https://docs.litellm.ai/) for a unified interface to multiple LLM providers. Set the API key for your provider as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
```

Configure the default model and named alternatives in `configs/default.toml`:

```toml
[llm]
model = "anthropic/claude-sonnet-4-20250514"   # default model
max_tokens = 8096

# Named models for runtime switching via /provider
default = "claude"
[llm.models]
claude = "anthropic/claude-sonnet-4-20250514"
gpt4 = "openai/gpt-4o"
```

Model strings use LiteLLM format: `anthropic/claude-...`, `openai/gpt-...`, etc. Switch models at runtime with the `/provider` CLI command (e.g. `/provider gpt4`).

### Control system adapter

The `[adapter]` section selects which control system backend to use:

```toml
[adapter]
type = "simulator"    # "simulator", "epics", or "tango"
```

The simulator adapter is an in-memory mock for development and testing. Real facility adapters implement the `ControlSystemAdapter` ABC in `accel_agent/adapters/base.py`.

### Operator identity

The operator name is recorded in the audit log for every plan approval/rejection. It resolves in order:

1. `operator` key in `configs/default.toml` (top-level)
2. `ACCEL_AGENT_OPERATOR` environment variable
3. `USER` or `USERNAME` environment variable
4. Falls back to `"unknown"`

```toml
# Optional: set a fixed operator name
operator = "jsmith"
```

### Safety limits

`configs/safety_limits.toml` defines allowed value ranges for device writes. Any write outside the defined range is rejected before reaching the control system. Devices without defined limits cannot be written to at all.

Limits can be defined per device or with wildcard patterns:

```toml
["LINAC:QUAD:*:CURRENT"]
min = -50.0
max = 50.0
units = "A"

["LINAC:RF:*:PHASE"]
min = -180.0
max = 180.0
units = "deg"
```

### Code execution sandbox

The `[sandbox]` section configures the Python code execution environment used by the `execute_code` and `run_script` tools:

```toml
[sandbox]
scripts_dir = "scripts"          # directory for reusable analysis scripts
work_dir = "logs/workspace"      # writable output directory (plots, CSVs)
data_dirs = []                   # read-only data directories (archives, databases)
timeout_seconds = 120            # max execution time per script
max_memory_mb = 2048             # memory limit
```

- Scripts can read from `data_dirs` and `scripts_dir`, and write only to `work_dir`.
- Control system libraries, network modules, and shell access are blocked by static analysis.
- New scripts written via `execute_code` with `save_as` are saved to `scripts_dir` for reuse.

### Facility context (AGENT.md)

`AGENT.md` is injected into the LLM system prompt to provide facility-specific knowledge. Edit it to describe your accelerator, device naming conventions, common procedures, and safety notes:

```toml
agent_md = "AGENT.md"    # path to facility context file
```

Example content:

```markdown
# AGENT.md — My Facility

## Facility
Our facility uses EPICS with Channel Access.

## Naming Convention
Devices follow SECTOR:TYPE:INSTANCE:ATTRIBUTE, e.g.:
- LINAC:BPM:01:X — BPM horizontal position
- LINAC:QUAD:01:CURRENT — quadrupole current setpoint

## Common Tasks
- Shift summary: read elog for the last 8 hours
- Orbit plot: read BPM archive data by sector

## Safety Notes
- Quad current changes > 5A require section lead approval

## Data Locations
- Archive: /data/archive/ (HDF5, one per day)
- E-log: /data/elog/elog.db (SQLite)
```

The agent uses this context to understand device names, choose appropriate tools, and follow facility conventions without needing a vector database or RAG system.

## CLI commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/context` | Display loaded AGENT.md |
| `/provider <name>` | Switch LLM model (e.g. `/provider gpt4`) |
| `/clear` | Clear conversation history |
| `/quit` | Exit the agent |
