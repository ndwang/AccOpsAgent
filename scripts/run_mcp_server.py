#!/usr/bin/env python3
"""Run the PyTao MCP server.

This script starts the MCP server that exposes PyTao accelerator
controls and diagnostics as tools for AI assistants.

Usage:
    uv run python scripts/run_mcp_server.py

The server communicates via stdio and can be used with MCP-compatible
clients such as Claude Desktop or other AI assistants.

Configuration for Claude Desktop (~/.config/claude/claude_desktop_config.json):

{
  "mcpServers": {
    "pytao": {
      "command": "uv",
      "args": ["run", "python", "scripts/run_mcp_server.py"],
      "cwd": "/path/to/AccOpsAgent"
    }
  }
}
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from accops_agent.mcp_server import main

if __name__ == "__main__":
    main()
