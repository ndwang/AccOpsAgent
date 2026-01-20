"""Module entrypoint for `python -m accops_agent.mcp_server`.

The agent's MCP client launches the server as a Python module:
`python -m accops_agent.mcp_server`.
"""

from .server import main


if __name__ == "__main__":
    main()

