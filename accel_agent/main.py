"""CLI entry point for the accelerator operator agent."""

import asyncio
import tomllib
from pathlib import Path


def main():
    config_path = Path("configs/default.toml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    from accel_agent.core.agent import OrchestratorAgent
    from accel_agent.cli.app import AgentCLI

    agent = OrchestratorAgent(config)
    cli = AgentCLI(agent)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
