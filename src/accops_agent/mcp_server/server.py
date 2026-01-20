"""MCP server implementation for PyTao accelerator backend.

This module provides an MCP server that exposes PyTao accelerator
controls and diagnostics as tools for AI assistants.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    TextContent,
    Tool,
)
from pydantic import AnyUrl

from .pytao import TaoBackend
from ..config.schema import AcceleratorConfig

logger = logging.getLogger(__name__)


class PyTaoMCPServer:
    """MCP server wrapping PyTao backend functionality.

    This server exposes tools for:
    - Connection management (connect, disconnect, status)
    - Diagnostic reading (single and bulk)
    - Parameter control (get, set)
    - Beam calculations
    - Raw Tao command execution

    Attributes:
        server: MCP Server instance
        backend: TaoBackend instance (None until connected)
        config: AcceleratorConfig (None until loaded)
        config_path: Path to loaded configuration file
    """

    def __init__(self):
        """Initialize the MCP server."""
        self.server = Server("pytao-mcp")
        self.backend: Optional[TaoBackend] = None
        self.config: Optional[AcceleratorConfig] = None
        self.config_path: Optional[str] = None

        self._register_tools()
        self._register_resources()

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available tools."""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="tao_connect",
                        description="Connect to Tao simulator with an accelerator configuration file. Must be called before using other Tao tools.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "config_path": {
                                    "type": "string",
                                    "description": "Path to accelerator YAML configuration file",
                                }
                            },
                            "required": ["config_path"],
                        },
                    ),
                    Tool(
                        name="tao_disconnect",
                        description="Disconnect from Tao simulator and release resources.",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="tao_status",
                        description="Get current Tao connection status and loaded configuration info.",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="tao_read_diagnostic",
                        description="Read a single diagnostic measurement from the accelerator (e.g., beam position, beam size).",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "diagnostic_name": {
                                    "type": "string",
                                    "description": "Name of the diagnostic to read (e.g., 'BPM1_X', 'BEAM_SIZE_X')",
                                }
                            },
                            "required": ["diagnostic_name"],
                        },
                    ),
                    Tool(
                        name="tao_read_all_diagnostics",
                        description="Read all configured diagnostics from the accelerator. Returns a list of all measurements.",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="tao_get_parameter",
                        description="Get the current value of a control parameter (knob) from the accelerator.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "parameter_name": {
                                    "type": "string",
                                    "description": "Name of the parameter to read (e.g., 'QF1_K1', 'HCOR1_KICK')",
                                }
                            },
                            "required": ["parameter_name"],
                        },
                    ),
                    Tool(
                        name="tao_get_all_parameters",
                        description="Get all configured parameter values from the accelerator. Returns a dictionary of all knob values.",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="tao_set_parameter",
                        description="Set a control parameter (knob) to a new value. The value must be within configured limits.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "parameter_name": {
                                    "type": "string",
                                    "description": "Name of the parameter to set (e.g., 'QF1_K1')",
                                },
                                "value": {
                                    "type": "number",
                                    "description": "New value to set for the parameter",
                                },
                            },
                            "required": ["parameter_name", "value"],
                        },
                    ),
                    Tool(
                        name="tao_run_calculation",
                        description="Run Tao beam propagation calculation after parameter changes. This propagates the effects of parameter changes through the lattice.",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="tao_execute_command",
                        description="Execute a raw Tao command. For advanced users who need direct Tao access. Returns the command output.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "Tao command to execute (e.g., 'show lat', 'python lat_list')",
                                }
                            },
                            "required": ["command"],
                        },
                    ),
                    Tool(
                        name="tao_list_knobs",
                        description="List all available control knobs with their limits and descriptions.",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="tao_list_diagnostics",
                        description="List all available diagnostics with their descriptions and alarm thresholds.",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ]
            )

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            try:
                result = await self._handle_tool_call(name, arguments)
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2))]
                )
            except Exception as e:
                logger.exception(f"Tool call failed: {name}")
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps({"error": str(e)}))]
                )

    def _register_resources(self) -> None:
        """Register MCP resources."""

        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available resources."""
            resources = []

            if self.config is not None:
                resources.extend([
                    Resource(
                        uri=AnyUrl("tao://config"),
                        name="Accelerator Configuration",
                        description="Current accelerator configuration including all knobs, diagnostics, and constraints",
                        mimeType="application/json",
                    ),
                    Resource(
                        uri=AnyUrl("tao://knobs"),
                        name="Available Knobs",
                        description="List of all controllable parameters with limits",
                        mimeType="application/json",
                    ),
                    Resource(
                        uri=AnyUrl("tao://diagnostics"),
                        name="Available Diagnostics",
                        description="List of all diagnostic measurements",
                        mimeType="application/json",
                    ),
                    Resource(
                        uri=AnyUrl("tao://constraints"),
                        name="Safety Constraints",
                        description="Safety constraints and interlocks",
                        mimeType="application/json",
                    ),
                ])

            return ListResourcesResult(resources=resources)

        @self.server.read_resource()
        async def read_resource(uri: AnyUrl) -> ReadResourceResult:
            """Read a resource."""
            uri_str = str(uri)

            if self.config is None:
                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=json.dumps({"error": "No configuration loaded. Use tao_connect first."}),
                        )
                    ]
                )

            if uri_str == "tao://config":
                config_data = {
                    "name": self.config.name,
                    "description": self.config.description,
                    "tao_init_file": self.config.tao_init_file,
                    "knobs_count": len(self.config.knobs),
                    "diagnostics_count": len(self.config.diagnostics),
                    "constraints_count": len(self.config.constraints),
                    "metadata": self.config.metadata,
                }
                return ReadResourceResult(
                    contents=[TextContent(type="text", text=json.dumps(config_data, indent=2))]
                )

            elif uri_str == "tao://knobs":
                knobs_data = [
                    {
                        "name": k.name,
                        "description": k.description,
                        "min_value": k.min_value,
                        "max_value": k.max_value,
                        "unit": k.unit,
                        "rate_limit": k.rate_limit,
                        "element_type": k.element_type,
                    }
                    for k in self.config.knobs
                ]
                return ReadResourceResult(
                    contents=[TextContent(type="text", text=json.dumps(knobs_data, indent=2))]
                )

            elif uri_str == "tao://diagnostics":
                diag_data = [
                    {
                        "name": d.name,
                        "description": d.description,
                        "measurement_type": d.measurement_type,
                        "unit": d.unit,
                        "nominal_value": d.nominal_value,
                        "tolerance": d.tolerance,
                        "alarm_threshold": d.alarm_threshold,
                    }
                    for d in self.config.diagnostics
                ]
                return ReadResourceResult(
                    contents=[TextContent(type="text", text=json.dumps(diag_data, indent=2))]
                )

            elif uri_str == "tao://constraints":
                constraint_data = [
                    {
                        "constraint_id": c.constraint_id,
                        "description": c.description,
                        "constraint_type": c.constraint_type,
                        "parameters": c.parameters,
                        "enabled": c.enabled,
                    }
                    for c in self.config.constraints
                ]
                return ReadResourceResult(
                    contents=[TextContent(type="text", text=json.dumps(constraint_data, indent=2))]
                )

            else:
                return ReadResourceResult(
                    contents=[
                        TextContent(type="text", text=json.dumps({"error": f"Unknown resource: {uri_str}"}))
                    ]
                )

    async def _handle_tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle a tool call and return the result.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Result dictionary
        """
        if name == "tao_connect":
            return await self._tool_connect(arguments["config_path"])

        elif name == "tao_disconnect":
            return await self._tool_disconnect()

        elif name == "tao_status":
            return await self._tool_status()

        elif name == "tao_read_diagnostic":
            return await self._tool_read_diagnostic(arguments["diagnostic_name"])

        elif name == "tao_read_all_diagnostics":
            return await self._tool_read_all_diagnostics()

        elif name == "tao_get_parameter":
            return await self._tool_get_parameter(arguments["parameter_name"])

        elif name == "tao_get_all_parameters":
            return await self._tool_get_all_parameters()

        elif name == "tao_set_parameter":
            return await self._tool_set_parameter(
                arguments["parameter_name"], arguments["value"]
            )

        elif name == "tao_run_calculation":
            return await self._tool_run_calculation()

        elif name == "tao_execute_command":
            return await self._tool_execute_command(arguments["command"])

        elif name == "tao_list_knobs":
            return await self._tool_list_knobs()

        elif name == "tao_list_diagnostics":
            return await self._tool_list_diagnostics()

        else:
            return {"error": f"Unknown tool: {name}"}

    async def _tool_connect(self, config_path: str) -> dict[str, Any]:
        """Connect to Tao with the given configuration."""
        # Load configuration
        config_file = Path(config_path)
        if not config_file.exists():
            return {"success": False, "error": f"Configuration file not found: {config_path}"}

        try:
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)

            self.config = AcceleratorConfig(**config_data)
            self.config_path = config_path

            # Create and initialize backend
            self.backend = TaoBackend(self.config)
            success = self.backend.initialize()

            if success:
                return {
                    "success": True,
                    "message": f"Connected to Tao with configuration: {self.config.name}",
                    "config_name": self.config.name,
                    "knobs_count": len(self.config.knobs),
                    "diagnostics_count": len(self.config.diagnostics),
                }
            else:
                error_msg = self.backend.connection.last_error or "Unknown error"
                self.backend = None
                return {"success": False, "error": f"Failed to initialize Tao: {error_msg}"}

        except Exception as e:
            self.backend = None
            self.config = None
            return {"success": False, "error": f"Failed to load configuration: {e}"}

    async def _tool_disconnect(self) -> dict[str, Any]:
        """Disconnect from Tao."""
        if self.backend is None:
            return {"success": True, "message": "Already disconnected"}

        success = self.backend.shutdown()
        self.backend = None

        return {
            "success": success,
            "message": "Disconnected from Tao" if success else "Disconnect failed",
        }

    async def _tool_status(self) -> dict[str, Any]:
        """Get connection status."""
        if self.backend is None:
            return {
                "connected": False,
                "config_loaded": self.config is not None,
                "config_name": self.config.name if self.config else None,
            }

        return {
            "connected": self.backend.is_connected(),
            "config_loaded": True,
            "config_name": self.config.name,
            "config_path": self.config_path,
            "knobs_count": len(self.config.knobs),
            "diagnostics_count": len(self.config.diagnostics),
        }

    async def _tool_read_diagnostic(self, diagnostic_name: str) -> dict[str, Any]:
        """Read a single diagnostic."""
        if self.backend is None or not self.backend.is_connected():
            return {"error": "Not connected to Tao. Use tao_connect first."}

        try:
            snapshot = self.backend.read_diagnostic(diagnostic_name)
            return {
                "diagnostic_name": snapshot.diagnostic_name,
                "value": snapshot.value,
                "unit": snapshot.unit,
                "status": snapshot.status.value,
                "message": snapshot.message,
                "timestamp": snapshot.timestamp,
            }
        except ValueError as e:
            return {"error": str(e)}
        except RuntimeError as e:
            return {"error": str(e)}

    async def _tool_read_all_diagnostics(self) -> dict[str, Any]:
        """Read all diagnostics."""
        if self.backend is None or not self.backend.is_connected():
            return {"error": "Not connected to Tao. Use tao_connect first."}

        try:
            snapshots = self.backend.read_all_diagnostics()
            return {
                "diagnostics": [
                    {
                        "diagnostic_name": s.diagnostic_name,
                        "value": s.value,
                        "unit": s.unit,
                        "status": s.status.value,
                        "message": s.message,
                    }
                    for s in snapshots
                ],
                "count": len(snapshots),
            }
        except RuntimeError as e:
            return {"error": str(e)}

    async def _tool_get_parameter(self, parameter_name: str) -> dict[str, Any]:
        """Get a parameter value."""
        if self.backend is None or not self.backend.is_connected():
            return {"error": "Not connected to Tao. Use tao_connect first."}

        try:
            param = self.backend.get_parameter(parameter_name)
            return {
                "parameter_name": param.parameter_name,
                "value": param.value,
                "unit": param.unit,
                "timestamp": param.timestamp,
            }
        except ValueError as e:
            return {"error": str(e)}
        except RuntimeError as e:
            return {"error": str(e)}

    async def _tool_get_all_parameters(self) -> dict[str, Any]:
        """Get all parameter values."""
        if self.backend is None or not self.backend.is_connected():
            return {"error": "Not connected to Tao. Use tao_connect first."}

        try:
            params = self.backend.get_all_parameters()
            return {
                "parameters": {
                    name: {
                        "value": p.value,
                        "unit": p.unit,
                    }
                    for name, p in params.items()
                },
                "count": len(params),
            }
        except RuntimeError as e:
            return {"error": str(e)}

    async def _tool_set_parameter(self, parameter_name: str, value: float) -> dict[str, Any]:
        """Set a parameter value."""
        if self.backend is None or not self.backend.is_connected():
            return {"error": "Not connected to Tao. Use tao_connect first."}

        result = self.backend.set_parameter(parameter_name, value)
        return {
            "success": result.success,
            "action_type": result.action_type,
            "message": result.message,
            "error": result.error,
            "parameters": result.parameters,
        }

    async def _tool_run_calculation(self) -> dict[str, Any]:
        """Run Tao calculation."""
        if self.backend is None or not self.backend.is_connected():
            return {"error": "Not connected to Tao. Use tao_connect first."}

        success = self.backend.run_calculation()
        return {
            "success": success,
            "message": "Calculation completed" if success else "Calculation failed",
        }

    async def _tool_execute_command(self, command: str) -> dict[str, Any]:
        """Execute a raw Tao command."""
        if self.backend is None or not self.backend.is_connected():
            return {"error": "Not connected to Tao. Use tao_connect first."}

        try:
            output = self.backend.connection.execute_command(command)
            return {"command": command, "output": output}
        except RuntimeError as e:
            return {"error": str(e)}

    async def _tool_list_knobs(self) -> dict[str, Any]:
        """List all available knobs."""
        if self.config is None:
            return {"error": "No configuration loaded. Use tao_connect first."}

        return {
            "knobs": [
                {
                    "name": k.name,
                    "description": k.description,
                    "min_value": k.min_value,
                    "max_value": k.max_value,
                    "unit": k.unit,
                    "rate_limit": k.rate_limit,
                    "element_type": k.element_type,
                }
                for k in self.config.knobs
            ],
            "count": len(self.config.knobs),
        }

    async def _tool_list_diagnostics(self) -> dict[str, Any]:
        """List all available diagnostics."""
        if self.config is None:
            return {"error": "No configuration loaded. Use tao_connect first."}

        return {
            "diagnostics": [
                {
                    "name": d.name,
                    "description": d.description,
                    "measurement_type": d.measurement_type,
                    "unit": d.unit,
                    "nominal_value": d.nominal_value,
                    "tolerance": d.tolerance,
                    "alarm_threshold": d.alarm_threshold,
                }
                for d in self.config.diagnostics
            ],
            "count": len(self.config.diagnostics),
        }

    async def run(self) -> None:
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


def create_server() -> PyTaoMCPServer:
    """Create a new PyTaoMCPServer instance.

    Returns:
        Configured PyTaoMCPServer
    """
    return PyTaoMCPServer()


def main() -> None:
    """Entry point for the MCP server."""
    logging.basicConfig(level=logging.INFO)
    server = create_server()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
