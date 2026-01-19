"""MCP-based backend for accelerator control via MCP server.

This is the primary production backend implementation. All accelerator access
is performed via MCP tools exposed by the PyTao MCP server (mcp_server/server.py).

The PyTao backend internals (TaoBackend, TaoConnection, etc.) live in
mcp_server/pytao and are only used by the MCP server process, not directly
by agent code.
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..config.schema import (
    AcceleratorConfig,
    ConstraintDefinition,
    DiagnosticDefinition,
    KnobDefinition,
)
from .data_models import ActionResult, DiagnosticSnapshot, DiagnosticStatus, ParameterValue
from .interfaces import AcceleratorBackend

logger = logging.getLogger(__name__)


class MCPBackend(AcceleratorBackend):
    """Backend that communicates with the MCP server for accelerator control.

    This backend spawns the PyTao MCP server as a subprocess and communicates
    with it using the MCP protocol. It bridges the synchronous AcceleratorBackend
    interface to the async MCP client.

    Attributes:
        config_path: Path to the accelerator configuration YAML file
        _session: MCP ClientSession when connected
        _connected: Whether the backend is currently connected
    """

    def __init__(self, config_path: str):
        """Initialize MCP backend.

        Args:
            config_path: Path to accelerator YAML configuration file.
                        The config will be loaded via the MCP server.
        """
        self.config_path = str(Path(config_path).absolute())
        self._session: Optional[ClientSession] = None
        self._connected = False
        self._config: Optional[AcceleratorConfig] = None
        self._read_stream = None
        self._write_stream = None
        self._cm = None

    @property
    def config(self) -> AcceleratorConfig:
        """Get the accelerator configuration.

        Returns:
            AcceleratorConfig loaded from MCP server

        Raises:
            RuntimeError: If not connected
        """
        if self._config is None:
            raise RuntimeError("Not connected. Call initialize() first.")
        return self._config

    def _run_async(self, coro):
        """Run an async coroutine synchronously.

        Args:
            coro: Coroutine to run

        Returns:
            Result of the coroutine
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're in an async context - create a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            # No running loop, use asyncio.run
            return asyncio.run(coro)

    async def _create_session(self) -> ClientSession:
        """Create and initialize an MCP client session.

        Returns:
            Initialized ClientSession
        """
        # Get the path to the MCP server entry point
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "accops_agent.mcp_server"],
        )

        self._cm = stdio_client(server_params)
        self._read_stream, self._write_stream = await self._cm.__aenter__()

        session = ClientSession(self._read_stream, self._write_stream)
        await session.__aenter__()
        await session.initialize()

        return session

    async def _close_session(self) -> None:
        """Close the MCP client session."""
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None

        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
            self._cm = None

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call an MCP tool and return the result.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Parsed JSON result from the tool

        Raises:
            RuntimeError: If not connected or tool call fails
        """
        if self._session is None:
            raise RuntimeError("Not connected. Call initialize() first.")

        result = await self._session.call_tool(tool_name, arguments or {})

        # Parse the text content from the result
        if result.content and len(result.content) > 0:
            text_content = result.content[0]
            if hasattr(text_content, "text"):
                return json.loads(text_content.text)

        return {}

    async def _read_resource(self, uri: str) -> Dict[str, Any]:
        """Read an MCP resource.

        Args:
            uri: Resource URI (e.g., "tao://config")

        Returns:
            Parsed JSON content from the resource

        Raises:
            RuntimeError: If not connected or resource read fails
        """
        if self._session is None:
            raise RuntimeError("Not connected. Call initialize() first.")

        result = await self._session.read_resource(uri)

        if result.contents and len(result.contents) > 0:
            content = result.contents[0]
            if hasattr(content, "text"):
                return json.loads(content.text)

        return {}

    async def _initialize_async(self) -> bool:
        """Async implementation of initialize.

        Returns:
            True if initialization successful
        """
        try:
            # Create MCP session
            self._session = await self._create_session()

            # Connect to Tao via MCP server
            result = await self._call_tool("tao_connect", {"config_path": self.config_path})

            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                logger.error(f"Failed to connect to Tao: {error}")
                await self._close_session()
                return False

            # Reconstruct AcceleratorConfig from MCP resources
            self._config = await self._reconstruct_config()
            self._connected = True

            logger.info(f"Connected to MCP server with config: {self._config.name}")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize MCP backend: {e}")
            await self._close_session()
            return False

    async def _reconstruct_config(self) -> AcceleratorConfig:
        """Reconstruct AcceleratorConfig from MCP tools and resources.

        Returns:
            AcceleratorConfig built from MCP server data
        """
        # Get basic config info
        config_data = await self._read_resource("tao://config")

        # Get knobs
        knobs_result = await self._call_tool("tao_list_knobs")
        knobs = [
            KnobDefinition(
                name=k["name"],
                description=k["description"],
                min_value=k["min_value"],
                max_value=k["max_value"],
                unit=k["unit"],
                rate_limit=k["rate_limit"],
                element_type=k["element_type"],
            )
            for k in knobs_result.get("knobs", [])
        ]

        # Get diagnostics
        diags_result = await self._call_tool("tao_list_diagnostics")
        diagnostics = [
            DiagnosticDefinition(
                name=d["name"],
                description=d["description"],
                measurement_type=d["measurement_type"],
                unit=d["unit"],
                nominal_value=d["nominal_value"],
                tolerance=d["tolerance"],
                alarm_threshold=d["alarm_threshold"],
            )
            for d in diags_result.get("diagnostics", [])
        ]

        # Get constraints
        constraints_data = await self._read_resource("tao://constraints")
        constraints = [
            ConstraintDefinition(
                constraint_id=c["constraint_id"],
                description=c["description"],
                constraint_type=c["constraint_type"],
                parameters=c["parameters"],
                enabled=c.get("enabled", True),
            )
            for c in constraints_data
            if isinstance(constraints_data, list)
        ] if constraints_data else []

        return AcceleratorConfig(
            name=config_data.get("name", "Unknown"),
            description=config_data.get("description", ""),
            knobs=knobs,
            diagnostics=diagnostics,
            constraints=constraints,
            tao_init_file=config_data.get("tao_init_file"),
            metadata=config_data.get("metadata", {}),
        )

    async def _shutdown_async(self) -> bool:
        """Async implementation of shutdown.

        Returns:
            True if shutdown successful
        """
        try:
            if self._session is not None:
                # Disconnect from Tao
                await self._call_tool("tao_disconnect")

            await self._close_session()
            self._connected = False
            self._config = None

            logger.info("MCP backend shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"Error during shutdown: {e}")
            return False

    def initialize(self) -> bool:
        """Initialize the MCP backend and connect to the MCP server.

        Returns:
            True if initialization successful
        """
        return self._run_async(self._initialize_async())

    def shutdown(self) -> bool:
        """Shutdown the MCP backend and close connections.

        Returns:
            True if shutdown successful
        """
        return self._run_async(self._shutdown_async())

    def is_connected(self) -> bool:
        """Check if backend is connected to MCP server.

        Returns:
            True if connected
        """
        return self._connected and self._session is not None

    def run_calculation(self) -> bool:
        """Run Tao calculation via MCP server.

        Returns:
            True if calculation successful
        """

        async def _run():
            result = await self._call_tool("tao_run_calculation")
            return result.get("success", False)

        return self._run_async(_run())

    def read_diagnostic(self, diagnostic_name: str) -> DiagnosticSnapshot:
        """Read a diagnostic measurement via MCP server.

        Args:
            diagnostic_name: Name of the diagnostic to read

        Returns:
            DiagnosticSnapshot with current measurement

        Raises:
            ValueError: If diagnostic not found
            RuntimeError: If reading fails
        """

        async def _read():
            result = await self._call_tool(
                "tao_read_diagnostic", {"diagnostic_name": diagnostic_name}
            )

            if "error" in result:
                raise ValueError(result["error"])

            status_str = result.get("status", "unknown")
            try:
                status = DiagnosticStatus(status_str)
            except ValueError:
                status = DiagnosticStatus.UNKNOWN

            return DiagnosticSnapshot(
                timestamp=result.get("timestamp", datetime.now().isoformat()),
                diagnostic_name=result["diagnostic_name"],
                value=result["value"],
                unit=result["unit"],
                status=status,
                message=result.get("message"),
            )

        return self._run_async(_read())

    def read_all_diagnostics(self) -> List[DiagnosticSnapshot]:
        """Read all configured diagnostics via MCP server.

        Returns:
            List of DiagnosticSnapshots
        """

        async def _read_all():
            result = await self._call_tool("tao_read_all_diagnostics")

            if "error" in result:
                raise RuntimeError(result["error"])

            snapshots = []
            for d in result.get("diagnostics", []):
                status_str = d.get("status", "unknown")
                try:
                    status = DiagnosticStatus(status_str)
                except ValueError:
                    status = DiagnosticStatus.UNKNOWN

                snapshots.append(
                    DiagnosticSnapshot(
                        timestamp=datetime.now().isoformat(),
                        diagnostic_name=d["diagnostic_name"],
                        value=d["value"],
                        unit=d["unit"],
                        status=status,
                        message=d.get("message"),
                    )
                )

            return snapshots

        return self._run_async(_read_all())

    def get_parameter(self, parameter_name: str) -> ParameterValue:
        """Get parameter value via MCP server.

        Args:
            parameter_name: Name of the parameter to read

        Returns:
            ParameterValue with current value

        Raises:
            ValueError: If parameter not found
        """

        async def _get():
            result = await self._call_tool(
                "tao_get_parameter", {"parameter_name": parameter_name}
            )

            if "error" in result:
                raise ValueError(result["error"])

            return ParameterValue(
                timestamp=result.get("timestamp", datetime.now().isoformat()),
                parameter_name=result["parameter_name"],
                value=result["value"],
                unit=result["unit"],
            )

        return self._run_async(_get())

    def get_all_parameters(self) -> Dict[str, ParameterValue]:
        """Get all parameter values via MCP server.

        Returns:
            Dictionary mapping parameter names to ParameterValues
        """

        async def _get_all():
            result = await self._call_tool("tao_get_all_parameters")

            if "error" in result:
                raise RuntimeError(result["error"])

            parameters = {}
            for name, p in result.get("parameters", {}).items():
                parameters[name] = ParameterValue(
                    timestamp=datetime.now().isoformat(),
                    parameter_name=name,
                    value=p["value"],
                    unit=p["unit"],
                )

            return parameters

        return self._run_async(_get_all())

    def set_parameter(self, parameter_name: str, value: float) -> ActionResult:
        """Set parameter value via MCP server.

        Args:
            parameter_name: Name of the parameter to set
            value: New value

        Returns:
            ActionResult indicating success or failure
        """

        async def _set():
            result = await self._call_tool(
                "tao_set_parameter",
                {"parameter_name": parameter_name, "value": value},
            )

            return ActionResult(
                success=result.get("success", False),
                action_type=result.get("action_type", "set_parameter"),
                parameters=result.get("parameters", {"parameter_name": parameter_name, "value": value}),
                message=result.get("message", ""),
                error=result.get("error"),
            )

        return self._run_async(_set())
