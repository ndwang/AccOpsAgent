"""Tests for MCPBackend."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from accops_agent.accelerator_interface.mcp_backend import MCPBackend
from accops_agent.accelerator_interface import DiagnosticStatus


class MockTextContent:
    """Mock MCP TextContent."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockCallToolResult:
    """Mock MCP CallToolResult."""

    def __init__(self, content: list):
        self.content = content


class MockReadResourceResult:
    """Mock MCP ReadResourceResult."""

    def __init__(self, contents: list):
        self.contents = contents


@pytest.fixture
def mock_session():
    """Create a mock MCP session."""
    session = AsyncMock()

    # Default tool responses
    session.call_tool = AsyncMock()
    session.read_resource = AsyncMock()
    session.initialize = AsyncMock()

    return session


@pytest.fixture
def config_path(tmp_path):
    """Create a temporary config file path."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("name: test\ndescription: test config\n")
    return str(config_file)


class TestMCPBackendInit:
    """Tests for MCPBackend initialization."""

    def test_init_stores_config_path(self, config_path):
        """Test that __init__ stores the config path."""
        backend = MCPBackend(config_path)
        assert config_path in backend.config_path

    def test_init_not_connected(self, config_path):
        """Test that backend is not connected after init."""
        backend = MCPBackend(config_path)
        assert not backend.is_connected()


class TestMCPBackendInitialize:
    """Tests for MCPBackend.initialize()."""

    @patch("accops_agent.accelerator_interface.mcp_backend.stdio_client")
    def test_initialize_success(self, mock_stdio_client, mock_session, config_path):
        """Test successful initialization."""
        # Setup mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_cm

        # Mock ClientSession
        with patch(
            "accops_agent.accelerator_interface.mcp_backend.ClientSession"
        ) as mock_client_session:
            mock_client_session.return_value = mock_session
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            # Mock tool responses
            mock_session.call_tool.side_effect = [
                # tao_connect
                MockCallToolResult([
                    MockTextContent(json.dumps({
                        "success": True,
                        "config_name": "Test Accelerator",
                    }))
                ]),
                # tao_list_knobs
                MockCallToolResult([
                    MockTextContent(json.dumps({
                        "knobs": [{
                            "name": "QF1_K1",
                            "description": "Test knob",
                            "min_value": 0.0,
                            "max_value": 10.0,
                            "unit": "T/m",
                            "rate_limit": 1.0,
                            "element_type": "quadrupole",
                        }],
                        "count": 1,
                    }))
                ]),
                # tao_list_diagnostics
                MockCallToolResult([
                    MockTextContent(json.dumps({
                        "diagnostics": [{
                            "name": "BPM1_X",
                            "description": "Test diagnostic",
                            "measurement_type": "orbit",
                            "unit": "mm",
                            "nominal_value": 0.0,
                            "tolerance": 0.1,
                            "alarm_threshold": 1.0,
                        }],
                        "count": 1,
                    }))
                ]),
            ]

            # Mock resource responses
            mock_session.read_resource.side_effect = [
                # tao://config
                MockReadResourceResult([
                    MockTextContent(json.dumps({
                        "name": "Test Accelerator",
                        "description": "A test accelerator",
                        "tao_init_file": None,
                        "metadata": {},
                    }))
                ]),
                # tao://constraints
                MockReadResourceResult([
                    MockTextContent(json.dumps([]))
                ]),
            ]

            backend = MCPBackend(config_path)
            result = backend.initialize()

            assert result is True
            assert backend.is_connected()
            assert backend.config.name == "Test Accelerator"
            assert len(backend.config.knobs) == 1
            assert len(backend.config.diagnostics) == 1

    @patch("accops_agent.accelerator_interface.mcp_backend.stdio_client")
    def test_initialize_connection_failure(self, mock_stdio_client, mock_session, config_path):
        """Test initialization failure when connection fails."""
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_cm

        with patch(
            "accops_agent.accelerator_interface.mcp_backend.ClientSession"
        ) as mock_client_session:
            mock_client_session.return_value = mock_session
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            # Mock connection failure
            mock_session.call_tool.return_value = MockCallToolResult([
                MockTextContent(json.dumps({
                    "success": False,
                    "error": "Connection failed",
                }))
            ])

            backend = MCPBackend(config_path)
            result = backend.initialize()

            assert result is False
            assert not backend.is_connected()


class TestMCPBackendReadDiagnostic:
    """Tests for MCPBackend.read_diagnostic()."""

    @patch("accops_agent.accelerator_interface.mcp_backend.stdio_client")
    def test_read_diagnostic_success(self, mock_stdio_client, mock_session, config_path):
        """Test successful diagnostic read."""
        # Setup mock for initialization
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_cm

        with patch(
            "accops_agent.accelerator_interface.mcp_backend.ClientSession"
        ) as mock_client_session:
            mock_client_session.return_value = mock_session
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            # Setup initialization responses
            mock_session.call_tool.side_effect = [
                # tao_connect
                MockCallToolResult([
                    MockTextContent(json.dumps({"success": True}))
                ]),
                # tao_list_knobs
                MockCallToolResult([
                    MockTextContent(json.dumps({
                        "knobs": [{
                            "name": "QF1_K1",
                            "description": "Test",
                            "min_value": 0.0,
                            "max_value": 10.0,
                            "unit": "T/m",
                            "rate_limit": 1.0,
                            "element_type": "quadrupole",
                        }],
                    }))
                ]),
                # tao_list_diagnostics
                MockCallToolResult([
                    MockTextContent(json.dumps({
                        "diagnostics": [{
                            "name": "BPM1_X",
                            "description": "Test",
                            "measurement_type": "orbit",
                            "unit": "mm",
                            "nominal_value": 0.0,
                            "tolerance": 0.1,
                            "alarm_threshold": 1.0,
                        }],
                    }))
                ]),
                # tao_read_diagnostic
                MockCallToolResult([
                    MockTextContent(json.dumps({
                        "diagnostic_name": "BPM1_X",
                        "value": 0.5,
                        "unit": "mm",
                        "status": "normal",
                        "message": "Within tolerance",
                        "timestamp": "2024-01-01T00:00:00",
                    }))
                ]),
            ]

            mock_session.read_resource.side_effect = [
                MockReadResourceResult([
                    MockTextContent(json.dumps({
                        "name": "Test",
                        "description": "Test",
                    }))
                ]),
                MockReadResourceResult([
                    MockTextContent(json.dumps([]))
                ]),
            ]

            backend = MCPBackend(config_path)
            backend.initialize()

            snapshot = backend.read_diagnostic("BPM1_X")

            assert snapshot.diagnostic_name == "BPM1_X"
            assert snapshot.value == 0.5
            assert snapshot.unit == "mm"
            assert snapshot.status == DiagnosticStatus.NORMAL


class TestMCPBackendSetParameter:
    """Tests for MCPBackend.set_parameter()."""

    @patch("accops_agent.accelerator_interface.mcp_backend.stdio_client")
    def test_set_parameter_success(self, mock_stdio_client, mock_session, config_path):
        """Test successful parameter set."""
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_cm

        with patch(
            "accops_agent.accelerator_interface.mcp_backend.ClientSession"
        ) as mock_client_session:
            mock_client_session.return_value = mock_session
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session.call_tool.side_effect = [
                # tao_connect
                MockCallToolResult([
                    MockTextContent(json.dumps({"success": True}))
                ]),
                # tao_list_knobs
                MockCallToolResult([
                    MockTextContent(json.dumps({
                        "knobs": [{
                            "name": "QF1_K1",
                            "description": "Test",
                            "min_value": 0.0,
                            "max_value": 10.0,
                            "unit": "T/m",
                            "rate_limit": 1.0,
                            "element_type": "quadrupole",
                        }],
                    }))
                ]),
                # tao_list_diagnostics
                MockCallToolResult([
                    MockTextContent(json.dumps({
                        "diagnostics": [{
                            "name": "BPM1_X",
                            "description": "Test",
                            "measurement_type": "orbit",
                            "unit": "mm",
                            "nominal_value": 0.0,
                            "tolerance": 0.1,
                            "alarm_threshold": 1.0,
                        }],
                    }))
                ]),
                # tao_set_parameter
                MockCallToolResult([
                    MockTextContent(json.dumps({
                        "success": True,
                        "action_type": "set_parameter",
                        "message": "Set QF1_K1 to 5.0",
                        "parameters": {"parameter_name": "QF1_K1", "value": 5.0},
                    }))
                ]),
            ]

            mock_session.read_resource.side_effect = [
                MockReadResourceResult([
                    MockTextContent(json.dumps({
                        "name": "Test",
                        "description": "Test",
                    }))
                ]),
                MockReadResourceResult([
                    MockTextContent(json.dumps([]))
                ]),
            ]

            backend = MCPBackend(config_path)
            backend.initialize()

            result = backend.set_parameter("QF1_K1", 5.0)

            assert result.success is True
            assert result.action_type == "set_parameter"


class TestMCPBackendShutdown:
    """Tests for MCPBackend.shutdown()."""

    @patch("accops_agent.accelerator_interface.mcp_backend.stdio_client")
    def test_shutdown_success(self, mock_stdio_client, mock_session, config_path):
        """Test successful shutdown."""
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_cm

        with patch(
            "accops_agent.accelerator_interface.mcp_backend.ClientSession"
        ) as mock_client_session:
            mock_client_session.return_value = mock_session
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session.call_tool.side_effect = [
                # tao_connect
                MockCallToolResult([
                    MockTextContent(json.dumps({"success": True}))
                ]),
                # tao_list_knobs
                MockCallToolResult([
                    MockTextContent(json.dumps({
                        "knobs": [{
                            "name": "QF1_K1",
                            "description": "Test",
                            "min_value": 0.0,
                            "max_value": 10.0,
                            "unit": "T/m",
                            "rate_limit": 1.0,
                            "element_type": "quadrupole",
                        }],
                    }))
                ]),
                # tao_list_diagnostics
                MockCallToolResult([
                    MockTextContent(json.dumps({
                        "diagnostics": [{
                            "name": "BPM1_X",
                            "description": "Test",
                            "measurement_type": "orbit",
                            "unit": "mm",
                            "nominal_value": 0.0,
                            "tolerance": 0.1,
                            "alarm_threshold": 1.0,
                        }],
                    }))
                ]),
                # tao_disconnect
                MockCallToolResult([
                    MockTextContent(json.dumps({"success": True}))
                ]),
            ]

            mock_session.read_resource.side_effect = [
                MockReadResourceResult([
                    MockTextContent(json.dumps({
                        "name": "Test",
                        "description": "Test",
                    }))
                ]),
                MockReadResourceResult([
                    MockTextContent(json.dumps([]))
                ]),
            ]

            backend = MCPBackend(config_path)
            backend.initialize()
            assert backend.is_connected()

            result = backend.shutdown()

            assert result is True
            assert not backend.is_connected()
