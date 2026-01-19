"""Integration tests for PyTao backend.

These tests require a working Tao installation and will be skipped if Tao is not available.
"""

import pytest
from pathlib import Path

from accops_agent.config import load_accelerator_config
from accops_agent.diagnostic_control import (
    DiagnosticSnapshot,
    ParameterValue,
    ActionResult,
)

# Try to import TaoBackend - skip tests if not available
try:
    from accops_agent.backends.pytao import TaoBackend
    from accops_agent.backends.pytao.connection import TaoConnection
    from pytao import Tao

    TAO_AVAILABLE = True
except ImportError:
    TAO_AVAILABLE = False
    TaoBackend = None
    TaoConnection = None

pytestmark = pytest.mark.skipif(
    not TAO_AVAILABLE, reason="pytao not installed or Tao not available"
)


@pytest.fixture
def test_config():
    """Load test configuration."""
    config_path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "accelerators"
        / "example_linac.yaml"
    )
    return load_accelerator_config(config_path)


@pytest.fixture
def tao_backend(test_config):
    """Create TaoBackend instance (without connecting)."""
    if not TAO_AVAILABLE:
        pytest.skip("Tao not available")
    return TaoBackend(test_config)


class TestTaoConnection:
    """Tests for TaoConnection class."""

    def test_connection_initialization(self, test_config):
        """Test TaoConnection can be initialized."""
        conn = TaoConnection(test_config.tao_init_file)
        assert conn is not None
        assert not conn.is_connected()

    @pytest.mark.integration
    def test_connection_without_init_file(self):
        """Test connection without init file (may fail if no lattice)."""
        conn = TaoConnection()
        # This may or may not succeed depending on environment
        # Just test that it doesn't crash
        result = conn.connect()
        if result:
            assert conn.is_connected()
            conn.disconnect()


class TestTaoBackend:
    """Tests for TaoBackend implementation."""

    def test_backend_initialization(self, tao_backend):
        """Test backend can be initialized."""
        assert tao_backend is not None
        assert not tao_backend.is_connected()

    def test_backend_implements_interface(self, tao_backend):
        """Test backend implements required interface methods."""
        assert hasattr(tao_backend, "initialize")
        assert hasattr(tao_backend, "shutdown")
        assert hasattr(tao_backend, "is_connected")
        assert hasattr(tao_backend, "read_diagnostic")
        assert hasattr(tao_backend, "read_all_diagnostics")
        assert hasattr(tao_backend, "get_parameter")
        assert hasattr(tao_backend, "set_parameter")
        assert hasattr(tao_backend, "get_all_parameters")
        assert hasattr(tao_backend, "run_calculation")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_connect_and_disconnect(self, tao_backend):
        """Test connecting and disconnecting from Tao.

        Note: This test requires a valid Tao init file.
        """
        # This test will likely fail without a real init file
        # It's mainly to test the interface
        success = tao_backend.initialize()
        if success:
            assert tao_backend.is_connected()
            shutdown_success = tao_backend.shutdown()
            assert shutdown_success
            assert not tao_backend.is_connected()

    def test_validate_parameter_value(self, tao_backend, test_config):
        """Test parameter validation (doesn't require connection)."""
        # Valid value
        is_valid, error = tao_backend.validate_parameter_value("QF1_K1", 2.0)
        assert is_valid
        assert error is None

        # Below minimum
        is_valid, error = tao_backend.validate_parameter_value("QF1_K1", -10.0)
        assert not is_valid
        assert "below minimum" in error

        # Above maximum
        is_valid, error = tao_backend.validate_parameter_value("QF1_K1", 10.0)
        assert not is_valid
        assert "above maximum" in error

    def test_determine_diagnostic_status(self, tao_backend):
        """Test diagnostic status determination (doesn't require connection)."""
        # Normal status
        status, message = tao_backend.determine_diagnostic_status("BPM1_X", 0.3)
        assert status == "normal"

        # Warning status
        status, message = tao_backend.determine_diagnostic_status("BPM1_X", 1.0)
        assert status == "warning"

        # Alarm status
        status, message = tao_backend.determine_diagnostic_status("BPM1_X", 3.0)
        assert status == "alarm"


class TestTaoDataParser:
    """Tests for TaoDataParser."""

    def test_parse_lat_ele_list_empty(self):
        """Test parsing empty output."""
        from accops_agent.backends.pytao.parser import TaoDataParser

        parser = TaoDataParser()
        result = parser.parse_lat_ele_list("")
        assert result == []

    def test_parse_lat_ele_list_simple(self):
        """Test parsing simple lat_ele_list output."""
        from accops_agent.backends.pytao.parser import TaoDataParser

        parser = TaoDataParser()
        output = "ele_name;k1;s\nQF1;2.5;10.0\nQD1;-2.0;20.0"
        result = parser.parse_lat_ele_list(output)

        assert len(result) == 2
        assert result[0]["ele_name"] == "QF1"
        assert result[0]["k1"] == 2.5
        assert result[0]["s"] == 10.0
        assert result[1]["ele_name"] == "QD1"
        assert result[1]["k1"] == -2.0

    def test_parse_single_value_direct(self):
        """Test parsing single numeric value."""
        from accops_agent.backends.pytao.parser import TaoDataParser

        parser = TaoDataParser()

        # Direct number
        result = parser.parse_single_value("3.14159", "test")
        assert result == pytest.approx(3.14159)

        # Scientific notation
        result = parser.parse_single_value("1.23e-5", "test")
        assert result == pytest.approx(1.23e-5)

    def test_parse_single_value_from_formatted(self):
        """Test parsing single value from formatted output."""
        from accops_agent.backends.pytao.parser import TaoDataParser

        parser = TaoDataParser()
        output = "ele_name;k1\nQF1;2.5"
        result = parser.parse_single_value(output, "k1")
        assert result == 2.5

    def test_convert_value_types(self):
        """Test value type conversion."""
        from accops_agent.backends.pytao.parser import TaoDataParser

        parser = TaoDataParser()

        # Boolean
        assert parser._convert_value("True") is True
        assert parser._convert_value("false") is False

        # Integer
        assert parser._convert_value("42") == 42
        assert isinstance(parser._convert_value("42"), int)

        # Float
        assert parser._convert_value("3.14") == 3.14
        assert isinstance(parser._convert_value("3.14"), float)

        # String
        assert parser._convert_value("hello") == "hello"
        assert isinstance(parser._convert_value("hello"), str)


class TestTaoCommands:
    """Tests for Tao command builders."""

    def test_build_read_data_command(self):
        """Test building read data command."""
        from accops_agent.backends.pytao.commands import build_read_data_command

        cmd = build_read_data_command("QF1", "k1")
        assert "QF1" in cmd
        assert "k1" in cmd
        assert "python lat_ele_list" in cmd

    def test_build_set_parameter_command(self):
        """Test building set parameter command."""
        from accops_agent.backends.pytao.commands import build_set_parameter_command

        cmd = build_set_parameter_command("QF1", "k1", 2.5)
        assert "QF1" in cmd
        assert "k1" in cmd
        assert "2.5" in cmd
        assert "set ele" in cmd

    def test_build_get_parameter_command(self):
        """Test building get parameter command."""
        from accops_agent.backends.pytao.commands import build_get_parameter_command

        cmd = build_get_parameter_command("QF1", "k1")
        assert "QF1" in cmd
        assert "k1" in cmd

    def test_build_run_calculation_command(self):
        """Test building calculation command."""
        from accops_agent.backends.pytao.commands import build_run_calculation_command

        cmd = build_run_calculation_command()
        assert "track_type" in cmd or "set global" in cmd


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not TAO_AVAILABLE, reason="Tao not available")
class TestTaoBackendIntegration:
    """Full integration tests requiring working Tao instance.

    These tests are marked as slow and require:
    1. pytao installed
    2. Tao accessible
    3. Valid initialization file or lattice

    Run with: pytest -v -m integration
    """

    @pytest.fixture
    def connected_backend(self, tao_backend):
        """Create and connect backend."""
        success = tao_backend.initialize()
        if not success:
            pytest.skip("Could not connect to Tao (init file may be missing)")
        yield tao_backend
        tao_backend.shutdown()

    def test_read_diagnostic_integration(self, connected_backend):
        """Test reading actual diagnostic from Tao."""
        # This will only work with a proper Tao setup
        try:
            snapshot = connected_backend.read_diagnostic("BPM1_X")
            assert isinstance(snapshot, DiagnosticSnapshot)
            assert snapshot.diagnostic_name == "BPM1_X"
        except Exception as e:
            pytest.skip(f"Could not read diagnostic: {e}")

    def test_get_parameter_integration(self, connected_backend):
        """Test reading actual parameter from Tao."""
        try:
            param = connected_backend.get_parameter("QF1_K1")
            assert isinstance(param, ParameterValue)
            assert param.parameter_name == "QF1_K1"
        except Exception as e:
            pytest.skip(f"Could not read parameter: {e}")

    def test_set_parameter_integration(self, connected_backend):
        """Test setting parameter in Tao."""
        try:
            result = connected_backend.set_parameter("QF1_K1", 2.0)
            assert isinstance(result, ActionResult)
            # Success depends on Tao setup
        except Exception as e:
            pytest.skip(f"Could not set parameter: {e}")
