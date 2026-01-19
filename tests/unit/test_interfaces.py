"""Tests for diagnostic/control interfaces and mock backend."""

import pytest
from pathlib import Path

from accops_agent.config import load_accelerator_config
from accops_agent.accelerator_interface import (
    DiagnosticSnapshot,
    DiagnosticStatus,
    ParameterValue,
    ActionResult,
)
from accops_agent.accelerator_interface.mock_backend import MockBackend


@pytest.fixture
def test_config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "accelerators" / "example_linac.yaml"
    return load_accelerator_config(config_path)


@pytest.fixture
def mock_backend(test_config):
    """Create and initialize mock backend."""
    backend = MockBackend(test_config)
    backend.initialize()
    return backend


class TestMockBackend:
    """Tests for MockBackend implementation."""

    def test_initialization(self, test_config):
        """Test backend initialization."""
        backend = MockBackend(test_config)
        assert not backend.is_connected()

        success = backend.initialize()
        assert success
        assert backend.is_connected()

    def test_shutdown(self, mock_backend):
        """Test backend shutdown."""
        assert mock_backend.is_connected()

        success = mock_backend.shutdown()
        assert success
        assert not mock_backend.is_connected()

    def test_read_diagnostic(self, mock_backend):
        """Test reading a single diagnostic."""
        snapshot = mock_backend.read_diagnostic("BPM1_X")

        assert isinstance(snapshot, DiagnosticSnapshot)
        assert snapshot.diagnostic_name == "BPM1_X"
        assert snapshot.unit == "mm"
        assert snapshot.value == 0.0  # Nominal value
        assert snapshot.status == DiagnosticStatus.NORMAL

    def test_read_all_diagnostics(self, mock_backend):
        """Test reading all diagnostics."""
        snapshots = mock_backend.read_all_diagnostics()

        assert len(snapshots) == 8  # Example LINAC has 8 diagnostics
        assert all(isinstance(s, DiagnosticSnapshot) for s in snapshots)

        # Check we got all expected diagnostics
        diagnostic_names = {s.diagnostic_name for s in snapshots}
        expected_names = {"BPM1_X", "BPM1_Y", "BPM2_X", "BPM2_Y", "BEAM_SIZE_X", "BEAM_SIZE_Y", "TRANSMISSION", "BEAM_ENERGY"}
        assert diagnostic_names == expected_names

    def test_read_nonexistent_diagnostic(self, mock_backend):
        """Test reading a diagnostic that doesn't exist."""
        with pytest.raises(ValueError, match="not found in configuration"):
            mock_backend.read_diagnostic("NONEXISTENT")

    def test_get_parameter(self, mock_backend):
        """Test getting a parameter value."""
        param_value = mock_backend.get_parameter("QF1_K1")

        assert isinstance(param_value, ParameterValue)
        assert param_value.parameter_name == "QF1_K1"
        assert param_value.unit == "T/m"
        assert param_value.value == 0.0  # Midpoint of [-5, 5]

    def test_get_all_parameters(self, mock_backend):
        """Test getting all parameters."""
        params = mock_backend.get_all_parameters()

        assert len(params) == 5  # Example LINAC has 5 knobs
        assert all(isinstance(p, ParameterValue) for p in params.values())

        # Check we got all expected parameters
        expected_names = {"QF1_K1", "QD1_K1", "HCOR1_KICK", "VCOR1_KICK", "CAV1_VOLTAGE"}
        assert set(params.keys()) == expected_names

    def test_set_parameter_valid(self, mock_backend):
        """Test setting a parameter to a valid value."""
        result = mock_backend.set_parameter("QF1_K1", 2.5)

        assert isinstance(result, ActionResult)
        assert result.success
        assert result.action_type == "set_parameter"
        assert result.error is None

        # Verify value was set
        param_value = mock_backend.get_parameter("QF1_K1")
        assert param_value.value == 2.5

    def test_set_parameter_below_min(self, mock_backend):
        """Test setting parameter below minimum limit."""
        result = mock_backend.set_parameter("QF1_K1", -10.0)  # Min is -5.0

        assert not result.success
        assert result.error is not None
        assert "below minimum" in result.error

        # Verify value was NOT changed
        param_value = mock_backend.get_parameter("QF1_K1")
        assert param_value.value == 0.0  # Still at initial midpoint

    def test_set_parameter_above_max(self, mock_backend):
        """Test setting parameter above maximum limit."""
        result = mock_backend.set_parameter("QF1_K1", 10.0)  # Max is 5.0

        assert not result.success
        assert result.error is not None
        assert "above maximum" in result.error

    def test_set_parameter_nonexistent(self, mock_backend):
        """Test setting a parameter that doesn't exist."""
        result = mock_backend.set_parameter("NONEXISTENT", 1.0)

        assert not result.success
        assert "not found in configuration" in result.error

    def test_run_calculation(self, mock_backend):
        """Test running calculation."""
        success = mock_backend.run_calculation()
        assert success

    def test_diagnostic_status_normal(self, mock_backend):
        """Test diagnostic status when within tolerance."""
        mock_backend.set_mock_diagnostic("BPM1_X", 0.3)  # Tolerance is 0.5
        snapshot = mock_backend.read_diagnostic("BPM1_X")

        assert snapshot.status == DiagnosticStatus.NORMAL
        assert snapshot.is_healthy()
        assert not snapshot.is_alarm()

    def test_diagnostic_status_warning(self, mock_backend):
        """Test diagnostic status when exceeds tolerance but below alarm."""
        mock_backend.set_mock_diagnostic("BPM1_X", 1.0)  # Tolerance: 0.5, Alarm: 2.0
        snapshot = mock_backend.read_diagnostic("BPM1_X")

        assert snapshot.status == DiagnosticStatus.WARNING
        assert snapshot.is_healthy()  # Warning is still considered healthy
        assert not snapshot.is_alarm()

    def test_diagnostic_status_alarm(self, mock_backend):
        """Test diagnostic status when exceeds alarm threshold."""
        mock_backend.set_mock_diagnostic("BPM1_X", 3.0)  # Alarm threshold: 2.0
        snapshot = mock_backend.read_diagnostic("BPM1_X")

        assert snapshot.status == DiagnosticStatus.ALARM
        assert not snapshot.is_healthy()
        assert snapshot.is_alarm()

    def test_validate_parameter_value(self, mock_backend):
        """Test parameter value validation helper."""
        # Valid value
        is_valid, error = mock_backend.validate_parameter_value("QF1_K1", 2.0)
        assert is_valid
        assert error is None

        # Below minimum
        is_valid, error = mock_backend.validate_parameter_value("QF1_K1", -10.0)
        assert not is_valid
        assert "below minimum" in error

        # Above maximum
        is_valid, error = mock_backend.validate_parameter_value("QF1_K1", 10.0)
        assert not is_valid
        assert "above maximum" in error

        # Nonexistent parameter
        is_valid, error = mock_backend.validate_parameter_value("NONEXISTENT", 1.0)
        assert not is_valid
        assert "not found" in error

    def test_reset(self, mock_backend):
        """Test resetting backend to defaults."""
        # Change some values
        mock_backend.set_parameter("QF1_K1", 3.0)
        mock_backend.set_mock_diagnostic("BPM1_X", 5.0)

        # Verify they changed
        assert mock_backend.get_parameter("QF1_K1").value == 3.0
        assert mock_backend.read_diagnostic("BPM1_X").value == 5.0

        # Reset
        mock_backend.reset()

        # Verify back to defaults
        assert mock_backend.get_parameter("QF1_K1").value == 0.0  # Midpoint
        assert mock_backend.read_diagnostic("BPM1_X").value == 0.0  # Nominal


class TestDataModels:
    """Tests for data models."""

    def test_diagnostic_snapshot_creation(self):
        """Test creating a DiagnosticSnapshot."""
        snapshot = DiagnosticSnapshot(
            diagnostic_name="TEST_DIAG",
            value=1.23,
            unit="mm",
            status=DiagnosticStatus.NORMAL,
        )

        assert snapshot.diagnostic_name == "TEST_DIAG"
        assert snapshot.value == 1.23
        assert snapshot.unit == "mm"
        assert snapshot.status == DiagnosticStatus.NORMAL
        assert snapshot.is_healthy()

    def test_parameter_value_creation(self):
        """Test creating a ParameterValue."""
        param = ParameterValue(
            parameter_name="TEST_PARAM",
            value=2.5,
            unit="T/m",
        )

        assert param.parameter_name == "TEST_PARAM"
        assert param.value == 2.5
        assert param.unit == "T/m"

    def test_parameter_is_at_setpoint(self):
        """Test checking if parameter is at setpoint."""
        param = ParameterValue(
            parameter_name="TEST",
            value=1.0,
            unit="T/m",
            setpoint=1.0,
        )
        assert param.is_at_setpoint()

        param.setpoint = 1.1
        assert not param.is_at_setpoint(tolerance=0.05)
        assert param.is_at_setpoint(tolerance=0.2)
