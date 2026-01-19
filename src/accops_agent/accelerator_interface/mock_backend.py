"""Mock backend for testing without real accelerator or simulation."""

from datetime import datetime
from typing import Dict, List

from ..config.schema import AcceleratorConfig
from .data_models import ActionResult, DiagnosticSnapshot, DiagnosticStatus, ParameterValue
from .interfaces import AcceleratorBackend


class MockBackend(AcceleratorBackend):
    """Mock implementation of AcceleratorBackend for testing.

    This backend simulates an accelerator using simple in-memory state.
    Useful for development and testing without requiring Tao or real hardware.

    Attributes:
        _diagnostics: Dictionary storing mock diagnostic values
        _parameters: Dictionary storing mock parameter values
        _connected: Connection status flag
    """

    def __init__(self, config: AcceleratorConfig):
        """Initialize mock backend.

        Args:
            config: AcceleratorConfig defining the accelerator
        """
        super().__init__(config)
        self._diagnostics: Dict[str, float] = {}
        self._parameters: Dict[str, float] = {}
        self._connected = False

        # Initialize diagnostics to nominal values
        for diag in config.diagnostics:
            self._diagnostics[diag.name] = diag.nominal_value

        # Initialize parameters to midpoint of range
        for knob in config.knobs:
            midpoint = (knob.min_value + knob.max_value) / 2.0
            self._parameters[knob.name] = midpoint

    def initialize(self) -> bool:
        """Initialize the mock backend."""
        self._connected = True
        return True

    def shutdown(self) -> bool:
        """Shutdown the mock backend."""
        self._connected = False
        return True

    def is_connected(self) -> bool:
        """Check if mock backend is connected."""
        return self._connected

    def run_calculation(self) -> bool:
        """Simulate running a calculation.

        For the mock backend, this applies simple physics-based updates
        to diagnostics based on parameter changes.
        """
        if not self._connected:
            return False

        # Simple simulation: parameters affect diagnostics
        # This is a placeholder - real simulations would use Tao/Bmad
        # For example, quadrupole changes affect beam size
        for knob_name, param_value in self._parameters.items():
            knob = self.config.get_knob(knob_name)
            if knob and "quad" in knob.element_type.lower():
                # Simulate quadrupole affecting beam size
                for diag_name in self._diagnostics:
                    diag = self.config.get_diagnostic(diag_name)
                    if diag and "beam_size" in diag.measurement_type.lower():
                        # Simple model: larger quad strength -> smaller beam
                        nominal_strength = (knob.min_value + knob.max_value) / 2.0
                        strength_factor = 1.0 - 0.1 * (param_value - nominal_strength)
                        self._diagnostics[diag_name] = diag.nominal_value * strength_factor

        return True

    def read_diagnostic(self, diagnostic_name: str) -> DiagnosticSnapshot:
        """Read a diagnostic from mock storage."""
        if diagnostic_name not in self._diagnostics:
            diag_def = self.config.get_diagnostic(diagnostic_name)
            if diag_def is None:
                raise ValueError(f"Diagnostic '{diagnostic_name}' not found in configuration")
            # Initialize if not present
            self._diagnostics[diagnostic_name] = diag_def.nominal_value

        value = self._diagnostics[diagnostic_name]
        diag_def = self.config.get_diagnostic(diagnostic_name)

        # Determine status using helper method
        status_str, message = self.determine_diagnostic_status(diagnostic_name, value)

        return DiagnosticSnapshot(
            timestamp=datetime.now().isoformat(),
            diagnostic_name=diagnostic_name,
            value=value,
            unit=diag_def.unit,
            status=DiagnosticStatus(status_str),
            message=message,
        )

    def read_all_diagnostics(self) -> List[DiagnosticSnapshot]:
        """Read all configured diagnostics."""
        return [self.read_diagnostic(diag.name) for diag in self.config.diagnostics]

    def get_parameter(self, parameter_name: str) -> ParameterValue:
        """Get parameter value from mock storage."""
        if parameter_name not in self._parameters:
            knob = self.config.get_knob(parameter_name)
            if knob is None:
                raise ValueError(f"Parameter '{parameter_name}' not found in configuration")
            # Initialize if not present
            midpoint = (knob.min_value + knob.max_value) / 2.0
            self._parameters[parameter_name] = midpoint

        value = self._parameters[parameter_name]
        knob = self.config.get_knob(parameter_name)

        return ParameterValue(
            timestamp=datetime.now().isoformat(),
            parameter_name=parameter_name,
            value=value,
            unit=knob.unit,
        )

    def set_parameter(self, parameter_name: str, value: float) -> ActionResult:
        """Set parameter value in mock storage."""
        # Validate using helper method
        is_valid, error_msg = self.validate_parameter_value(parameter_name, value)
        if not is_valid:
            return ActionResult(
                success=False,
                action_type="set_parameter",
                parameters={"parameter_name": parameter_name, "value": value},
                message="Validation failed",
                error=error_msg,
            )

        # Set the value
        self._parameters[parameter_name] = value

        return ActionResult(
            success=True,
            action_type="set_parameter",
            parameters={"parameter_name": parameter_name, "value": value},
            message=f"Set {parameter_name} to {value}",
        )

    def get_all_parameters(self) -> Dict[str, ParameterValue]:
        """Get all parameter values."""
        return {knob.name: self.get_parameter(knob.name) for knob in self.config.knobs}

    # Helper methods for testing

    def set_mock_diagnostic(self, diagnostic_name: str, value: float) -> None:
        """Directly set a diagnostic value for testing.

        Args:
            diagnostic_name: Name of the diagnostic
            value: Value to set
        """
        self._diagnostics[diagnostic_name] = value

    def get_mock_parameter_value(self, parameter_name: str) -> float:
        """Directly get parameter value for testing.

        Args:
            parameter_name: Name of the parameter

        Returns:
            Current parameter value
        """
        return self._parameters.get(parameter_name, 0.0)

    def reset(self) -> None:
        """Reset all diagnostics and parameters to defaults."""
        # Reset diagnostics to nominal
        for diag in self.config.diagnostics:
            self._diagnostics[diag.name] = diag.nominal_value

        # Reset parameters to midpoint
        for knob in self.config.knobs:
            midpoint = (knob.min_value + knob.max_value) / 2.0
            self._parameters[knob.name] = midpoint
