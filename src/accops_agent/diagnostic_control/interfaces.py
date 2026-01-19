"""Abstract interfaces for accelerator diagnostic and control systems."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..config.schema import AcceleratorConfig
from .data_models import ActionResult, DiagnosticSnapshot, ParameterValue


class DiagnosticProvider(ABC):
    """Abstract interface for reading diagnostic measurements."""

    @abstractmethod
    def read_diagnostic(self, diagnostic_name: str) -> DiagnosticSnapshot:
        """Read a single diagnostic measurement.

        Args:
            diagnostic_name: Name of the diagnostic to read

        Returns:
            DiagnosticSnapshot with current measurement

        Raises:
            ValueError: If diagnostic_name is not found in configuration
            RuntimeError: If reading fails
        """
        pass

    @abstractmethod
    def read_all_diagnostics(self) -> List[DiagnosticSnapshot]:
        """Read all configured diagnostics.

        Returns:
            List of DiagnosticSnapshots for all diagnostics

        Raises:
            RuntimeError: If reading fails
        """
        pass


class ControlProvider(ABC):
    """Abstract interface for controlling accelerator parameters."""

    @abstractmethod
    def get_parameter(self, parameter_name: str) -> ParameterValue:
        """Get current value of a parameter.

        Args:
            parameter_name: Name of the parameter (knob) to read

        Returns:
            ParameterValue with current value

        Raises:
            ValueError: If parameter_name is not found in configuration
            RuntimeError: If reading fails
        """
        pass

    @abstractmethod
    def set_parameter(self, parameter_name: str, value: float) -> ActionResult:
        """Set a parameter to a new value.

        This method should validate the value against configured limits
        before attempting to set it.

        Args:
            parameter_name: Name of the parameter (knob) to set
            value: New value to set

        Returns:
            ActionResult indicating success or failure

        Raises:
            ValueError: If parameter_name is not found or value is invalid
            RuntimeError: If setting fails
        """
        pass

    @abstractmethod
    def get_all_parameters(self) -> Dict[str, ParameterValue]:
        """Get current values of all configured parameters.

        Returns:
            Dictionary mapping parameter names to ParameterValues

        Raises:
            RuntimeError: If reading fails
        """
        pass


class AcceleratorBackend(DiagnosticProvider, ControlProvider, ABC):
    """Combined interface for complete accelerator control system.

    This is the main interface that concrete backends (e.g., Tao, EPICS)
    should implement. It combines diagnostic reading and parameter control
    with additional lifecycle and calculation methods.
    """

    def __init__(self, config: AcceleratorConfig):
        """Initialize backend with accelerator configuration.

        Args:
            config: AcceleratorConfig with knobs, diagnostics, and constraints
        """
        self.config = config

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend and establish connections.

        Returns:
            True if initialization successful, False otherwise

        Raises:
            RuntimeError: If initialization fails critically
        """
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """Clean shutdown of backend and close connections.

        Returns:
            True if shutdown successful, False otherwise
        """
        pass

    @abstractmethod
    def run_calculation(self) -> bool:
        """Run simulation calculation to propagate parameter changes.

        For simulation backends (e.g., Tao), this triggers calculation.
        For real machine backends, this may be a no-op or trigger
        settling time.

        Returns:
            True if calculation successful, False otherwise

        Raises:
            RuntimeError: If calculation fails
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if backend is connected and ready.

        Returns:
            True if connected and operational, False otherwise
        """
        pass

    def validate_parameter_value(self, parameter_name: str, value: float) -> tuple[bool, Optional[str]]:
        """Validate a parameter value against configuration limits.

        This is a concrete helper method that backends can use.

        Args:
            parameter_name: Name of the parameter
            value: Proposed value

        Returns:
            Tuple of (is_valid, error_message)
        """
        knob = self.config.get_knob(parameter_name)
        if knob is None:
            return False, f"Parameter '{parameter_name}' not found in configuration"

        if value < knob.min_value:
            return False, f"Value {value} below minimum {knob.min_value} {knob.unit}"

        if value > knob.max_value:
            return False, f"Value {value} above maximum {knob.max_value} {knob.unit}"

        return True, None

    def determine_diagnostic_status(
        self, diagnostic_name: str, value: float
    ) -> tuple[str, str]:
        """Determine status of a diagnostic based on thresholds.

        This is a concrete helper method that backends can use.

        Args:
            diagnostic_name: Name of the diagnostic
            value: Measured value

        Returns:
            Tuple of (status_string, message)
        """
        diag = self.config.get_diagnostic(diagnostic_name)
        if diag is None:
            return "unknown", f"Diagnostic '{diagnostic_name}' not configured"

        deviation = abs(value - diag.nominal_value)

        if deviation <= diag.tolerance:
            return "normal", "Within tolerance"
        elif deviation <= diag.alarm_threshold:
            return "warning", f"Deviation {deviation:.3f} {diag.unit} exceeds tolerance"
        else:
            return "alarm", f"Deviation {deviation:.3f} {diag.unit} exceeds alarm threshold"
