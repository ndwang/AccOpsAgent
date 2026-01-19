"""Tao backend implementation for AcceleratorBackend interface."""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from ...config.schema import AcceleratorConfig
from ...diagnostic_control import (
    AcceleratorBackend,
    ActionResult,
    DiagnosticSnapshot,
    DiagnosticStatus,
    ParameterValue,
)
from .commands import (
    build_get_parameter_command,
    build_read_beam_size_command,
    build_read_data_command,
    build_run_calculation_command,
    build_set_parameter_command,
)
from .connection import TaoConnection
from .parser import TaoDataParser

logger = logging.getLogger(__name__)


class TaoBackend(AcceleratorBackend):
    """Concrete implementation of AcceleratorBackend using PyTao.

    This backend connects to Bmad/Tao simulations to read diagnostics
    and control parameters.

    Attributes:
        connection: TaoConnection instance
        parser: TaoDataParser for parsing Tao outputs
    """

    def __init__(self, config: AcceleratorConfig):
        """Initialize Tao backend.

        Args:
            config: AcceleratorConfig defining the accelerator
        """
        super().__init__(config)
        self.connection = TaoConnection(config.tao_init_file)
        self.parser = TaoDataParser()
        logger.info(f"TaoBackend initialized for {config.name}")

    def initialize(self) -> bool:
        """Initialize connection to Tao."""
        success = self.connection.connect()
        if success:
            logger.info("TaoBackend connected successfully")
        else:
            logger.error("TaoBackend connection failed")
        return success

    def shutdown(self) -> bool:
        """Shutdown Tao connection."""
        success = self.connection.disconnect()
        if success:
            logger.info("TaoBackend disconnected")
        return success

    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self.connection.is_connected()

    def run_calculation(self) -> bool:
        """Run Tao calculation to propagate parameter changes."""
        if not self.is_connected():
            logger.error("Cannot run calculation: not connected")
            return False

        try:
            command = build_run_calculation_command()
            self.connection.execute_command(command)
            logger.debug("Tao calculation completed")
            return True

        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            return False

    def read_diagnostic(self, diagnostic_name: str) -> DiagnosticSnapshot:
        """Read a diagnostic measurement from Tao.

        Args:
            diagnostic_name: Name of the diagnostic

        Returns:
            DiagnosticSnapshot with measurement data

        Raises:
            ValueError: If diagnostic not found in configuration
            RuntimeError: If reading fails
        """
        diag_def = self.config.get_diagnostic(diagnostic_name)
        if diag_def is None:
            raise ValueError(
                f"Diagnostic '{diagnostic_name}' not found in configuration"
            )

        if not self.is_connected():
            raise RuntimeError("Cannot read diagnostic: not connected to Tao")

        try:
            # Map diagnostic to Tao element and attribute
            # This is a simplified mapping - real implementation needs config
            value = self._read_tao_diagnostic(diagnostic_name, diag_def)

            # Determine status
            status_str, message = self.determine_diagnostic_status(
                diagnostic_name, value
            )

            return DiagnosticSnapshot(
                timestamp=datetime.now().isoformat(),
                diagnostic_name=diagnostic_name,
                value=value,
                unit=diag_def.unit,
                status=DiagnosticStatus(status_str),
                message=message,
            )

        except Exception as e:
            logger.error(f"Failed to read diagnostic {diagnostic_name}: {e}")
            raise RuntimeError(f"Failed to read diagnostic: {e}") from e

    def _read_tao_diagnostic(self, diagnostic_name: str, diag_def) -> float:
        """Read diagnostic value from Tao.

        Args:
            diagnostic_name: Name of the diagnostic
            diag_def: Diagnostic definition from config

        Returns:
            Measured value

        Raises:
            RuntimeError: If reading fails
        """
        # Map measurement type to Tao attributes
        measurement_type = diag_def.measurement_type.lower()

        if "orbit" in measurement_type or "bpm" in diagnostic_name.lower():
            # Read orbit position
            axis = "x" if "_x" in diagnostic_name.lower() else "y"
            attribute = f"orbit_{axis}"
            element = diagnostic_name.split("_")[0]  # e.g., BPM1_X -> BPM1

        elif "beam_size" in measurement_type:
            # Read beam size
            axis = "x" if "_x" in diagnostic_name.lower() else "y"
            attribute = f"sig_{axis}"
            element = "end"  # Or specific element

        elif "transmission" in measurement_type:
            # Read beam current ratio
            attribute = "charge"
            element = "end"

        elif "energy" in measurement_type:
            # Read beam energy
            attribute = "e_tot"
            element = "end"

        else:
            # Generic read
            attribute = "value"
            element = diagnostic_name

        # Execute Tao command
        command = build_read_data_command(element, attribute)
        output = self.connection.execute_command(command)

        # Parse result
        value = self.parser.parse_single_value(output, attribute)
        if value is None:
            logger.warning(
                f"Failed to parse value from Tao, returning nominal: {diag_def.nominal_value}"
            )
            value = diag_def.nominal_value

        return value

    def read_all_diagnostics(self) -> List[DiagnosticSnapshot]:
        """Read all configured diagnostics from Tao."""
        return [self.read_diagnostic(diag.name) for diag in self.config.diagnostics]

    def get_parameter(self, parameter_name: str) -> ParameterValue:
        """Get parameter value from Tao.

        Args:
            parameter_name: Name of the parameter

        Returns:
            ParameterValue with current value

        Raises:
            ValueError: If parameter not found in configuration
            RuntimeError: If reading fails
        """
        knob = self.config.get_knob(parameter_name)
        if knob is None:
            raise ValueError(
                f"Parameter '{parameter_name}' not found in configuration"
            )

        if not self.is_connected():
            raise RuntimeError("Cannot read parameter: not connected to Tao")

        try:
            # Map parameter to Tao element and attribute
            value = self._read_tao_parameter(parameter_name, knob)

            return ParameterValue(
                timestamp=datetime.now().isoformat(),
                parameter_name=parameter_name,
                value=value,
                unit=knob.unit,
            )

        except Exception as e:
            logger.error(f"Failed to read parameter {parameter_name}: {e}")
            raise RuntimeError(f"Failed to read parameter: {e}") from e

    def _read_tao_parameter(self, parameter_name: str, knob) -> float:
        """Read parameter value from Tao.

        Args:
            parameter_name: Name of the parameter
            knob: Knob definition from config

        Returns:
            Current parameter value

        Raises:
            RuntimeError: If reading fails
        """
        # Map element type to Tao attributes
        element_type = knob.element_type.lower()

        if "quad" in element_type:
            attribute = "k1"
        elif "dipole" in element_type or "bend" in element_type:
            attribute = "g"
        elif "corrector" in element_type or "cor" in element_type:
            attribute = "kick"
        elif "cavity" in element_type or "rf" in element_type:
            attribute = "voltage"
        else:
            # Generic attribute
            attribute = "value"

        # Extract element name from parameter name
        # e.g., QF1_K1 -> QF1
        element = parameter_name.split("_")[0]

        # Execute Tao command
        command = build_get_parameter_command(element, attribute)
        output = self.connection.execute_command(command)

        # Parse result
        value = self.parser.parse_single_value(output, attribute)
        if value is None:
            logger.warning(
                f"Failed to parse value from Tao, returning midpoint: {(knob.min_value + knob.max_value) / 2.0}"
            )
            value = (knob.min_value + knob.max_value) / 2.0

        return value

    def set_parameter(self, parameter_name: str, value: float) -> ActionResult:
        """Set parameter value in Tao.

        Args:
            parameter_name: Name of the parameter
            value: New value to set

        Returns:
            ActionResult indicating success or failure
        """
        # Validate
        is_valid, error_msg = self.validate_parameter_value(parameter_name, value)
        if not is_valid:
            return ActionResult(
                success=False,
                action_type="set_parameter",
                parameters={"parameter_name": parameter_name, "value": value},
                message="Validation failed",
                error=error_msg,
            )

        if not self.is_connected():
            return ActionResult(
                success=False,
                action_type="set_parameter",
                parameters={"parameter_name": parameter_name, "value": value},
                message="Not connected",
                error="Not connected to Tao",
            )

        try:
            knob = self.config.get_knob(parameter_name)
            element_type = knob.element_type.lower()

            # Map to Tao attribute
            if "quad" in element_type:
                attribute = "k1"
            elif "dipole" in element_type or "bend" in element_type:
                attribute = "g"
            elif "corrector" in element_type or "cor" in element_type:
                attribute = "kick"
            elif "cavity" in element_type or "rf" in element_type:
                attribute = "voltage"
            else:
                attribute = "value"

            # Extract element name
            element = parameter_name.split("_")[0]

            # Execute set command
            command = build_set_parameter_command(element, attribute, value)
            self.connection.execute_command(command)

            logger.info(f"Set {parameter_name} to {value} {knob.unit}")

            return ActionResult(
                success=True,
                action_type="set_parameter",
                parameters={"parameter_name": parameter_name, "value": value},
                message=f"Set {parameter_name} to {value} {knob.unit}",
            )

        except Exception as e:
            logger.error(f"Failed to set parameter {parameter_name}: {e}")
            return ActionResult(
                success=False,
                action_type="set_parameter",
                parameters={"parameter_name": parameter_name, "value": value},
                message="Execution failed",
                error=str(e),
            )

    def get_all_parameters(self) -> Dict[str, ParameterValue]:
        """Get all parameter values from Tao."""
        return {knob.name: self.get_parameter(knob.name) for knob in self.config.knobs}
