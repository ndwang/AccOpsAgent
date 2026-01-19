"""Pydantic models for accelerator configuration."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class KnobDefinition(BaseModel):
    """Defines a controllable parameter (knob) in the accelerator.

    Attributes:
        name: Unique identifier for the knob
        description: Human-readable description
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        unit: Physical unit (e.g., "T/m", "rad", "MeV")
        rate_limit: Maximum change allowed per step
        element_type: Type of element (e.g., "quadrupole", "corrector", "cavity")
        tao_variable: Tao variable name to control this parameter (optional)
    """

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    min_value: float
    max_value: float
    unit: str
    rate_limit: float = Field(..., gt=0)
    element_type: str
    tao_variable: Optional[str] = None

    @field_validator("max_value")
    @classmethod
    def validate_max_greater_than_min(cls, v: float, info) -> float:
        """Ensure max_value is greater than min_value."""
        if "min_value" in info.data and v <= info.data["min_value"]:
            raise ValueError("max_value must be greater than min_value")
        return v


class DiagnosticDefinition(BaseModel):
    """Defines a diagnostic measurement in the accelerator.

    Attributes:
        name: Unique identifier for the diagnostic
        description: Human-readable description
        measurement_type: Type of measurement (e.g., "orbit", "beam_size", "transmission")
        unit: Physical unit
        nominal_value: Expected value under normal operation
        tolerance: Acceptable deviation from nominal
        alarm_threshold: Deviation that triggers an alarm
        tao_variable: Tao variable name to read this diagnostic (optional)
    """

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    measurement_type: str
    unit: str
    nominal_value: float
    tolerance: float = Field(..., ge=0)
    alarm_threshold: float = Field(..., ge=0)
    tao_variable: Optional[str] = None

    @field_validator("alarm_threshold")
    @classmethod
    def validate_alarm_greater_than_tolerance(cls, v: float, info) -> float:
        """Ensure alarm_threshold is greater than tolerance."""
        if "tolerance" in info.data and v < info.data["tolerance"]:
            raise ValueError("alarm_threshold should be >= tolerance")
        return v


class ConstraintDefinition(BaseModel):
    """Defines a safety constraint for the accelerator.

    Attributes:
        constraint_id: Unique identifier for the constraint
        description: Human-readable description
        constraint_type: Type of constraint (e.g., "parameter_limit", "rate_limit", "interlock")
        parameters: Dictionary of constraint-specific parameters
        enabled: Whether this constraint is active
    """

    constraint_id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    constraint_type: str
    parameters: Dict[str, Any]
    enabled: bool = True


class AcceleratorConfig(BaseModel):
    """Complete configuration for an accelerator.

    Attributes:
        name: Accelerator name
        description: Human-readable description
        knobs: List of controllable parameters
        diagnostics: List of diagnostic measurements
        constraints: List of safety constraints
        tao_init_file: Path to Tao initialization file (optional)
        metadata: Additional metadata
    """

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    knobs: List[KnobDefinition] = Field(..., min_length=1)
    diagnostics: List[DiagnosticDefinition] = Field(..., min_length=1)
    constraints: List[ConstraintDefinition] = Field(default_factory=list)
    tao_init_file: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_knob(self, name: str) -> Optional[KnobDefinition]:
        """Get a knob definition by name."""
        for knob in self.knobs:
            if knob.name == name:
                return knob
        return None

    def get_diagnostic(self, name: str) -> Optional[DiagnosticDefinition]:
        """Get a diagnostic definition by name."""
        for diagnostic in self.diagnostics:
            if diagnostic.name == name:
                return diagnostic
        return None

    def get_constraint(self, constraint_id: str) -> Optional[ConstraintDefinition]:
        """Get a constraint definition by ID."""
        for constraint in self.constraints:
            if constraint.constraint_id == constraint_id:
                return constraint
        return None

    @field_validator("knobs")
    @classmethod
    def validate_unique_knob_names(cls, v: List[KnobDefinition]) -> List[KnobDefinition]:
        """Ensure all knob names are unique."""
        names = [knob.name for knob in v]
        if len(names) != len(set(names)):
            raise ValueError("Knob names must be unique")
        return v

    @field_validator("diagnostics")
    @classmethod
    def validate_unique_diagnostic_names(cls, v: List[DiagnosticDefinition]) -> List[DiagnosticDefinition]:
        """Ensure all diagnostic names are unique."""
        names = [diag.name for diag in v]
        if len(names) != len(set(names)):
            raise ValueError("Diagnostic names must be unique")
        return v

    @field_validator("constraints")
    @classmethod
    def validate_unique_constraint_ids(cls, v: List[ConstraintDefinition]) -> List[ConstraintDefinition]:
        """Ensure all constraint IDs are unique."""
        ids = [constraint.constraint_id for constraint in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Constraint IDs must be unique")
        return v
