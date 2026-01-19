"""Constraint data models for safety validation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConstraintType(str, Enum):
    """Types of safety constraints."""

    PARAMETER_LIMIT = "parameter_limit"  # min/max bounds from knob config
    RATE_LIMIT = "rate_limit"  # max change per step from knob config
    GLOBAL_RATE_LIMIT = "global_rate_limit"  # max changes per time window
    INTERLOCK = "interlock"  # conditional restrictions based on diagnostics
    MAX_SIMULTANEOUS = "limit"  # max knobs changed at once


class ViolationSeverity(str, Enum):
    """Severity levels for constraint violations."""

    ERROR = "error"  # Must be rejected
    WARNING = "warning"  # Can proceed with caution


@dataclass
class Violation:
    """A single constraint violation.

    Attributes:
        constraint_type: Type of constraint violated
        constraint_id: ID of the constraint (if from config)
        parameter_name: Name of the parameter involved (if applicable)
        message: Human-readable description of the violation
        severity: Severity level (error or warning)
        details: Additional details about the violation
    """

    constraint_type: ConstraintType
    constraint_id: Optional[str]
    parameter_name: Optional[str]
    message: str
    severity: ViolationSeverity = ViolationSeverity.ERROR
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of the violation."""
        prefix = f"[{self.severity.value.upper()}]"
        if self.parameter_name:
            return f"{prefix} {self.parameter_name}: {self.message}"
        return f"{prefix} {self.message}"


@dataclass
class ValidationResult:
    """Result of validating an action or set of actions.

    Attributes:
        is_valid: Whether all constraints passed
        violations: List of constraint violations
        action_index: Index of the action (for batch validation)
        parameter_name: Name of the parameter validated
    """

    is_valid: bool
    violations: List[Violation] = field(default_factory=list)
    action_index: Optional[int] = None
    parameter_name: Optional[str] = None

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level violations."""
        return any(v.severity == ViolationSeverity.ERROR for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level violations."""
        return any(v.severity == ViolationSeverity.WARNING for v in self.violations)

    @property
    def error_messages(self) -> List[str]:
        """Get all error messages."""
        return [
            str(v) for v in self.violations if v.severity == ViolationSeverity.ERROR
        ]

    @property
    def warning_messages(self) -> List[str]:
        """Get all warning messages."""
        return [
            str(v) for v in self.violations if v.severity == ViolationSeverity.WARNING
        ]

    def __str__(self) -> str:
        """String representation of the validation result."""
        if self.is_valid:
            return "Validation passed"

        messages = [str(v) for v in self.violations]
        return f"Validation failed: {'; '.join(messages)}"


@dataclass
class BatchValidationResult:
    """Result of validating multiple actions.

    Attributes:
        is_valid: Whether all actions passed validation
        results: Individual validation results for each action
        global_violations: Violations that apply to the batch as a whole
    """

    is_valid: bool
    results: List[ValidationResult] = field(default_factory=list)
    global_violations: List[Violation] = field(default_factory=list)

    @property
    def all_violations(self) -> List[Violation]:
        """Get all violations from all results."""
        violations = list(self.global_violations)
        for result in self.results:
            violations.extend(result.violations)
        return violations

    @property
    def failed_actions(self) -> List[int]:
        """Get indices of failed actions."""
        return [r.action_index for r in self.results if not r.is_valid and r.action_index is not None]

    def __str__(self) -> str:
        """String representation of the batch validation result."""
        if self.is_valid:
            return f"Batch validation passed ({len(self.results)} actions)"

        failed = len([r for r in self.results if not r.is_valid])
        return f"Batch validation failed: {failed}/{len(self.results)} actions invalid"
