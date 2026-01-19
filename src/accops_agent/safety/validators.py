"""Constraint validation for safety checking."""

import logging
import operator
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from ..config.schema import AcceleratorConfig, ConstraintDefinition, KnobDefinition
from ..accelerator_interface import DiagnosticSnapshot
from ..graph.state import ProposedAction
from ..utils.constants import (
    DEFAULT_EXECUTION_HISTORY_MAX_SIZE,
    DEFAULT_MAX_CHANGES_PER_MINUTE,
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
)
from ..utils.exceptions import ConstraintViolationError
from .constraints import (
    BatchValidationResult,
    ConstraintType,
    ValidationResult,
    Violation,
    ViolationSeverity,
)

# Comparison operators for interlock checks
COMPARISON_OPERATORS: Dict[str, Callable[[float, float], bool]] = {
    "greater_than": operator.gt,
    "less_than": operator.lt,
    "equals": operator.eq,
    "greater_than_or_equal": operator.ge,
    "less_than_or_equal": operator.le,
}

logger = logging.getLogger(__name__)


class ConstraintChecker:
    """Validates actions against safety constraints.

    This class checks proposed actions against:
    - Parameter limits (min/max from knob configuration)
    - Rate limits (max change per step from knob configuration)
    - Global rate limits (max changes per time window)
    - Interlocks (conditional restrictions based on diagnostics)
    - Simultaneous change limits

    Attributes:
        config: Accelerator configuration
        execution_times: Deque of recent execution timestamps for rate limiting
        current_diagnostics: Current diagnostic readings for interlock checks
    """

    def __init__(self, config: AcceleratorConfig):
        """Initialize the constraint checker.

        Args:
            config: Accelerator configuration with knobs and constraints
        """
        self.config = config
        self.execution_times: Deque[float] = deque(maxlen=DEFAULT_EXECUTION_HISTORY_MAX_SIZE)
        self.current_diagnostics: Dict[str, DiagnosticSnapshot] = {}

        # Build lookup tables for fast access
        self._knob_map: Dict[str, KnobDefinition] = {
            knob.name: knob for knob in config.knobs
        }
        self._constraint_map: Dict[str, ConstraintDefinition] = {
            c.constraint_id: c for c in config.constraints
        }

        logger.info(
            f"Initialized ConstraintChecker with {len(self._knob_map)} knobs "
            f"and {len(self._constraint_map)} constraints"
        )

    def update_diagnostics(self, diagnostics: List[DiagnosticSnapshot]) -> None:
        """Update current diagnostic readings for interlock checks.

        Args:
            diagnostics: List of current diagnostic snapshots
        """
        self.current_diagnostics = {d.diagnostic_name: d for d in diagnostics}
        logger.debug(f"Updated {len(diagnostics)} diagnostics for interlock checks")

    def record_execution(self) -> None:
        """Record that an action was executed (for rate limiting)."""
        self.execution_times.append(time.time())

    def validate_action(
        self,
        action: ProposedAction,
        action_index: Optional[int] = None,
    ) -> ValidationResult:
        """Validate a single action against all constraints.

        Args:
            action: The proposed action to validate
            action_index: Optional index for batch validation

        Returns:
            ValidationResult with any violations found
        """
        violations: List[Violation] = []
        parameter_name = action.get("parameter_name", "")

        # Check if parameter exists
        knob = self._knob_map.get(parameter_name)
        if not knob:
            violations.append(
                Violation(
                    constraint_type=ConstraintType.PARAMETER_LIMIT,
                    constraint_id=None,
                    parameter_name=parameter_name,
                    message=f"Unknown parameter: {parameter_name}",
                    severity=ViolationSeverity.ERROR,
                )
            )
            return ValidationResult(
                is_valid=False,
                violations=violations,
                action_index=action_index,
                parameter_name=parameter_name,
            )

        # Check parameter limits
        limit_violations = self._check_parameter_limits(action, knob)
        violations.extend(limit_violations)

        # Check rate limit (per-parameter)
        rate_violations = self._check_rate_limit(action, knob)
        violations.extend(rate_violations)

        # Check interlock constraints
        interlock_violations = self._check_interlocks()
        violations.extend(interlock_violations)

        is_valid = not any(v.severity == ViolationSeverity.ERROR for v in violations)

        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            action_index=action_index,
            parameter_name=parameter_name,
        )

    def validate_actions(
        self,
        actions: List[ProposedAction],
    ) -> BatchValidationResult:
        """Validate a batch of actions against all constraints.

        Args:
            actions: List of proposed actions to validate

        Returns:
            BatchValidationResult with results for each action and global violations
        """
        results: List[ValidationResult] = []
        global_violations: List[Violation] = []

        # Validate individual actions
        for i, action in enumerate(actions):
            result = self.validate_action(action, action_index=i)
            results.append(result)

        # Check global constraints
        global_violations.extend(self._check_global_rate_limit())
        global_violations.extend(self._check_max_simultaneous(actions))

        # Determine overall validity
        individual_valid = all(r.is_valid for r in results)
        global_valid = not any(
            v.severity == ViolationSeverity.ERROR for v in global_violations
        )
        is_valid = individual_valid and global_valid

        return BatchValidationResult(
            is_valid=is_valid,
            results=results,
            global_violations=global_violations,
        )

    def validate_and_raise(
        self,
        actions: List[ProposedAction],
    ) -> None:
        """Validate actions and raise exception if invalid (strict mode).

        Args:
            actions: List of proposed actions to validate

        Raises:
            ConstraintViolationError: If any constraint is violated
        """
        result = self.validate_actions(actions)

        if not result.is_valid:
            violations = result.all_violations
            error_messages = [
                str(v) for v in violations if v.severity == ViolationSeverity.ERROR
            ]
            raise ConstraintViolationError(
                f"Safety validation failed: {'; '.join(error_messages)}"
            )

    def _check_parameter_limits(
        self,
        action: ProposedAction,
        knob: KnobDefinition,
    ) -> List[Violation]:
        """Check if proposed value is within parameter limits.

        Args:
            action: The proposed action
            knob: The knob definition with limits

        Returns:
            List of violations (empty if valid)
        """
        violations: List[Violation] = []
        proposed_value = action.get("proposed_value")

        if proposed_value is None:
            violations.append(
                Violation(
                    constraint_type=ConstraintType.PARAMETER_LIMIT,
                    constraint_id=None,
                    parameter_name=knob.name,
                    message="No proposed value specified",
                    severity=ViolationSeverity.ERROR,
                )
            )
            return violations

        # Check minimum
        if proposed_value < knob.min_value:
            violations.append(
                Violation(
                    constraint_type=ConstraintType.PARAMETER_LIMIT,
                    constraint_id=None,
                    parameter_name=knob.name,
                    message=f"Value {proposed_value:.6f} below minimum {knob.min_value:.6f}",
                    severity=ViolationSeverity.ERROR,
                    details={
                        "proposed_value": proposed_value,
                        "min_value": knob.min_value,
                    },
                )
            )

        # Check maximum
        if proposed_value > knob.max_value:
            violations.append(
                Violation(
                    constraint_type=ConstraintType.PARAMETER_LIMIT,
                    constraint_id=None,
                    parameter_name=knob.name,
                    message=f"Value {proposed_value:.6f} above maximum {knob.max_value:.6f}",
                    severity=ViolationSeverity.ERROR,
                    details={
                        "proposed_value": proposed_value,
                        "max_value": knob.max_value,
                    },
                )
            )

        return violations

    def _check_rate_limit(
        self,
        action: ProposedAction,
        knob: KnobDefinition,
    ) -> List[Violation]:
        """Check if proposed change exceeds rate limit.

        Args:
            action: The proposed action
            knob: The knob definition with rate limit

        Returns:
            List of violations (empty if valid)
        """
        violations: List[Violation] = []

        current_value = action.get("current_value")
        proposed_value = action.get("proposed_value")

        if current_value is None or proposed_value is None:
            return violations  # Can't check rate without both values

        change = abs(proposed_value - current_value)

        if change > knob.rate_limit:
            violations.append(
                Violation(
                    constraint_type=ConstraintType.RATE_LIMIT,
                    constraint_id=None,
                    parameter_name=knob.name,
                    message=f"Change {change:.6f} exceeds rate limit {knob.rate_limit:.6f}",
                    severity=ViolationSeverity.ERROR,
                    details={
                        "change": change,
                        "rate_limit": knob.rate_limit,
                        "current_value": current_value,
                        "proposed_value": proposed_value,
                    },
                )
            )

        return violations

    def _check_global_rate_limit(self) -> List[Violation]:
        """Check global rate limit (changes per time window).

        Returns:
            List of violations (empty if valid)
        """
        violations: List[Violation] = []

        # Find global rate limit constraint
        for constraint in self.config.constraints:
            if constraint.constraint_type == "rate_limit" and constraint.enabled:
                max_changes = constraint.parameters.get(
                    "max_changes_per_minute", DEFAULT_MAX_CHANGES_PER_MINUTE
                )
                window_seconds = constraint.parameters.get(
                    "window_seconds", DEFAULT_RATE_LIMIT_WINDOW_SECONDS
                )

                # Count executions in window
                now = time.time()
                recent_count = sum(
                    1 for t in self.execution_times if now - t < window_seconds
                )

                if recent_count >= max_changes:
                    violations.append(
                        Violation(
                            constraint_type=ConstraintType.GLOBAL_RATE_LIMIT,
                            constraint_id=constraint.constraint_id,
                            parameter_name=None,
                            message=f"Global rate limit reached: {recent_count} changes in last {window_seconds}s (max: {max_changes})",
                            severity=ViolationSeverity.ERROR,
                            details={
                                "recent_count": recent_count,
                                "max_changes": max_changes,
                                "window_seconds": window_seconds,
                            },
                        )
                    )

        return violations

    def _check_interlocks(self) -> List[Violation]:
        """Check interlock constraints based on current diagnostics.

        Returns:
            List of violations (empty if valid)
        """
        violations: List[Violation] = []

        for constraint in self.config.constraints:
            if constraint.constraint_type == "interlock" and constraint.enabled:
                diagnostic_name = constraint.parameters.get("diagnostic")
                threshold = constraint.parameters.get("threshold")
                comparison = constraint.parameters.get("comparison", "greater_than")
                action_type = constraint.parameters.get("action", "halt")

                if not diagnostic_name or threshold is None:
                    continue

                # Get current diagnostic value
                diagnostic = self.current_diagnostics.get(diagnostic_name)
                if not diagnostic:
                    continue  # Can't check without diagnostic data

                # Check condition using comparison operators
                compare_func = COMPARISON_OPERATORS.get(comparison)
                if not compare_func:
                    logger.warning(f"Unknown comparison operator: {comparison}")
                    continue

                # Use absolute value for greater_than to handle signed diagnostics
                diagnostic_value = (
                    abs(diagnostic.value) if comparison == "greater_than" else diagnostic.value
                )
                triggered = compare_func(diagnostic_value, threshold)

                if triggered and action_type == "halt":
                    violations.append(
                        Violation(
                            constraint_type=ConstraintType.INTERLOCK,
                            constraint_id=constraint.constraint_id,
                            parameter_name=None,
                            message=f"Interlock triggered: {diagnostic_name}={diagnostic.value:.4f} {comparison} {threshold}",
                            severity=ViolationSeverity.ERROR,
                            details={
                                "diagnostic_name": diagnostic_name,
                                "diagnostic_value": diagnostic.value,
                                "threshold": threshold,
                                "comparison": comparison,
                            },
                        )
                    )

        return violations

    def _check_max_simultaneous(
        self,
        actions: List[ProposedAction],
    ) -> List[Violation]:
        """Check maximum simultaneous changes constraint.

        Args:
            actions: List of proposed actions

        Returns:
            List of violations (empty if valid)
        """
        violations: List[Violation] = []

        for constraint in self.config.constraints:
            if constraint.constraint_type == "limit" and constraint.enabled:
                max_knobs = constraint.parameters.get("max_knobs")

                if max_knobs is not None and len(actions) > max_knobs:
                    violations.append(
                        Violation(
                            constraint_type=ConstraintType.MAX_SIMULTANEOUS,
                            constraint_id=constraint.constraint_id,
                            parameter_name=None,
                            message=f"Too many simultaneous changes: {len(actions)} > {max_knobs}",
                            severity=ViolationSeverity.ERROR,
                            details={
                                "action_count": len(actions),
                                "max_knobs": max_knobs,
                            },
                        )
                    )

        return violations

    def get_constraints_summary(self) -> str:
        """Get a human-readable summary of active constraints.

        Returns:
            Summary string for display or prompts
        """
        lines = ["Active safety constraints:"]

        # Parameter limits from knobs
        lines.append("\nParameter limits:")
        for knob in self.config.knobs:
            lines.append(
                f"  - {knob.name}: [{knob.min_value}, {knob.max_value}] {knob.unit}, "
                f"max change: {knob.rate_limit}"
            )

        # Global constraints
        lines.append("\nGlobal constraints:")
        for constraint in self.config.constraints:
            if constraint.enabled:
                lines.append(f"  - {constraint.description}")

        return "\n".join(lines)


def validate_action(
    action: ProposedAction,
    config: AcceleratorConfig,
    diagnostics: Optional[List[DiagnosticSnapshot]] = None,
) -> ValidationResult:
    """Convenience function to validate a single action.

    Args:
        action: The proposed action
        config: Accelerator configuration
        diagnostics: Optional current diagnostics for interlock checks

    Returns:
        ValidationResult
    """
    checker = ConstraintChecker(config)
    if diagnostics:
        checker.update_diagnostics(diagnostics)
    return checker.validate_action(action)


def validate_actions(
    actions: List[ProposedAction],
    config: AcceleratorConfig,
    diagnostics: Optional[List[DiagnosticSnapshot]] = None,
    raise_on_error: bool = True,
) -> BatchValidationResult:
    """Convenience function to validate multiple actions.

    Args:
        actions: List of proposed actions
        config: Accelerator configuration
        diagnostics: Optional current diagnostics for interlock checks
        raise_on_error: If True, raise ConstraintViolationError on failure

    Returns:
        BatchValidationResult

    Raises:
        ConstraintViolationError: If raise_on_error=True and validation fails
    """
    checker = ConstraintChecker(config)
    if diagnostics:
        checker.update_diagnostics(diagnostics)

    if raise_on_error:
        checker.validate_and_raise(actions)

    return checker.validate_actions(actions)
