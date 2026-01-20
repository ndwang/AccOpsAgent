"""Tests for safety validation module."""

import time
import pytest
from pathlib import Path

from accops_agent.config import load_accelerator_config
from accops_agent.accelerator_interface import DiagnosticSnapshot, DiagnosticStatus
from accops_agent.graph.state import ProposedAction
from accops_agent.safety import (
    ConstraintChecker,
    ConstraintType,
    ValidationResult,
    Violation,
    validate_action,
    validate_actions,
)
from accops_agent.safety.constraints import (
    BatchValidationResult,
    ViolationSeverity,
)
from accops_agent.utils.exceptions import ConstraintViolationError


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
def checker(test_config):
    """Create a constraint checker."""
    return ConstraintChecker(test_config)


@pytest.fixture
def valid_action() -> ProposedAction:
    """Create a valid action within limits."""
    return ProposedAction(
        parameter_name="QF1_K1",
        current_value=1.0,
        proposed_value=1.2,  # Within rate limit of 0.5
        rationale="Test action",
    )


@pytest.fixture
def out_of_bounds_action() -> ProposedAction:
    """Create an action with value outside limits."""
    return ProposedAction(
        parameter_name="QF1_K1",
        current_value=4.0,
        proposed_value=10.0,  # Max is 5.0
        rationale="Test action",
    )


@pytest.fixture
def rate_limit_exceeded_action() -> ProposedAction:
    """Create an action that exceeds rate limit."""
    return ProposedAction(
        parameter_name="QF1_K1",
        current_value=1.0,
        proposed_value=2.0,  # Change of 1.0, rate limit is 0.5
        rationale="Test action",
    )


@pytest.fixture
def unknown_parameter_action() -> ProposedAction:
    """Create an action with unknown parameter."""
    return ProposedAction(
        parameter_name="NONEXISTENT_PARAM",
        current_value=0.0,
        proposed_value=1.0,
        rationale="Test action",
    )


class TestViolation:
    """Tests for Violation class."""

    def test_violation_str_with_parameter(self):
        """Test string representation with parameter name."""
        violation = Violation(
            constraint_type=ConstraintType.PARAMETER_LIMIT,
            constraint_id=None,
            parameter_name="QF1_K1",
            message="Value out of range",
            severity=ViolationSeverity.ERROR,
        )
        result = str(violation)
        assert "ERROR" in result
        assert "QF1_K1" in result
        assert "Value out of range" in result

    def test_violation_str_without_parameter(self):
        """Test string representation without parameter name."""
        violation = Violation(
            constraint_type=ConstraintType.GLOBAL_RATE_LIMIT,
            constraint_id="global_limit",
            parameter_name=None,
            message="Rate limit exceeded",
            severity=ViolationSeverity.WARNING,
        )
        result = str(violation)
        assert "WARNING" in result
        assert "Rate limit exceeded" in result


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_valid_result(self):
        """Test a valid result with no violations."""
        result = ValidationResult(is_valid=True, violations=[])
        assert result.is_valid
        assert not result.has_errors
        assert not result.has_warnings
        assert "passed" in str(result).lower()

    def test_result_with_errors(self):
        """Test result with error violations."""
        result = ValidationResult(
            is_valid=False,
            violations=[
                Violation(
                    constraint_type=ConstraintType.PARAMETER_LIMIT,
                    constraint_id=None,
                    parameter_name="QF1",
                    message="Out of range",
                    severity=ViolationSeverity.ERROR,
                )
            ],
        )
        assert not result.is_valid
        assert result.has_errors
        assert len(result.error_messages) == 1

    def test_result_with_warnings(self):
        """Test result with warning violations."""
        result = ValidationResult(
            is_valid=True,
            violations=[
                Violation(
                    constraint_type=ConstraintType.RATE_LIMIT,
                    constraint_id=None,
                    parameter_name="QF1",
                    message="Near limit",
                    severity=ViolationSeverity.WARNING,
                )
            ],
        )
        assert result.is_valid
        assert result.has_warnings
        assert len(result.warning_messages) == 1


class TestBatchValidationResult:
    """Tests for BatchValidationResult class."""

    def test_batch_all_valid(self):
        """Test batch with all valid actions."""
        result = BatchValidationResult(
            is_valid=True,
            results=[
                ValidationResult(is_valid=True, action_index=0),
                ValidationResult(is_valid=True, action_index=1),
            ],
        )
        assert result.is_valid
        assert len(result.failed_actions) == 0

    def test_batch_with_failures(self):
        """Test batch with some invalid actions."""
        result = BatchValidationResult(
            is_valid=False,
            results=[
                ValidationResult(is_valid=True, action_index=0),
                ValidationResult(is_valid=False, action_index=1),
            ],
        )
        assert not result.is_valid
        assert result.failed_actions == [1]

    def test_batch_with_global_violations(self):
        """Test batch with global violations."""
        result = BatchValidationResult(
            is_valid=False,
            results=[ValidationResult(is_valid=True, action_index=0)],
            global_violations=[
                Violation(
                    constraint_type=ConstraintType.MAX_SIMULTANEOUS,
                    constraint_id="test",
                    parameter_name=None,
                    message="Too many changes",
                    severity=ViolationSeverity.ERROR,
                )
            ],
        )
        assert not result.is_valid
        assert len(result.all_violations) == 1


class TestConstraintChecker:
    """Tests for ConstraintChecker class."""

    def test_checker_initialization(self, checker, test_config):
        """Test checker initializes with config."""
        assert checker.config == test_config
        assert len(checker._knob_map) == len(test_config.knobs)
        assert len(checker._constraint_map) == len(test_config.constraints)

    def test_validate_valid_action(self, checker, valid_action):
        """Test validation of a valid action."""
        result = checker.validate_action(valid_action)
        assert result.is_valid
        assert len(result.violations) == 0

    def test_validate_unknown_parameter(self, checker, unknown_parameter_action):
        """Test validation rejects unknown parameters."""
        result = checker.validate_action(unknown_parameter_action)
        assert not result.is_valid
        assert any(
            "Unknown parameter" in str(v) for v in result.violations
        )

    def test_validate_out_of_bounds_high(self, checker, out_of_bounds_action):
        """Test validation rejects values above maximum."""
        result = checker.validate_action(out_of_bounds_action)
        assert not result.is_valid
        assert any(
            "above maximum" in str(v) for v in result.violations
        )

    def test_validate_out_of_bounds_low(self, checker):
        """Test validation rejects values below minimum."""
        action = ProposedAction(
            parameter_name="QF1_K1",
            current_value=-4.0,
            proposed_value=-10.0,  # Min is -5.0
            rationale="Test",
        )
        result = checker.validate_action(action)
        assert not result.is_valid
        assert any(
            "below minimum" in str(v) for v in result.violations
        )

    def test_validate_rate_limit_exceeded(self, checker, rate_limit_exceeded_action):
        """Test validation rejects changes exceeding rate limit."""
        result = checker.validate_action(rate_limit_exceeded_action)
        assert not result.is_valid
        assert any(
            "exceeds rate limit" in str(v) for v in result.violations
        )

    def test_validate_within_rate_limit(self, checker):
        """Test validation accepts changes within rate limit."""
        action = ProposedAction(
            parameter_name="QF1_K1",
            current_value=1.0,
            proposed_value=1.4,  # Change of 0.4, limit is 0.5
            rationale="Test",
        )
        result = checker.validate_action(action)
        assert result.is_valid

    def test_validate_actions_batch(self, checker, valid_action):
        """Test batch validation of multiple valid actions."""
        actions = [
            valid_action,
            ProposedAction(
                parameter_name="QD1_K1",
                current_value=-1.0,
                proposed_value=-1.2,
                rationale="Test",
            ),
        ]
        result = checker.validate_actions(actions)
        assert result.is_valid
        assert len(result.results) == 2

    def test_validate_actions_with_invalid(self, checker, valid_action, out_of_bounds_action):
        """Test batch validation with some invalid actions."""
        actions = [valid_action, out_of_bounds_action]
        result = checker.validate_actions(actions)
        assert not result.is_valid
        assert 1 in result.failed_actions

    def test_max_simultaneous_constraint(self, checker):
        """Test max simultaneous changes constraint."""
        # Config has max_knobs: 3, so 4 actions should fail
        actions = [
            ProposedAction(
                parameter_name="QF1_K1",
                current_value=1.0,
                proposed_value=1.1,
                rationale="Test",
            ),
            ProposedAction(
                parameter_name="QD1_K1",
                current_value=-1.0,
                proposed_value=-1.1,
                rationale="Test",
            ),
            ProposedAction(
                parameter_name="HCOR1_KICK",
                current_value=0.0,
                proposed_value=0.0001,
                rationale="Test",
            ),
            ProposedAction(
                parameter_name="VCOR1_KICK",
                current_value=0.0,
                proposed_value=0.0001,
                rationale="Test",
            ),
        ]
        result = checker.validate_actions(actions)
        assert not result.is_valid
        assert any(
            v.constraint_type == ConstraintType.MAX_SIMULTANEOUS
            for v in result.global_violations
        )

    def test_validate_and_raise_valid(self, checker, valid_action):
        """Test validate_and_raise doesn't raise for valid actions."""
        # Should not raise
        checker.validate_and_raise([valid_action])

    def test_validate_and_raise_invalid(self, checker, out_of_bounds_action):
        """Test validate_and_raise raises for invalid actions."""
        with pytest.raises(ConstraintViolationError) as exc_info:
            checker.validate_and_raise([out_of_bounds_action])
        assert "Safety validation failed" in str(exc_info.value)


class TestInterlocks:
    """Tests for interlock constraints."""

    def test_interlock_not_triggered(self, checker):
        """Test interlock passes when diagnostic is normal."""
        # Update diagnostics with normal values
        diagnostics = [
            DiagnosticSnapshot(
                diagnostic_name="BPM1_X",
                value=0.5,  # Well below threshold of 5.0
                unit="mm",
                status=DiagnosticStatus.NORMAL,
            ),
            DiagnosticSnapshot(
                diagnostic_name="TRANSMISSION",
                value=95.0,  # Above threshold of 80.0
                unit="percent",
                status=DiagnosticStatus.NORMAL,
            ),
        ]
        checker.update_diagnostics(diagnostics)

        action = ProposedAction(
            parameter_name="QF1_K1",
            current_value=1.0,
            proposed_value=1.2,
            rationale="Test",
        )
        result = checker.validate_action(action)
        assert result.is_valid

    def test_interlock_triggered_orbit(self, checker):
        """Test interlock triggers when orbit exceeds threshold."""
        diagnostics = [
            DiagnosticSnapshot(
                diagnostic_name="BPM1_X",
                value=6.0,  # Above threshold of 5.0
                unit="mm",
                status=DiagnosticStatus.ALARM,
            ),
        ]
        checker.update_diagnostics(diagnostics)

        action = ProposedAction(
            parameter_name="QF1_K1",
            current_value=1.0,
            proposed_value=1.2,
            rationale="Test",
        )
        result = checker.validate_action(action)
        assert not result.is_valid
        assert any(
            v.constraint_type == ConstraintType.INTERLOCK for v in result.violations
        )

    def test_interlock_triggered_transmission(self, checker):
        """Test interlock triggers when transmission drops below threshold."""
        diagnostics = [
            DiagnosticSnapshot(
                diagnostic_name="TRANSMISSION",
                value=70.0,  # Below threshold of 80.0
                unit="percent",
                status=DiagnosticStatus.ALARM,
            ),
        ]
        checker.update_diagnostics(diagnostics)

        action = ProposedAction(
            parameter_name="QF1_K1",
            current_value=1.0,
            proposed_value=1.2,
            rationale="Test",
        )
        result = checker.validate_action(action)
        assert not result.is_valid
        assert any(
            "Interlock triggered" in str(v) for v in result.violations
        )


class TestGlobalRateLimit:
    """Tests for global rate limiting."""

    def test_global_rate_limit_not_exceeded(self, checker, valid_action):
        """Test global rate limit passes when under limit."""
        # No prior executions
        result = checker.validate_actions([valid_action])
        assert not any(
            v.constraint_type == ConstraintType.GLOBAL_RATE_LIMIT
            for v in result.global_violations
        )

    def test_global_rate_limit_exceeded(self, checker, valid_action):
        """Test global rate limit fails when exceeded."""
        # Simulate many recent executions
        now = time.time()
        for i in range(10):
            checker.execution_times.append(now - i)

        result = checker.validate_actions([valid_action])
        assert any(
            v.constraint_type == ConstraintType.GLOBAL_RATE_LIMIT
            for v in result.global_violations
        )

    def test_global_rate_limit_window_expiry(self, checker, valid_action):
        """Test global rate limit resets after window expires."""
        # Simulate old executions (outside window)
        old_time = time.time() - 120  # 2 minutes ago
        for i in range(10):
            checker.execution_times.append(old_time - i)

        result = checker.validate_actions([valid_action])
        assert not any(
            v.constraint_type == ConstraintType.GLOBAL_RATE_LIMIT
            for v in result.global_violations
        )


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_validate_action_function(self, test_config, valid_action):
        """Test validate_action convenience function."""
        result = validate_action(valid_action, test_config)
        assert result.is_valid

    def test_validate_action_with_diagnostics(self, test_config, valid_action):
        """Test validate_action with diagnostics."""
        diagnostics = [
            DiagnosticSnapshot(
                diagnostic_name="BPM1_X",
                value=0.5,
                unit="mm",
                status=DiagnosticStatus.NORMAL,
            ),
        ]
        result = validate_action(valid_action, test_config, diagnostics=diagnostics)
        assert result.is_valid

    def test_validate_actions_function(self, test_config, valid_action):
        """Test validate_actions convenience function."""
        result = validate_actions(
            [valid_action], test_config, raise_on_error=False
        )
        assert result.is_valid

    def test_validate_actions_raises_on_error(self, test_config, out_of_bounds_action):
        """Test validate_actions raises when invalid and raise_on_error=True."""
        with pytest.raises(ConstraintViolationError):
            validate_actions([out_of_bounds_action], test_config, raise_on_error=True)


class TestConstraintsSummary:
    """Tests for constraints summary generation."""

    def test_get_constraints_summary(self, checker):
        """Test constraints summary generation."""
        summary = checker.get_constraints_summary()
        assert "safety constraints" in summary.lower()
        assert "QF1_K1" in summary
        assert "Parameter limits" in summary
        assert "Global constraints" in summary

    def test_summary_includes_all_knobs(self, checker, test_config):
        """Test summary includes all knob limits."""
        summary = checker.get_constraints_summary()
        for knob in test_config.knobs:
            assert knob.name in summary


class TestEdgeCases:
    """Tests for edge cases and special conditions."""

    def test_action_without_current_value(self, checker):
        """Test validation when current_value is missing (rate limit skipped)."""
        action = ProposedAction(
            parameter_name="QF1_K1",
            proposed_value=1.0,  # No current_value
            rationale="Test",
        )
        result = checker.validate_action(action)
        # Should still check parameter limits but skip rate limit
        assert result.is_valid

    def test_action_without_proposed_value(self, checker):
        """Test validation when proposed_value is missing."""
        action = ProposedAction(
            parameter_name="QF1_K1",
            current_value=1.0,
            # No proposed_value
            rationale="Test",
        )
        result = checker.validate_action(action)
        assert not result.is_valid
        assert any("No proposed value" in str(v) for v in result.violations)

    def test_empty_actions_list(self, checker):
        """Test validation of empty actions list."""
        result = checker.validate_actions([])
        assert result.is_valid
        assert len(result.results) == 0

    def test_action_at_exact_limit(self, checker):
        """Test action exactly at parameter limits."""
        action = ProposedAction(
            parameter_name="QF1_K1",
            current_value=4.5,
            proposed_value=5.0,  # Exactly at max
            rationale="Test",
        )
        result = checker.validate_action(action)
        assert result.is_valid

    def test_action_at_exact_rate_limit(self, checker):
        """Test action exactly at rate limit."""
        action = ProposedAction(
            parameter_name="QF1_K1",
            current_value=1.0,
            proposed_value=1.5,  # Change of exactly 0.5
            rationale="Test",
        )
        result = checker.validate_action(action)
        assert result.is_valid
