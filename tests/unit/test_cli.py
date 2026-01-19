"""Tests for CLI module."""

import pytest
from io import StringIO
from unittest.mock import Mock, patch

from accops_agent.cli.display import (
    Colors,
    format_action_for_display,
    format_actions_table,
    format_approval_prompt,
    format_diagnostic_status,
    format_diagnostic_summary,
    format_diagnostics_table,
    format_execution_result,
    format_state_summary,
    format_verification_result,
    print_banner,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from accops_agent.cli.input_handler import (
    ApprovalResponse,
    ApprovalStatus,
    get_user_input,
    parse_approval_response,
)
from accops_agent.accelerator_interface import (
    ActionResult,
    DiagnosticSnapshot,
    DiagnosticStatus,
)
from accops_agent.graph.state import AgentState, ProposedAction, create_initial_state


class TestColors:
    """Tests for color constants."""

    def test_color_codes_are_strings(self):
        """Test that color codes are strings."""
        assert isinstance(Colors.RESET, str)
        assert isinstance(Colors.BOLD, str)
        assert isinstance(Colors.RED, str)
        assert isinstance(Colors.GREEN, str)
        assert isinstance(Colors.YELLOW, str)
        assert isinstance(Colors.BLUE, str)

    def test_color_codes_start_with_escape(self):
        """Test that color codes are ANSI escape sequences."""
        assert Colors.RESET.startswith("\033[")
        assert Colors.BOLD.startswith("\033[")


class TestPrintFunctions:
    """Tests for print helper functions."""

    def test_print_banner(self, capsys):
        """Test banner printing."""
        print_banner()
        captured = capsys.readouterr()
        assert "AccOps Agent" in captured.out
        assert "Human-in-the-Loop" in captured.out

    def test_print_info(self, capsys):
        """Test info message printing."""
        print_info("Test message")
        captured = capsys.readouterr()
        assert "[INFO]" in captured.out
        assert "Test message" in captured.out

    def test_print_success(self, capsys):
        """Test success message printing."""
        print_success("Operation successful")
        captured = capsys.readouterr()
        assert "[SUCCESS]" in captured.out
        assert "Operation successful" in captured.out

    def test_print_warning(self, capsys):
        """Test warning message printing."""
        print_warning("Warning message")
        captured = capsys.readouterr()
        assert "[WARNING]" in captured.out
        assert "Warning message" in captured.out

    def test_print_error(self, capsys):
        """Test error message printing."""
        print_error("Error occurred")
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out
        assert "Error occurred" in captured.out


class TestFormatDiagnosticStatus:
    """Tests for diagnostic status formatting."""

    def test_format_normal_status(self):
        """Test formatting normal status."""
        result = format_diagnostic_status(DiagnosticStatus.NORMAL)
        assert "NORMAL" in result
        assert Colors.GREEN in result

    def test_format_warning_status(self):
        """Test formatting warning status."""
        result = format_diagnostic_status(DiagnosticStatus.WARNING)
        assert "WARNING" in result
        assert Colors.YELLOW in result

    def test_format_alarm_status(self):
        """Test formatting alarm status."""
        result = format_diagnostic_status(DiagnosticStatus.ALARM)
        assert "ALARM" in result
        assert Colors.RED in result

    def test_format_error_status(self):
        """Test formatting error status."""
        result = format_diagnostic_status(DiagnosticStatus.ERROR)
        assert "ERROR" in result
        assert Colors.RED in result

    def test_format_unknown_status(self):
        """Test formatting unknown status."""
        result = format_diagnostic_status(DiagnosticStatus.UNKNOWN)
        assert "UNKNOWN" in result


class TestFormatDiagnosticsTable:
    """Tests for diagnostics table formatting."""

    @pytest.fixture
    def sample_diagnostics(self):
        """Create sample diagnostic snapshots."""
        return [
            DiagnosticSnapshot(
                diagnostic_name="BPM1_X",
                value=0.123,
                unit="mm",
                status=DiagnosticStatus.NORMAL,
            ),
            DiagnosticSnapshot(
                diagnostic_name="BPM2_X",
                value=1.5,
                unit="mm",
                status=DiagnosticStatus.WARNING,
            ),
            DiagnosticSnapshot(
                diagnostic_name="BPM3_X",
                value=5.0,
                unit="mm",
                status=DiagnosticStatus.ALARM,
            ),
        ]

    def test_format_empty_diagnostics(self):
        """Test formatting empty diagnostic list."""
        result = format_diagnostics_table([])
        assert "No diagnostics available" in result

    def test_format_diagnostics_has_header(self, sample_diagnostics):
        """Test that table has header row."""
        result = format_diagnostics_table(sample_diagnostics)
        assert "Diagnostic" in result
        assert "Value" in result
        assert "Unit" in result
        assert "Status" in result

    def test_format_diagnostics_has_values(self, sample_diagnostics):
        """Test that table contains diagnostic values."""
        result = format_diagnostics_table(sample_diagnostics)
        assert "BPM1_X" in result
        assert "BPM2_X" in result
        assert "BPM3_X" in result
        assert "0.123" in result or "0.1230" in result

    def test_format_diagnostics_has_separator(self, sample_diagnostics):
        """Test that table has separator line."""
        result = format_diagnostics_table(sample_diagnostics)
        assert "---" in result


class TestFormatDiagnosticSummary:
    """Tests for diagnostic summary formatting."""

    @pytest.fixture
    def mixed_diagnostics(self):
        """Create diagnostics with mixed statuses."""
        return [
            DiagnosticSnapshot(
                diagnostic_name="D1", value=0, unit="mm", status=DiagnosticStatus.NORMAL
            ),
            DiagnosticSnapshot(
                diagnostic_name="D2", value=0, unit="mm", status=DiagnosticStatus.NORMAL
            ),
            DiagnosticSnapshot(
                diagnostic_name="D3", value=0, unit="mm", status=DiagnosticStatus.WARNING
            ),
            DiagnosticSnapshot(
                diagnostic_name="D4", value=0, unit="mm", status=DiagnosticStatus.ALARM
            ),
        ]

    def test_format_empty_summary(self):
        """Test summary of empty diagnostics."""
        result = format_diagnostic_summary([])
        assert "No diagnostics available" in result

    def test_format_summary_counts(self, mixed_diagnostics):
        """Test that summary includes counts."""
        result = format_diagnostic_summary(mixed_diagnostics)
        assert "Total: 4" in result
        assert "Normal: 2" in result
        assert "Warning: 1" in result
        assert "Alarm: 1" in result


class TestFormatActionForDisplay:
    """Tests for action formatting."""

    @pytest.fixture
    def sample_action(self) -> ProposedAction:
        """Create sample action."""
        return ProposedAction(
            parameter_name="QF1_K1",
            current_value=1.0,
            proposed_value=1.1,
            rationale="Increase focus strength",
            expected_impact="Reduce beam size",
            priority=1,
        )

    def test_format_action_includes_parameter(self, sample_action):
        """Test that formatted action includes parameter name."""
        result = format_action_for_display(sample_action)
        assert "QF1_K1" in result

    def test_format_action_includes_values(self, sample_action):
        """Test that formatted action includes current and proposed values."""
        result = format_action_for_display(sample_action)
        assert "1.0" in result
        assert "1.1" in result

    def test_format_action_includes_rationale(self, sample_action):
        """Test that formatted action includes rationale."""
        result = format_action_for_display(sample_action)
        assert "Increase focus strength" in result

    def test_format_action_with_index(self, sample_action):
        """Test action formatting with index."""
        result = format_action_for_display(sample_action, index=2)
        assert "Action 3" in result  # 1-based display


class TestFormatActionsTable:
    """Tests for actions table formatting."""

    @pytest.fixture
    def sample_actions(self) -> list[ProposedAction]:
        """Create sample actions list."""
        return [
            ProposedAction(
                parameter_name="QF1_K1",
                current_value=1.0,
                proposed_value=1.1,
                rationale="Increase focus",
            ),
            ProposedAction(
                parameter_name="QD1_K1",
                current_value=-0.5,
                proposed_value=-0.6,
                rationale="Increase defocus",
            ),
        ]

    def test_format_empty_actions(self):
        """Test formatting empty actions list."""
        result = format_actions_table([])
        assert "No actions proposed" in result

    def test_format_actions_has_header(self, sample_actions):
        """Test that table has header."""
        result = format_actions_table(sample_actions)
        assert "Parameter" in result
        assert "Current" in result
        assert "Proposed" in result

    def test_format_actions_has_all_actions(self, sample_actions):
        """Test that table contains all actions."""
        result = format_actions_table(sample_actions)
        assert "QF1_K1" in result
        assert "QD1_K1" in result


class TestFormatExecutionResult:
    """Tests for execution result formatting."""

    def test_format_success_result(self):
        """Test formatting successful execution result."""
        result = ActionResult(
            success=True,
            action_type="set_parameter",
            parameters={"name": "QF1_K1", "value": 1.1},
            message="Parameter set successfully",
        )
        formatted = format_execution_result(result)
        assert "SUCCESS" in formatted
        assert "Parameter set successfully" in formatted

    def test_format_failed_result(self):
        """Test formatting failed execution result."""
        result = ActionResult(
            success=False,
            action_type="set_parameter",
            parameters={"name": "QF1_K1", "value": 1.1},
            message="Failed to set parameter",
            error="Parameter out of range",
        )
        formatted = format_execution_result(result)
        assert "FAILED" in formatted
        assert "Parameter out of range" in formatted


class TestFormatStateSummary:
    """Tests for state summary formatting."""

    def test_format_initial_state(self):
        """Test formatting initial state."""
        state = create_initial_state("Optimize beam")
        formatted = format_state_summary(state)
        assert "Optimize beam" in formatted
        assert "Agent State Summary" in formatted

    def test_format_state_with_diagnostics(self):
        """Test formatting state with diagnostics."""
        state = create_initial_state("Optimize beam")
        state["current_diagnostics"] = [
            DiagnosticSnapshot(
                diagnostic_name="BPM1", value=0.1, unit="mm", status=DiagnosticStatus.NORMAL
            )
        ]
        state["machine_status_summary"] = "NORMAL: All good"
        formatted = format_state_summary(state)
        assert "NORMAL" in formatted

    def test_format_state_with_issues(self):
        """Test formatting state with identified issues."""
        state = create_initial_state("Optimize beam")
        state["identified_issues"] = ["Issue 1", "Issue 2"]
        formatted = format_state_summary(state)
        assert "Issue 1" in formatted
        assert "Issue 2" in formatted


class TestFormatVerificationResult:
    """Tests for verification result formatting."""

    def test_format_verification(self):
        """Test formatting verification result."""
        verification = "Action was effective (score: 8/10)"
        formatted = format_verification_result(verification)
        assert "Verification Result" in formatted
        assert "effective" in formatted


class TestFormatApprovalPrompt:
    """Tests for approval prompt formatting."""

    def test_format_approval_prompt(self):
        """Test formatting approval prompt."""
        actions = [
            ProposedAction(
                parameter_name="QF1_K1",
                current_value=1.0,
                proposed_value=1.1,
                rationale="Test",
            )
        ]
        formatted = format_approval_prompt(actions)
        assert "Pending Approval" in formatted
        # Check for options (color codes may split the text)
        assert "[a]" in formatted and "pprove" in formatted
        assert "[r]" in formatted and "eject" in formatted
        assert "[m]" in formatted and "odify" in formatted
        assert "[d]" in formatted and "etails" in formatted


class TestParseApprovalResponse:
    """Tests for approval response parsing."""

    def test_parse_approve_responses(self):
        """Test parsing approve responses."""
        for response in ["a", "approve", "yes", "y", "A", "APPROVE"]:
            status, details = parse_approval_response(response)
            assert status == ApprovalStatus.APPROVED
            assert details is False

    def test_parse_reject_responses(self):
        """Test parsing reject responses."""
        for response in ["r", "reject", "no", "n", "R", "REJECT"]:
            status, details = parse_approval_response(response)
            assert status == ApprovalStatus.REJECTED
            assert details is False

    def test_parse_modify_responses(self):
        """Test parsing modify responses."""
        for response in ["m", "modify", "feedback", "f", "M"]:
            status, details = parse_approval_response(response)
            assert status == ApprovalStatus.MODIFIED
            assert details is False

    def test_parse_details_responses(self):
        """Test parsing details responses."""
        for response in ["d", "details", "show", "D"]:
            status, details = parse_approval_response(response)
            assert status is None
            assert details is True

    def test_parse_invalid_responses(self):
        """Test parsing invalid responses."""
        for response in ["x", "invalid", "123", ""]:
            status, details = parse_approval_response(response)
            assert status is None
            assert details is False


class TestGetUserInput:
    """Tests for user input handling."""

    def test_get_input_with_value(self):
        """Test getting user input with value."""
        with patch("builtins.input", return_value="test_input"):
            result = get_user_input("Enter value")
            assert result == "test_input"

    def test_get_input_with_default(self):
        """Test getting user input with default."""
        with patch("builtins.input", return_value=""):
            result = get_user_input("Enter value", default="default_value")
            assert result == "default_value"

    def test_get_input_strips_whitespace(self):
        """Test that input is stripped of whitespace."""
        with patch("builtins.input", return_value="  value  "):
            result = get_user_input("Enter value")
            assert result == "value"

    def test_get_input_handles_eof(self):
        """Test handling EOF error."""
        with patch("builtins.input", side_effect=EOFError):
            result = get_user_input("Enter value")
            assert result == ""

    def test_get_input_handles_keyboard_interrupt(self):
        """Test handling keyboard interrupt."""
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = get_user_input("Enter value")
            assert result == ""


class TestApprovalResponse:
    """Tests for ApprovalResponse dataclass."""

    def test_create_approved_response(self):
        """Test creating approved response."""
        response = ApprovalResponse(status=ApprovalStatus.APPROVED)
        assert response.status == ApprovalStatus.APPROVED
        assert response.feedback is None

    def test_create_modified_response_with_feedback(self):
        """Test creating modified response with feedback."""
        response = ApprovalResponse(
            status=ApprovalStatus.MODIFIED,
            feedback="Change the value to 1.2 instead",
        )
        assert response.status == ApprovalStatus.MODIFIED
        assert "1.2" in response.feedback

    def test_create_rejected_response(self):
        """Test creating rejected response."""
        response = ApprovalResponse(status=ApprovalStatus.REJECTED)
        assert response.status == ApprovalStatus.REJECTED
