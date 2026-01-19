"""Tests for agent state definition."""

import pytest

from accops_agent.graph.state import AgentState, ProposedAction, ExecutionHistoryEntry, create_initial_state
from accops_agent.diagnostic_control import DiagnosticSnapshot, DiagnosticStatus, ActionResult


class TestProposedAction:
    """Tests for ProposedAction TypedDict."""

    def test_create_proposed_action(self):
        """Test creating a ProposedAction."""
        action: ProposedAction = {
            "parameter_name": "QF1_K1",
            "current_value": 2.0,
            "proposed_value": 2.5,
            "rationale": "Increase focusing strength",
            "expected_impact": "Reduce beam size",
            "priority": 1,
        }

        assert action["parameter_name"] == "QF1_K1"
        assert action["current_value"] == 2.0
        assert action["proposed_value"] == 2.5
        assert action["priority"] == 1

    def test_proposed_action_partial(self):
        """Test creating ProposedAction with partial fields."""
        action: ProposedAction = {
            "parameter_name": "QF1_K1",
            "proposed_value": 2.5,
        }

        assert action["parameter_name"] == "QF1_K1"
        assert "current_value" not in action


class TestExecutionHistoryEntry:
    """Tests for ExecutionHistoryEntry TypedDict."""

    def test_create_execution_history_entry(self):
        """Test creating an ExecutionHistoryEntry."""
        action: ProposedAction = {
            "parameter_name": "QF1_K1",
            "current_value": 2.0,
            "proposed_value": 2.5,
            "rationale": "Test",
            "expected_impact": "Test impact",
            "priority": 1,
        }

        result = ActionResult(
            success=True,
            action_type="set_parameter",
            parameters={"parameter_name": "QF1_K1", "value": 2.5},
            message="Success",
        )

        diag = DiagnosticSnapshot(
            diagnostic_name="BPM1_X",
            value=0.5,
            unit="mm",
            status=DiagnosticStatus.NORMAL,
        )

        entry: ExecutionHistoryEntry = {
            "action": action,
            "result": result,
            "diagnostics_before": [diag],
            "diagnostics_after": [diag],
            "timestamp": "2024-01-01T00:00:00",
        }

        assert entry["action"]["parameter_name"] == "QF1_K1"
        assert entry["result"].success
        assert len(entry["diagnostics_before"]) == 1


class TestAgentState:
    """Tests for AgentState TypedDict."""

    def test_create_minimal_state(self):
        """Test creating minimal AgentState."""
        state: AgentState = {
            "user_intent": "Optimize beam size",
        }

        assert state["user_intent"] == "Optimize beam size"

    def test_create_complete_state(self):
        """Test creating complete AgentState with all fields."""
        state: AgentState = {
            "user_intent": "Optimize beam size",
            "backend_type": "mock",
            "current_diagnostics": [],
            "current_parameters": {},
            "machine_status_summary": "NORMAL",
            "diagnostic_interpretation": "All normal",
            "identified_issues": [],
            "strategy": "Test strategy",
            "reasoning": "Test reasoning",
            "proposed_actions": [],
            "action_index": 0,
            "awaiting_approval": False,
            "approval_status": "",
            "user_feedback": "",
            "execution_history": [],
            "verification_result": "",
            "goal_achieved": False,
            "continue_optimization": True,
            "iteration_count": 0,
            "max_iterations": 10,
            "error": None,
            "error_type": None,
            "metadata": {},
        }

        assert state["user_intent"] == "Optimize beam size"
        assert state["backend_type"] == "mock"
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 10

    def test_state_with_diagnostics(self):
        """Test state with diagnostic data."""
        diag = DiagnosticSnapshot(
            diagnostic_name="BPM1_X",
            value=0.5,
            unit="mm",
            status=DiagnosticStatus.NORMAL,
        )

        state: AgentState = {
            "user_intent": "Check orbit",
            "current_diagnostics": [diag],
        }

        assert len(state["current_diagnostics"]) == 1
        assert state["current_diagnostics"][0].diagnostic_name == "BPM1_X"

    def test_state_with_proposed_actions(self):
        """Test state with proposed actions."""
        action: ProposedAction = {
            "parameter_name": "QF1_K1",
            "current_value": 2.0,
            "proposed_value": 2.5,
            "rationale": "Test",
            "expected_impact": "Test impact",
            "priority": 1,
        }

        state: AgentState = {
            "user_intent": "Test",
            "proposed_actions": [action],
            "action_index": 0,
        }

        assert len(state["proposed_actions"]) == 1
        assert state["proposed_actions"][0]["parameter_name"] == "QF1_K1"
        assert state["action_index"] == 0


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_create_initial_state_minimal(self):
        """Test creating initial state with minimal args."""
        state = create_initial_state("Optimize beam size")

        assert state["user_intent"] == "Optimize beam size"
        assert state["backend_type"] == "mock"
        assert state["current_diagnostics"] == []
        assert state["current_parameters"] == {}
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 10
        assert not state["awaiting_approval"]
        assert not state["goal_achieved"]

    def test_create_initial_state_with_backend_type(self):
        """Test creating initial state with custom backend type."""
        state = create_initial_state("Test intent", backend_type="pytao")

        assert state["user_intent"] == "Test intent"
        assert state["backend_type"] == "pytao"

    def test_initial_state_has_all_required_fields(self):
        """Test that initial state has all necessary fields."""
        state = create_initial_state("Test")

        # Check key fields are present
        assert "user_intent" in state
        assert "backend_type" in state
        assert "current_diagnostics" in state
        assert "current_parameters" in state
        assert "proposed_actions" in state
        assert "execution_history" in state
        assert "iteration_count" in state
        assert "max_iterations" in state
        assert "goal_achieved" in state
        assert "continue_optimization" in state

    def test_initial_state_defaults(self):
        """Test default values in initial state."""
        state = create_initial_state("Test")

        assert state["action_index"] == 0
        assert state["awaiting_approval"] is False
        assert state["approval_status"] == ""
        assert state["goal_achieved"] is False
        assert state["continue_optimization"] is True
        assert state["error"] is None
        assert state["error_type"] is None
        assert isinstance(state["metadata"], dict)
