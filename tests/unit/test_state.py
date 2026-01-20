"""Tests for agent state definition."""

import pytest

from accops_agent.graph.state import AgentState, ProposedAction, ExecutionHistoryEntry, create_initial_state
from accops_agent.accelerator_interface import DiagnosticSnapshot, DiagnosticStatus, ActionResult


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
            "current_diagnostics": [],
            "current_parameters": {},
            "machine_status_summary": "NORMAL",
            "analysis": {
                "interpretation": "All normal",
                "issues": [],
                "strategy": "Test strategy",
                "reasoning": "Test reasoning",
            },
            "proposed_actions": [],
            "action_index": 0,
            "workflow": {
                "awaiting_approval": False,
                "approval_status": "",
                "user_feedback": "",
                "goal_achieved": False,
                "continue_optimization": True,
                "iteration_count": 0,
                "max_iterations": 10,
            },
            "execution_history": [],
            "verification_result": "",
            "error": {"message": None, "type": None},
            "safety_violations": [],
        }

        assert state["user_intent"] == "Optimize beam size"
        assert state["workflow"]["iteration_count"] == 0
        assert state["workflow"]["max_iterations"] == 10

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
        assert state["current_diagnostics"] == []
        assert state["current_parameters"] == {}
        assert state["workflow"]["iteration_count"] == 0
        assert state["workflow"]["max_iterations"] == 10
        assert not state["workflow"]["awaiting_approval"]
        assert not state["workflow"]["goal_achieved"]

    def test_initial_state_has_all_required_fields(self):
        """Test that initial state has all necessary fields."""
        state = create_initial_state("Test")

        # Check key fields are present
        assert "user_intent" in state
        assert "current_diagnostics" in state
        assert "current_parameters" in state
        assert "proposed_actions" in state
        assert "execution_history" in state
        assert "workflow" in state
        assert "iteration_count" in state["workflow"]
        assert "max_iterations" in state["workflow"]
        assert "goal_achieved" in state["workflow"]
        assert "continue_optimization" in state["workflow"]

    def test_initial_state_defaults(self):
        """Test default values in initial state."""
        state = create_initial_state("Test")

        assert state["action_index"] == 0
        assert state["workflow"]["awaiting_approval"] is False
        assert state["workflow"]["approval_status"] == ""
        assert state["workflow"]["goal_achieved"] is False
        assert state["workflow"]["continue_optimization"] is True
        assert state["error"]["message"] is None
        assert state["error"]["type"] is None
        assert "analysis" in state
