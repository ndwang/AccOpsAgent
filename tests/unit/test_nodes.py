"""Tests for graph node implementations."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from accops_agent.config import load_accelerator_config
from accops_agent.accelerator_interface import DiagnosticStatus
from accops_agent.accelerator_interface.mock_backend import MockBackend
from accops_agent.graph.state import create_initial_state, ProposedAction
from accops_agent.graph.nodes import (
    ingest_diagnostics_node,
    interpret_diagnostics_node,
    reasoning_planning_node,
    generate_actions_node,
    human_approval_node,
    execute_action_node,
    verify_results_node,
    decide_continuation_node,
)
from accops_agent.utils.exceptions import GraphExecutionError


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
def mock_backend(test_config):
    """Create and initialize mock backend."""
    backend = MockBackend(test_config)
    backend.initialize()
    return backend


@pytest.fixture
def node_config(mock_backend):
    """Create node config with backend."""
    return {"configurable": {"backend": mock_backend}}


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    client.generate = Mock(return_value="Mock LLM response")
    return client


@pytest.fixture
def node_config_with_llm(mock_backend, mock_llm_client):
    """Create node config with backend and LLM client."""
    return {"configurable": {"backend": mock_backend, "llm_client": mock_llm_client}}


class TestIngestDiagnosticsNode:
    """Tests for ingest_diagnostics_node."""

    def test_ingest_diagnostics_success(self, mock_backend, node_config):
        """Test successful diagnostic ingestion."""
        state = create_initial_state("Test")
        result = ingest_diagnostics_node(state, node_config)

        assert "current_diagnostics" in result
        assert "current_parameters" in result
        assert "machine_status_summary" in result
        assert len(result["current_diagnostics"]) > 0
        assert len(result["current_parameters"]) > 0

    def test_ingest_diagnostics_status_normal(self, mock_backend, node_config):
        """Test status summary when all diagnostics normal."""
        state = create_initial_state("Test")
        result = ingest_diagnostics_node(state, node_config)

        assert "NORMAL" in result["machine_status_summary"]

    def test_ingest_diagnostics_status_alarm(self, mock_backend, node_config):
        """Test status summary with alarm conditions."""
        # Set a diagnostic to alarm state
        mock_backend.set_mock_diagnostic("BPM1_X", 5.0)  # Far from nominal

        state = create_initial_state("Test")
        result = ingest_diagnostics_node(state, node_config)

        # Should detect alarm
        assert "ALARM" in result["machine_status_summary"] or "WARNING" in result["machine_status_summary"]

    def test_ingest_diagnostics_not_connected(self, mock_backend, node_config):
        """Test error handling when backend not connected."""
        mock_backend.shutdown()

        state = create_initial_state("Test")
        result = ingest_diagnostics_node(state, node_config)

        assert "error" in result
        assert result["error"]["type"] == "diagnostic_read_error"


class TestInterpretDiagnosticsNode:
    """Tests for interpret_diagnostics_node."""

    def test_interpret_no_diagnostics(self):
        """Test interpretation with no diagnostics."""
        state = create_initial_state("Test")
        result = interpret_diagnostics_node(state)

        assert "analysis" in result
        assert "interpretation" in result["analysis"]
        assert "issues" in result["analysis"]
        assert "No diagnostic" in result["analysis"]["interpretation"]

    def test_interpret_requires_llm_client(self, mock_backend):
        """Test that interpretation requires LLM client."""
        state = create_initial_state("Test")
        state["current_diagnostics"] = mock_backend.read_all_diagnostics()

        with pytest.raises(GraphExecutionError):
            interpret_diagnostics_node(state)

    def test_interpret_normal_diagnostics(self, mock_backend, node_config_with_llm, mock_llm_client):
        """Test interpretation with normal diagnostics."""
        # Configure mock to return a normal interpretation
        mock_llm_client.generate.return_value = "All diagnostics are within normal operating parameters. No issues detected."

        state = create_initial_state("Test")
        state["current_diagnostics"] = mock_backend.read_all_diagnostics()

        result = interpret_diagnostics_node(state, node_config_with_llm)

        assert "analysis" in result
        assert "interpretation" in result["analysis"]
        assert "issues" in result["analysis"]
        mock_llm_client.generate.assert_called_once()

    def test_interpret_with_alarms(self, mock_backend, node_config_with_llm, mock_llm_client):
        """Test interpretation with alarm conditions."""
        # Configure mock to return an interpretation with issues
        mock_llm_client.generate.return_value = "ALARM detected:\n- BPM1_X: Large deviation from nominal value"

        mock_backend.set_mock_diagnostic("BPM1_X", 5.0)
        state = create_initial_state("Test")
        state["current_diagnostics"] = mock_backend.read_all_diagnostics()

        result = interpret_diagnostics_node(state, node_config_with_llm)

        assert "analysis" in result
        assert "issues" in result["analysis"]
        assert "interpretation" in result["analysis"]
        mock_llm_client.generate.assert_called_once()


class TestReasoningPlanningNode:
    """Tests for reasoning_planning_node."""

    def test_reasoning_no_intent(self):
        """Test reasoning with no user intent."""
        state = create_initial_state("")
        state["user_intent"] = ""
        result = reasoning_planning_node(state)

        assert "analysis" in result
        assert "strategy" in result["analysis"]
        assert "reasoning" in result["analysis"]

    def test_reasoning_requires_llm_client(self):
        """Test that reasoning requires LLM client."""
        state = create_initial_state("Optimize beam size")
        state["analysis"]["issues"] = []

        with pytest.raises(GraphExecutionError):
            reasoning_planning_node(state)

    def test_reasoning_with_intent_no_issues(self, node_config_with_llm, mock_llm_client):
        """Test reasoning with intent but no issues."""
        mock_llm_client.generate.return_value = "**Strategy**: Optimize beam size by adjusting quadrupole strengths.\n\n**Reasoning**: Focus on beam optics."

        state = create_initial_state("Optimize beam size")
        state["analysis"]["issues"] = []

        result = reasoning_planning_node(state, node_config_with_llm)

        assert "analysis" in result
        assert "strategy" in result["analysis"]
        assert "reasoning" in result["analysis"]
        mock_llm_client.generate.assert_called_once()

    def test_reasoning_with_issues(self, node_config_with_llm, mock_llm_client):
        """Test reasoning with identified issues."""
        mock_llm_client.generate.return_value = "**Strategy**: Address the issue with BPM1_X.\n\n**Reasoning**: Correct the orbit deviation."

        state = create_initial_state("Correct orbit")
        state["analysis"]["issues"] = ["BPM1_X: ALARM - Large deviation"]

        result = reasoning_planning_node(state, node_config_with_llm)

        assert "analysis" in result
        assert "strategy" in result["analysis"]
        assert "reasoning" in result["analysis"]
        mock_llm_client.generate.assert_called_once()


class TestGenerateActionsNode:
    """Tests for generate_actions_node."""

    def test_generate_actions_requires_llm_and_backend(self):
        """Test that action generation requires LLM client and backend."""
        state = create_initial_state("Optimize beam size")
        state["current_parameters"] = {"QF1_K1": 2.0}

        with pytest.raises(GraphExecutionError):
            generate_actions_node(state)

    def test_generate_actions_beam_size_optimization(self, node_config_with_llm, mock_llm_client):
        """Test action generation for beam size optimization."""
        # Configure mock to return valid JSON actions
        mock_llm_client.generate.return_value = '''```json
[{"parameter_name": "QF1_K1", "current_value": 2.0, "proposed_value": 2.5, "rationale": "Increase focusing", "expected_impact": "Reduce beam size", "priority": 1}]
```'''

        state = create_initial_state("Optimize beam size")
        state["current_parameters"] = {
            "QF1_K1": 2.0,
            "QD1_K1": -1.5,
            "HCOR1_KICK": 0.0,
        }

        result = generate_actions_node(state, node_config_with_llm)

        assert "proposed_actions" in result
        assert len(result["proposed_actions"]) > 0
        assert "action_index" in result
        assert result["action_index"] == 0
        mock_llm_client.generate.assert_called_once()

    def test_generate_actions_orbit_correction(self, node_config_with_llm, mock_llm_client):
        """Test action generation for orbit correction."""
        # Use a change within rate limit (0.0002 for HCOR1_KICK)
        mock_llm_client.generate.return_value = '''```json
[{"parameter_name": "HCOR1_KICK", "current_value": 0.0, "proposed_value": 0.0001, "rationale": "Correct orbit", "expected_impact": "Center beam", "priority": 1}]
```'''

        state = create_initial_state("Correct orbit")
        state["current_parameters"] = {
            "QF1_K1": 2.0,
            "HCOR1_KICK": 0.0,
            "VCOR1_KICK": 0.0,
        }
        state["analysis"]["issues"] = ["BPM1_X: Large deviation"]

        result = generate_actions_node(state, node_config_with_llm)

        assert "proposed_actions" in result
        assert len(result["proposed_actions"]) > 0

    def test_proposed_action_structure(self, node_config_with_llm, mock_llm_client):
        """Test structure of proposed actions."""
        mock_llm_client.generate.return_value = '''```json
[{"parameter_name": "QF1_K1", "current_value": 2.0, "proposed_value": 2.5, "rationale": "Test rationale", "expected_impact": "Test impact", "priority": 1}]
```'''

        state = create_initial_state("Test")
        state["current_parameters"] = {"QF1_K1": 2.0}

        result = generate_actions_node(state, node_config_with_llm)

        assert result["proposed_actions"]
        action = result["proposed_actions"][0]
        assert "parameter_name" in action
        assert "current_value" in action
        assert "proposed_value" in action
        assert "rationale" in action
        assert "expected_impact" in action
        assert "priority" in action


class TestHumanApprovalNode:
    """Tests for human_approval_node."""

    def test_approval_with_actions(self):
        """Test approval node with proposed actions."""
        state = create_initial_state("Test")
        state["proposed_actions"] = [
            {
                "parameter_name": "QF1_K1",
                "current_value": 2.0,
                "proposed_value": 2.5,
                "rationale": "Test",
                "expected_impact": "Test",
                "priority": 1,
            }
        ]

        result = human_approval_node(state)

        assert "workflow" in result
        assert result["workflow"]["awaiting_approval"] is True
        assert result["workflow"]["approval_status"] == "pending"

    def test_approval_without_actions(self):
        """Test approval node with no actions."""
        state = create_initial_state("Test")
        state["proposed_actions"] = []

        result = human_approval_node(state)

        assert "workflow" in result
        assert result["workflow"]["awaiting_approval"] is False
        assert result["workflow"]["approval_status"] == "no_actions"


class TestExecuteActionNode:
    """Tests for execute_action_node."""

    def test_execute_action_success(self, mock_backend, node_config):
        """Test successful action execution."""
        # Use a change within rate limit (0.5 for QF1_K1)
        action: ProposedAction = {
            "parameter_name": "QF1_K1",
            "current_value": 2.0,
            "proposed_value": 2.3,  # Change of 0.3, within rate limit of 0.5
            "rationale": "Test",
            "expected_impact": "Test",
            "priority": 1,
        }

        state = create_initial_state("Test")
        state["proposed_actions"] = [action]
        state["action_index"] = 0
        state["execution_history"] = []

        result = execute_action_node(state, node_config)

        assert "current_execution_result" in result
        assert result["current_execution_result"].success
        assert "execution_history" in result
        assert len(result["execution_history"]) == 1
        assert "current_diagnostics" in result

    def test_execute_action_safety_violation(self, mock_backend, node_config):
        """Test execution blocked by safety validation (out of limits)."""
        action: ProposedAction = {
            "parameter_name": "QF1_K1",
            "current_value": 0.0,
            "proposed_value": 100.0,  # Outside limits (max is 5.0)
            "rationale": "Test",
            "expected_impact": "Test",
            "priority": 1,
        }

        state = create_initial_state("Test")
        state["proposed_actions"] = [action]
        state["action_index"] = 0

        result = execute_action_node(state, node_config)

        # Safety check should block execution
        assert "error" in result
        assert result["error"]["type"] == "safety_violation"
        assert "safety_violations" in result

    def test_execute_action_rate_limit_violation(self, mock_backend, node_config):
        """Test execution blocked by rate limit violation."""
        action: ProposedAction = {
            "parameter_name": "QF1_K1",
            "current_value": 0.0,
            "proposed_value": 2.0,  # Change of 2.0, exceeds rate limit of 0.5
            "rationale": "Test",
            "expected_impact": "Test",
            "priority": 1,
        }

        state = create_initial_state("Test")
        state["proposed_actions"] = [action]
        state["action_index"] = 0

        result = execute_action_node(state, node_config)

        # Safety check should block execution
        assert "error" in result
        assert result["error"]["type"] == "safety_violation"
        assert "rate limit" in result["error"]["message"].lower()

    def test_execute_action_no_actions(self, mock_backend, node_config):
        """Test execution with no actions."""
        state = create_initial_state("Test")
        state["proposed_actions"] = []

        result = execute_action_node(state, node_config)

        assert "error" in result
        assert result["error"]["type"] == "invalid_action_index"

    def test_execute_action_backend_not_connected(self, mock_backend, node_config):
        """Test execution when backend not connected."""
        mock_backend.shutdown()

        # Use a change within rate limit
        action: ProposedAction = {
            "parameter_name": "QF1_K1",
            "current_value": 2.0,
            "proposed_value": 2.3,  # Within rate limit
            "rationale": "Test",
            "expected_impact": "Test",
            "priority": 1,
        }

        state = create_initial_state("Test")
        state["proposed_actions"] = [action]

        result = execute_action_node(state, node_config)

        assert "error" in result


class TestVerifyResultsNode:
    """Tests for verify_results_node."""

    def test_verify_no_result(self):
        """Test verification with no execution result."""
        state = create_initial_state("Test")
        result = verify_results_node(state)

        assert "verification_result" in result

    def test_verify_failed_execution(self):
        """Test verification of failed execution."""
        from accops_agent.accelerator_interface import ActionResult

        state = create_initial_state("Test")
        state["current_execution_result"] = ActionResult(
            success=False,
            action_type="set_parameter",
            parameters={},
            message="Failed",
            error="Test error",
        )

        result = verify_results_node(state)

        assert "verification_result" in result
        assert "failed" in result["verification_result"].lower()

    def test_verify_requires_llm_client(self, mock_backend):
        """Test that verification requires LLM client when history exists."""
        from accops_agent.accelerator_interface import ActionResult

        state = create_initial_state("Test")
        state["current_execution_result"] = ActionResult(
            success=True,
            action_type="set_parameter",
            parameters={},
            message="Success",
        )

        diags = mock_backend.read_all_diagnostics()
        state["execution_history"] = [
            {
                "action": {
                    "parameter_name": "QF1_K1",
                    "current_value": 0.0,
                    "proposed_value": 2.5,
                    "rationale": "Test",
                    "expected_impact": "Test",
                    "priority": 1,
                },
                "result": state["current_execution_result"],
                "diagnostics_before": diags,
                "diagnostics_after": diags,
                "timestamp": "2024-01-01T00:00:00",
            }
        ]

        with pytest.raises(GraphExecutionError):
            verify_results_node(state)

    def test_verify_successful_execution(self, mock_backend, node_config_with_llm, mock_llm_client):
        """Test verification of successful execution."""
        from accops_agent.accelerator_interface import ActionResult

        mock_llm_client.generate.return_value = "Assessment: EFFECTIVE\nEffectiveness: 8/10\nRecommendation: CONTINUE"

        state = create_initial_state("Test")
        state["current_execution_result"] = ActionResult(
            success=True,
            action_type="set_parameter",
            parameters={},
            message="Success",
        )

        # Add execution history
        diags = mock_backend.read_all_diagnostics()
        state["execution_history"] = [
            {
                "action": {
                    "parameter_name": "QF1_K1",
                    "current_value": 0.0,
                    "proposed_value": 2.5,
                    "rationale": "Test",
                    "expected_impact": "Test",
                    "priority": 1,
                },
                "result": state["current_execution_result"],
                "diagnostics_before": diags,
                "diagnostics_after": diags,
                "timestamp": "2024-01-01T00:00:00",
            }
        ]

        result = verify_results_node(state, node_config_with_llm)

        assert "verification_result" in result


class TestDecideContinuationNode:
    """Tests for decide_continuation_node."""

    def test_decide_max_iterations_reached(self):
        """Test decision when max iterations reached."""
        state = create_initial_state("Test")
        state["workflow"]["iteration_count"] = 9
        state["workflow"]["max_iterations"] = 10

        result = decide_continuation_node(state)

        assert "workflow" in result
        assert result["workflow"]["continue_optimization"] is False
        assert result["workflow"]["iteration_count"] == 10

    def test_decide_no_alarms(self, mock_backend):
        """Test decision when no alarms present."""
        state = create_initial_state("Test")
        state["current_diagnostics"] = mock_backend.read_all_diagnostics()
        state["workflow"]["iteration_count"] = 0
        state["workflow"]["max_iterations"] = 10

        result = decide_continuation_node(state)

        assert "workflow" in result
        assert result["workflow"]["goal_achieved"] is True
        assert result["workflow"]["continue_optimization"] is False

    def test_decide_continue_with_issues(self, mock_backend):
        """Test decision to continue when issues remain."""
        mock_backend.set_mock_diagnostic("BPM1_X", 5.0)

        state = create_initial_state("Test")
        state["current_diagnostics"] = mock_backend.read_all_diagnostics()
        state["workflow"]["iteration_count"] = 0
        state["workflow"]["max_iterations"] = 10

        result = decide_continuation_node(state)

        assert "workflow" in result
        assert result["workflow"]["continue_optimization"] is True
        assert result["workflow"]["goal_achieved"] is False
