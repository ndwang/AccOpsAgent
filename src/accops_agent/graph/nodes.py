"""Node implementations for the AccOps agent graph."""

import logging
from datetime import datetime
from typing import Any, Dict

from ..diagnostic_control import AcceleratorBackend
from ..utils.exceptions import DiagnosticReadError, GraphExecutionError
from .state import AgentState, ExecutionHistoryEntry

logger = logging.getLogger(__name__)


def ingest_diagnostics_node(state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Read all diagnostics from the accelerator backend.

    This node reads current diagnostic measurements and parameter values
    from the backend and updates the state.

    Args:
        state: Current agent state
        config: Configuration including backend instance

    Returns:
        Updated state with current_diagnostics and current_parameters
    """
    logger.info("Ingesting diagnostics from backend")

    try:
        # Get backend from config
        backend: AcceleratorBackend = config["configurable"]["backend"]

        if not backend.is_connected():
            raise DiagnosticReadError("Backend is not connected")

        # Read all diagnostics
        diagnostics = backend.read_all_diagnostics()
        logger.info(f"Read {len(diagnostics)} diagnostics")

        # Read all parameters
        parameters = backend.get_all_parameters()
        param_dict = {name: pval.value for name, pval in parameters.items()}
        logger.info(f"Read {len(param_dict)} parameters")

        # Generate machine status summary
        alarm_count = sum(1 for d in diagnostics if d.is_alarm())
        warning_count = sum(1 for d in diagnostics if not d.is_healthy())

        if alarm_count > 0:
            status_summary = f"ALARM: {alarm_count} diagnostic(s) in alarm state"
        elif warning_count > 0:
            status_summary = f"WARNING: {warning_count} diagnostic(s) outside tolerance"
        else:
            status_summary = "NORMAL: All diagnostics within tolerance"

        return {
            "current_diagnostics": diagnostics,
            "current_parameters": param_dict,
            "machine_status_summary": status_summary,
        }

    except Exception as e:
        logger.error(f"Failed to ingest diagnostics: {e}")
        return {
            "error": str(e),
            "error_type": "diagnostic_read_error",
        }


def interpret_diagnostics_node(state: AgentState) -> Dict[str, Any]:
    """Interpret diagnostic measurements using LLM.

    This node analyzes the diagnostic data and identifies any issues
    or anomalies that need attention.

    Args:
        state: Current agent state with diagnostics

    Returns:
        Updated state with diagnostic_interpretation and identified_issues
    """
    logger.info("Interpreting diagnostics")

    # TODO: This will use LLM in Phase 5
    # For now, provide rule-based interpretation

    diagnostics = state.get("current_diagnostics", [])

    if not diagnostics:
        return {
            "diagnostic_interpretation": "No diagnostic data available",
            "identified_issues": ["No diagnostics available"],
        }

    issues = []
    for diag in diagnostics:
        if diag.is_alarm():
            issues.append(
                f"{diag.diagnostic_name}: ALARM - {diag.message or 'Exceeds alarm threshold'}"
            )
        elif not diag.is_healthy():
            issues.append(
                f"{diag.diagnostic_name}: WARNING - {diag.message or 'Outside tolerance'}"
            )

    if issues:
        interpretation = (
            f"Identified {len(issues)} issue(s) requiring attention. "
            f"Machine status: {state.get('machine_status_summary', 'Unknown')}"
        )
    else:
        interpretation = (
            "All diagnostics are within normal operating ranges. "
            "Machine is operating nominally."
        )

    return {
        "diagnostic_interpretation": interpretation,
        "identified_issues": issues,
    }


def reasoning_planning_node(state: AgentState) -> Dict[str, Any]:
    """Generate strategy and reasoning to achieve user intent.

    This node uses LLM to create a high-level strategy for addressing
    the user's goal based on current machine state and identified issues.

    Args:
        state: Current agent state

    Returns:
        Updated state with strategy and reasoning
    """
    logger.info("Generating strategy and reasoning")

    # TODO: This will use LLM in Phase 5
    # For now, provide rule-based reasoning

    user_intent = state.get("user_intent", "")
    issues = state.get("identified_issues", [])

    if not user_intent:
        return {
            "strategy": "No user intent specified",
            "reasoning": "Cannot plan without user goal",
        }

    # Simple rule-based strategy
    if issues:
        strategy = f"Address {len(issues)} identified issue(s) to achieve goal: {user_intent}"
        reasoning = (
            f"The machine has issues that need correction: {', '.join(issues[:3])}. "
            f"Will propose parameter adjustments to resolve these issues and achieve: {user_intent}"
        )
    else:
        strategy = f"Optimize machine parameters to achieve: {user_intent}"
        reasoning = (
            f"Machine is operating nominally. Will fine-tune parameters to "
            f"optimize for user goal: {user_intent}"
        )

    return {
        "strategy": strategy,
        "reasoning": reasoning,
    }


def generate_actions_node(state: AgentState) -> Dict[str, Any]:
    """Generate specific parameter adjustment actions.

    This node creates concrete actions (parameter changes) based on
    the strategy and current machine state.

    Args:
        state: Current agent state

    Returns:
        Updated state with proposed_actions
    """
    logger.info("Generating actions")

    # TODO: This will use LLM in Phase 5
    # For now, generate simple test actions

    user_intent = state.get("user_intent", "").lower()
    issues = state.get("identified_issues", [])
    current_params = state.get("current_parameters", {})

    actions = []

    # Simple rule-based action generation
    if "optimize" in user_intent and "beam size" in user_intent:
        # Example: optimize beam size by adjusting quadrupoles
        for param_name, current_value in current_params.items():
            if "QF" in param_name or "QD" in param_name:
                # Propose small adjustment
                proposed_value = current_value + 0.1
                actions.append({
                    "parameter_name": param_name,
                    "current_value": current_value,
                    "proposed_value": proposed_value,
                    "rationale": f"Adjust {param_name} to optimize beam size",
                    "expected_impact": "Reduce beam size at focal point",
                    "priority": 1,
                })
                break  # Only propose one action for now

    elif "correct orbit" in user_intent or any("BPM" in issue for issue in issues):
        # Correct orbit using correctors
        for param_name, current_value in current_params.items():
            if "COR" in param_name or "KICK" in param_name:
                proposed_value = current_value + 0.05
                actions.append({
                    "parameter_name": param_name,
                    "current_value": current_value,
                    "proposed_value": proposed_value,
                    "rationale": f"Adjust {param_name} to correct orbit",
                    "expected_impact": "Reduce orbit deviation at BPMs",
                    "priority": 1,
                })
                break

    # If no specific actions, propose a generic one
    if not actions and current_params:
        first_param = list(current_params.keys())[0]
        current_value = current_params[first_param]
        actions.append({
            "parameter_name": first_param,
            "current_value": current_value,
            "proposed_value": current_value + 0.1,
            "rationale": f"Test adjustment of {first_param}",
            "expected_impact": "Observe response to parameter change",
            "priority": 2,
        })

    logger.info(f"Generated {len(actions)} action(s)")

    return {
        "proposed_actions": actions,
        "action_index": 0,
    }


def human_approval_node(state: AgentState) -> Dict[str, Any]:
    """Present actions to human for approval.

    This node sets the state to await human approval. The graph will
    interrupt here until human provides approval/rejection/modification.

    Args:
        state: Current agent state with proposed_actions

    Returns:
        Updated state with awaiting_approval flag
    """
    logger.info("Requesting human approval")

    proposed_actions = state.get("proposed_actions", [])

    if not proposed_actions:
        logger.warning("No actions to approve")
        return {
            "awaiting_approval": False,
            "approval_status": "no_actions",
        }

    logger.info(f"Awaiting approval for {len(proposed_actions)} action(s)")

    return {
        "awaiting_approval": True,
        "approval_status": "pending",
    }


def execute_action_node(state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute approved parameter changes.

    This node executes the approved action(s) via the backend and
    records the results.

    Args:
        state: Current agent state
        config: Configuration including backend instance

    Returns:
        Updated state with execution results
    """
    logger.info("Executing action")

    try:
        backend: AcceleratorBackend = config["configurable"]["backend"]

        if not backend.is_connected():
            raise GraphExecutionError("Backend is not connected")

        proposed_actions = state.get("proposed_actions", [])
        action_index = state.get("action_index", 0)

        if not proposed_actions or action_index >= len(proposed_actions):
            return {
                "error": "No action to execute",
                "error_type": "invalid_action_index",
            }

        action = proposed_actions[action_index]

        # Read diagnostics before execution
        diagnostics_before = backend.read_all_diagnostics()

        # Execute the parameter change
        result = backend.set_parameter(
            action["parameter_name"], action["proposed_value"]
        )

        if result.success:
            # Run calculation to propagate changes
            backend.run_calculation()

            # Read diagnostics after execution
            diagnostics_after = backend.read_all_diagnostics()

            # Record execution history
            execution_entry: ExecutionHistoryEntry = {
                "action": action,
                "result": result,
                "diagnostics_before": diagnostics_before,
                "diagnostics_after": diagnostics_after,
                "timestamp": datetime.now().isoformat(),
            }

            execution_history = state.get("execution_history", [])
            execution_history.append(execution_entry)

            logger.info(f"Successfully executed action: {action['parameter_name']}")

            return {
                "current_execution_result": result,
                "execution_history": execution_history,
                "current_diagnostics": diagnostics_after,
            }
        else:
            logger.error(f"Action execution failed: {result.error}")
            return {
                "current_execution_result": result,
                "error": result.error,
                "error_type": "execution_failed",
            }

    except Exception as e:
        logger.error(f"Exception during action execution: {e}")
        return {
            "error": str(e),
            "error_type": "execution_exception",
        }


def verify_results_node(state: AgentState) -> Dict[str, Any]:
    """Verify results of action execution.

    This node analyzes whether the executed action had the desired
    effect on diagnostics.

    Args:
        state: Current agent state

    Returns:
        Updated state with verification_result
    """
    logger.info("Verifying results")

    execution_result = state.get("current_execution_result")

    if not execution_result:
        return {
            "verification_result": "No execution result to verify",
        }

    if not execution_result.success:
        return {
            "verification_result": f"Action failed: {execution_result.error}",
        }

    # TODO: More sophisticated verification with LLM in Phase 5
    # For now, simple check

    execution_history = state.get("execution_history", [])
    if execution_history:
        latest = execution_history[-1]
        diags_before = latest["diagnostics_before"]
        diags_after = latest["diagnostics_after"]

        # Count alarms before and after
        alarms_before = sum(1 for d in diags_before if d.is_alarm())
        alarms_after = sum(1 for d in diags_after if d.is_alarm())

        if alarms_after < alarms_before:
            verification = "Positive result: Reduced alarm count"
        elif alarms_after > alarms_before:
            verification = "Negative result: Increased alarm count"
        else:
            verification = "Neutral result: No change in alarm count"
    else:
        verification = "Action executed successfully"

    return {
        "verification_result": verification,
    }


def decide_continuation_node(state: AgentState) -> Dict[str, Any]:
    """Decide whether to continue optimization or terminate.

    This node determines if the goal has been achieved and whether
    to continue iterating.

    Args:
        state: Current agent state

    Returns:
        Updated state with goal_achieved and continue_optimization flags
    """
    logger.info("Deciding continuation")

    iteration_count = state.get("iteration_count", 0) + 1
    max_iterations = state.get("max_iterations", 10)

    # Check if max iterations reached
    if iteration_count >= max_iterations:
        logger.info(f"Reached max iterations ({max_iterations})")
        return {
            "goal_achieved": False,
            "continue_optimization": False,
            "iteration_count": iteration_count,
        }

    # TODO: Use LLM to assess goal achievement in Phase 5
    # For now, simple rule-based decision

    current_diagnostics = state.get("current_diagnostics", [])
    alarm_count = sum(1 for d in current_diagnostics if d.is_alarm())

    if alarm_count == 0:
        logger.info("Goal achieved: No alarms present")
        return {
            "goal_achieved": True,
            "continue_optimization": False,
            "iteration_count": iteration_count,
        }

    # Continue if issues remain and under iteration limit
    logger.info(f"Continuing optimization (iteration {iteration_count}/{max_iterations})")
    return {
        "goal_achieved": False,
        "continue_optimization": True,
        "iteration_count": iteration_count,
    }
