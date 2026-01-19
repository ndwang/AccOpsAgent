"""Node implementations for the AccOps agent graph."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from ..diagnostic_control import AcceleratorBackend
from ..llm import LLMClient
from ..llm.parsers import parse_actions_from_llm, parse_issues_from_text, parse_verification_result
from ..llm.prompts import (
    DIAGNOSTIC_INTERPRETATION_SYSTEM,
    REASONING_PLANNING_SYSTEM,
    ACTION_GENERATION_SYSTEM,
    VERIFICATION_SYSTEM,
    create_diagnostic_interpretation_prompt,
    create_reasoning_planning_prompt,
    create_action_generation_prompt,
    create_verification_prompt,
)
from ..safety import ConstraintChecker
from ..utils.exceptions import ConstraintViolationError, DiagnosticReadError, GraphExecutionError
from .state import AgentState, ExecutionHistoryEntry, ProposedAction

logger = logging.getLogger(__name__)


def ingest_diagnostics_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
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


def interpret_diagnostics_node(state: AgentState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Interpret diagnostic measurements using LLM.

    This node analyzes the diagnostic data and identifies any issues
    or anomalies that need attention.

    Args:
        state: Current agent state with diagnostics
        config: Config with LLM client (required)

    Returns:
        Updated state with diagnostic_interpretation and identified_issues

    Raises:
        GraphExecutionError: If LLM client not available or generation fails
    """
    logger.info("Interpreting diagnostics with LLM")

    diagnostics = state.get("current_diagnostics", [])
    machine_status_summary = state.get("machine_status_summary", "Unknown")

    if not diagnostics:
        return {
            "diagnostic_interpretation": "No diagnostic data available",
            "identified_issues": ["No diagnostics available"],
        }

    # Get LLM client from config
    llm_client: Optional[LLMClient] = None
    if config and "configurable" in config:
        llm_client = config["configurable"].get("llm_client")

    if not llm_client:
        raise GraphExecutionError("LLM client not available in config")

    # Generate LLM prompt
    prompt = create_diagnostic_interpretation_prompt(diagnostics, machine_status_summary)

    # Get LLM interpretation
    interpretation = llm_client.generate(
        prompt=prompt,
        system_prompt=DIAGNOSTIC_INTERPRETATION_SYSTEM,
        temperature=0.3,  # Lower temperature for diagnostic analysis
    )

    # Parse issues from interpretation
    issues = parse_issues_from_text(interpretation)

    logger.info(f"LLM identified {len(issues)} issue(s)")

    return {
        "diagnostic_interpretation": interpretation,
        "identified_issues": issues,
    }


def reasoning_planning_node(state: AgentState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Generate strategy and reasoning to achieve user intent.

    This node uses LLM to create a high-level strategy for addressing
    the user's goal based on current machine state and identified issues.

    Args:
        state: Current agent state
        config: Config with LLM client (required)

    Returns:
        Updated state with strategy and reasoning

    Raises:
        GraphExecutionError: If LLM client not available or generation fails
    """
    logger.info("Generating strategy and reasoning with LLM")

    user_intent = state.get("user_intent", "")
    diagnostic_interpretation = state.get("diagnostic_interpretation", "")
    issues = state.get("identified_issues", [])
    current_parameters = state.get("current_parameters", {})

    if not user_intent:
        return {
            "strategy": "No user intent specified",
            "reasoning": "Cannot plan without user goal",
        }

    # Get LLM client from config
    llm_client: Optional[LLMClient] = None
    if config and "configurable" in config:
        llm_client = config["configurable"].get("llm_client")

    if not llm_client:
        raise GraphExecutionError("LLM client not available in config")

    # Generate LLM prompt
    prompt = create_reasoning_planning_prompt(
        user_intent, diagnostic_interpretation, issues, current_parameters
    )

    # Get LLM reasoning
    llm_output = llm_client.generate(
        prompt=prompt,
        system_prompt=REASONING_PLANNING_SYSTEM,
        temperature=0.5,
    )

    # Extract strategy and reasoning from output
    strategy = "See full reasoning"
    reasoning = llm_output

    # Try to extract strategy section
    if "Strategy:" in llm_output or "**Strategy**" in llm_output:
        import re
        strategy_match = re.search(
            r"\*\*Strategy\*\*:?\s*(.+?)(?:\n\n|\*\*)", llm_output, re.DOTALL
        )
        if strategy_match:
            strategy = strategy_match.group(1).strip()

    logger.info("Generated strategy with LLM")

    return {
        "strategy": strategy,
        "reasoning": reasoning,
    }


def generate_actions_node(state: AgentState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Generate specific parameter adjustment actions using LLM.

    This node creates concrete actions (parameter changes) based on
    the strategy and current machine state. Actions are validated
    against safety constraints before being proposed.

    Args:
        state: Current agent state
        config: Config with LLM client and backend (required)

    Returns:
        Updated state with proposed_actions and any safety_violations

    Raises:
        GraphExecutionError: If LLM client or backend not available, or if generation fails
    """
    logger.info("Generating actions with LLM")

    user_intent = state.get("user_intent", "")
    strategy = state.get("strategy", "")
    reasoning = state.get("reasoning", "")
    current_params = state.get("current_parameters", {})
    current_diagnostics = state.get("current_diagnostics", [])

    # Get LLM client and backend from config
    llm_client: Optional[LLMClient] = None
    backend: Optional[AcceleratorBackend] = None
    if config and "configurable" in config:
        llm_client = config["configurable"].get("llm_client")
        backend = config["configurable"].get("backend")

    if not llm_client:
        raise GraphExecutionError("LLM client not available in config")

    if not backend:
        raise GraphExecutionError("Backend not available in config")

    # Build parameter metadata from backend config (include rate limits for LLM)
    parameter_metadata = {}
    for knob in backend.config.knobs:
        parameter_metadata[knob.name] = {
            "min": knob.min_value,
            "max": knob.max_value,
            "rate_limit": knob.rate_limit,
            "type": knob.element_type,
            "unit": knob.unit,
        }

    # Generate LLM prompt
    prompt = create_action_generation_prompt(
        user_intent, strategy, reasoning, current_params, parameter_metadata
    )

    # Get LLM actions
    llm_output = llm_client.generate(
        prompt=prompt,
        system_prompt=ACTION_GENERATION_SYSTEM,
        temperature=0.3,  # Lower temperature for parameter choices
    )

    # Parse actions from output
    actions = parse_actions_from_llm(llm_output)

    if not actions:
        raise GraphExecutionError("LLM produced no valid actions")

    logger.info(f"Generated {len(actions)} action(s) with LLM")

    # Validate actions against safety constraints (strict mode)
    checker = ConstraintChecker(backend.config)
    checker.update_diagnostics(current_diagnostics)

    validation_result = checker.validate_actions(actions)
    safety_violations: List[str] = []

    if not validation_result.is_valid:
        # Collect all violation messages
        for violation in validation_result.all_violations:
            safety_violations.append(str(violation))

        # Filter out invalid actions (keep only valid ones)
        valid_actions: List[ProposedAction] = []
        for i, result in enumerate(validation_result.results):
            if result.is_valid:
                valid_actions.append(actions[i])
            else:
                logger.warning(f"Action {i} rejected: {result.error_messages}")

        # Check if any global violations would block all actions
        if validation_result.global_violations:
            for violation in validation_result.global_violations:
                logger.warning(f"Global safety violation: {violation}")

            # If there are global violations (like interlocks), reject all actions
            if any(v.severity.value == "error" for v in validation_result.global_violations):
                logger.error("Global safety constraint violated - all actions rejected")
                return {
                    "proposed_actions": [],
                    "action_index": 0,
                    "safety_violations": safety_violations,
                    "error": "Safety constraints violated - actions rejected",
                    "error_type": "safety_violation",
                }

        actions = valid_actions

    if not actions:
        logger.error("No actions passed safety validation")
        return {
            "proposed_actions": [],
            "action_index": 0,
            "safety_violations": safety_violations,
            "error": "All proposed actions failed safety validation",
            "error_type": "safety_violation",
        }

    logger.info(f"{len(actions)} action(s) passed safety validation")

    return {
        "proposed_actions": actions,
        "action_index": 0,
        "safety_violations": safety_violations,
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


def execute_action_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute approved parameter changes.

    This node executes the approved action(s) via the backend and
    records the results. A final safety check is performed before
    execution to catch any state changes since validation.

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

        # Final safety check before execution (strict mode)
        checker = ConstraintChecker(backend.config)
        checker.update_diagnostics(diagnostics_before)

        validation_result = checker.validate_action(action)
        if not validation_result.is_valid:
            error_msgs = "; ".join(validation_result.error_messages)
            logger.error(f"Final safety check failed: {error_msgs}")
            return {
                "error": f"Safety check failed before execution: {error_msgs}",
                "error_type": "safety_violation",
                "safety_violations": validation_result.error_messages,
            }

        # Execute the parameter change
        result = backend.set_parameter(
            action["parameter_name"], action["proposed_value"]
        )

        if result.success:
            # Record execution for global rate limiting
            checker.record_execution()

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

    except ConstraintViolationError as e:
        logger.error(f"Safety constraint violation: {e}")
        return {
            "error": str(e),
            "error_type": "safety_violation",
        }
    except Exception as e:
        logger.error(f"Exception during action execution: {e}")
        return {
            "error": str(e),
            "error_type": "execution_exception",
        }


def verify_results_node(state: AgentState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Verify results of action execution using LLM.

    This node analyzes whether the executed action had the desired
    effect on diagnostics.

    Args:
        state: Current agent state
        config: Config with LLM client (required)

    Returns:
        Updated state with verification_result

    Raises:
        GraphExecutionError: If LLM client not available or generation fails
    """
    logger.info("Verifying results with LLM")

    execution_result = state.get("current_execution_result")

    if not execution_result:
        return {
            "verification_result": "No execution result to verify",
        }

    if not execution_result.success:
        return {
            "verification_result": f"Action failed: {execution_result.error}",
        }

    execution_history = state.get("execution_history", [])
    if not execution_history:
        return {
            "verification_result": "Action executed successfully (no history available)",
        }

    latest = execution_history[-1]
    diags_before = latest["diagnostics_before"]
    diags_after = latest["diagnostics_after"]
    action = latest["action"]
    user_intent = state.get("user_intent", "")

    # Get LLM client from config
    llm_client: Optional[LLMClient] = None
    if config and "configurable" in config:
        llm_client = config["configurable"].get("llm_client")

    if not llm_client:
        raise GraphExecutionError("LLM client not available in config")

    # Create action description
    action_description = (
        f"Set {action['parameter_name']} from {action['current_value']:.4f} "
        f"to {action['proposed_value']:.4f}. "
        f"Rationale: {action.get('rationale', 'N/A')}"
    )

    # Generate LLM prompt
    prompt = create_verification_prompt(
        action_description, diags_before, diags_after, user_intent
    )

    # Get LLM verification
    llm_output = llm_client.generate(
        prompt=prompt,
        system_prompt=VERIFICATION_SYSTEM,
        temperature=0.3,
    )

    # Parse verification result
    verification_data = parse_verification_result(llm_output)

    # Store detailed verification
    verification = (
        f"{verification_data['assessment']} (effectiveness: {verification_data['effectiveness']}/10) - "
        f"Recommendation: {verification_data['recommendation']}\n\n{llm_output[:200]}"
    )

    logger.info(f"LLM verification: {verification_data['assessment']}")

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
