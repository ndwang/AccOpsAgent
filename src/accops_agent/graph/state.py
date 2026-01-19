"""Agent state definition for LangGraph."""

from typing import Any, Dict, List, Optional, TypedDict

from ..diagnostic_control import ActionResult, DiagnosticSnapshot


class ProposedAction(TypedDict, total=False):
    """A proposed parameter adjustment action.

    Attributes:
        parameter_name: Name of the parameter to adjust
        current_value: Current parameter value
        proposed_value: Proposed new value
        rationale: Explanation of why this change is proposed
        expected_impact: Expected impact on diagnostics
        priority: Priority level (1=highest)
    """

    parameter_name: str
    current_value: float
    proposed_value: float
    rationale: str
    expected_impact: str
    priority: int


class ExecutionHistoryEntry(TypedDict, total=False):
    """Record of an executed action.

    Attributes:
        action: The action that was executed
        result: Result of the action execution
        diagnostics_before: Diagnostics before execution
        diagnostics_after: Diagnostics after execution
        timestamp: When the action was executed
    """

    action: ProposedAction
    result: ActionResult
    diagnostics_before: List[DiagnosticSnapshot]
    diagnostics_after: List[DiagnosticSnapshot]
    timestamp: str


class AgentState(TypedDict, total=False):
    """State for the AccOps agent graph.

    This TypedDict defines all state that flows through the LangGraph.
    Fields are optional (total=False) to allow incremental state building.

    Attributes:
        # User input and configuration
        user_intent: User's goal or request
        backend_type: Type of backend ('mock', 'pytao', etc.)

        # Current machine state
        current_diagnostics: Latest diagnostic readings
        current_parameters: Latest parameter values
        machine_status_summary: Human-readable status summary

        # Agent reasoning
        diagnostic_interpretation: LLM's interpretation of diagnostics
        identified_issues: List of identified problems
        strategy: High-level strategy to achieve user intent
        reasoning: Detailed reasoning and planning

        # Proposed actions
        proposed_actions: List of actions to execute
        action_index: Index of current action being processed

        # Human approval workflow
        awaiting_approval: Whether waiting for human approval
        approval_status: 'approved', 'rejected', or 'modified'
        user_feedback: Feedback provided during approval

        # Execution
        execution_history: Record of all executed actions
        current_execution_result: Result of most recent execution

        # Verification and continuation
        verification_result: Assessment of action effectiveness
        goal_achieved: Whether the user's goal has been achieved
        continue_optimization: Whether to continue iterating
        iteration_count: Number of optimization iterations
        max_iterations: Maximum allowed iterations

        # Error handling
        error: Error message if something went wrong
        error_type: Type of error encountered

        # Additional context
        metadata: Additional metadata for tracking
    """

    # User input and configuration
    user_intent: str
    backend_type: str

    # Current machine state
    current_diagnostics: List[DiagnosticSnapshot]
    current_parameters: Dict[str, float]
    machine_status_summary: str

    # Agent reasoning
    diagnostic_interpretation: str
    identified_issues: List[str]
    strategy: str
    reasoning: str

    # Proposed actions
    proposed_actions: List[ProposedAction]
    action_index: int

    # Human approval workflow
    awaiting_approval: bool
    approval_status: str
    user_feedback: str

    # Execution
    execution_history: List[ExecutionHistoryEntry]
    current_execution_result: ActionResult

    # Verification and continuation
    verification_result: str
    goal_achieved: bool
    continue_optimization: bool
    iteration_count: int
    max_iterations: int

    # Error handling
    error: Optional[str]
    error_type: Optional[str]

    # Safety validation
    safety_violations: List[str]

    # Additional context
    metadata: Dict[str, Any]


def create_initial_state(user_intent: str, backend_type: str = "mock") -> AgentState:
    """Create initial agent state.

    Args:
        user_intent: User's goal or request
        backend_type: Type of backend to use

    Returns:
        Initial AgentState
    """
    return AgentState(
        user_intent=user_intent,
        backend_type=backend_type,
        current_diagnostics=[],
        current_parameters={},
        machine_status_summary="",
        diagnostic_interpretation="",
        identified_issues=[],
        strategy="",
        reasoning="",
        proposed_actions=[],
        action_index=0,
        awaiting_approval=False,
        approval_status="",
        user_feedback="",
        execution_history=[],
        verification_result="",
        goal_achieved=False,
        continue_optimization=True,
        iteration_count=0,
        max_iterations=10,
        error=None,
        error_type=None,
        safety_violations=[],
        metadata={},
    )
