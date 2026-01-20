"""Agent state definition for LangGraph."""

from typing import Dict, List, Optional, TypedDict

from ..accelerator_interface import ActionResult, DiagnosticSnapshot


class ProposedAction(TypedDict, total=False):
    """A proposed parameter adjustment action."""

    parameter_name: str
    current_value: float
    proposed_value: float
    rationale: str
    expected_impact: str
    priority: int


class ExecutionHistoryEntry(TypedDict, total=False):
    """Record of an executed action."""

    action: ProposedAction
    result: ActionResult
    diagnostics_before: List[DiagnosticSnapshot]
    diagnostics_after: List[DiagnosticSnapshot]
    timestamp: str


class AnalysisState(TypedDict, total=False):
    """LLM analysis and reasoning outputs."""

    interpretation: str  # LLM's interpretation of diagnostics
    issues: List[str]  # Identified problems
    strategy: str  # High-level strategy to achieve user intent
    reasoning: str  # Detailed reasoning and planning


class WorkflowState(TypedDict, total=False):
    """Human-in-the-loop workflow and iteration control."""

    awaiting_approval: bool
    approval_status: str  # 'approved', 'rejected', or 'modified'
    user_feedback: str
    goal_achieved: bool
    continue_optimization: bool
    iteration_count: int
    max_iterations: int


class ErrorState(TypedDict, total=False):
    """Error information."""

    message: Optional[str]
    type: Optional[str]


class AgentState(TypedDict, total=False):
    """State for the AccOps agent graph.

    Consolidated state with nested TypedDicts for cleaner LangSmith traces.
    """

    # User input
    user_intent: str

    # Current machine state
    current_diagnostics: List[DiagnosticSnapshot]
    current_parameters: Dict[str, float]
    machine_status_summary: str

    # LLM analysis (consolidated)
    analysis: AnalysisState

    # Proposed actions
    proposed_actions: List[ProposedAction]
    action_index: int

    # Execution
    execution_history: List[ExecutionHistoryEntry]
    current_execution_result: ActionResult
    verification_result: str

    # Workflow control (consolidated)
    workflow: WorkflowState

    # Error handling (consolidated)
    error: ErrorState

    # Safety validation
    safety_violations: List[str]


def create_initial_state(user_intent: str, max_iterations: int = 10) -> AgentState:
    """Create initial agent state.

    Args:
        user_intent: User's goal or request
        max_iterations: Maximum optimization iterations

    Returns:
        Initial AgentState
    """
    return AgentState(
        user_intent=user_intent,
        current_diagnostics=[],
        current_parameters={},
        machine_status_summary="",
        analysis=AnalysisState(
            interpretation="",
            issues=[],
            strategy="",
            reasoning="",
        ),
        proposed_actions=[],
        action_index=0,
        execution_history=[],
        verification_result="",
        workflow=WorkflowState(
            awaiting_approval=False,
            approval_status="",
            user_feedback="",
            goal_achieved=False,
            continue_optimization=True,
            iteration_count=0,
            max_iterations=max_iterations,
        ),
        error=ErrorState(message=None, type=None),
        safety_violations=[],
    )
