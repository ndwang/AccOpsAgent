"""Graph assembly and conditional edges for the AccOps agent."""

import logging
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    decide_continuation_node,
    execute_action_node,
    generate_actions_node,
    human_approval_node,
    ingest_diagnostics_node,
    interpret_diagnostics_node,
    reasoning_planning_node,
    verify_results_node,
)
from .state import AgentState

logger = logging.getLogger(__name__)

# Node names as constants
INGEST_DIAGNOSTICS = "ingest_diagnostics"
INTERPRET_DIAGNOSTICS = "interpret_diagnostics"
REASONING_PLANNING = "reasoning_planning"
GENERATE_ACTIONS = "generate_actions"
HUMAN_APPROVAL = "human_approval"
EXECUTE_ACTION = "execute_action"
VERIFY_RESULTS = "verify_results"
DECIDE_CONTINUATION = "decide_continuation"


def route_after_approval(state: AgentState) -> Literal["execute_action", "generate_actions", "__end__"]:
    """Route after human approval based on approval status.

    Args:
        state: Current agent state

    Returns:
        Next node name based on approval status
    """
    workflow = state.get("workflow", {})
    approval_status = workflow.get("approval_status", "")

    if approval_status == "approved":
        logger.info("Actions approved, proceeding to execution")
        return EXECUTE_ACTION
    elif approval_status == "modified":
        logger.info("Actions modified, regenerating with feedback")
        return GENERATE_ACTIONS
    else:
        # rejected or no_actions
        logger.info(f"Actions {approval_status}, ending workflow")
        return END


def route_after_continuation(state: AgentState) -> Literal["ingest_diagnostics", "__end__"]:
    """Route after continuation decision.

    Args:
        state: Current agent state

    Returns:
        Next node name based on continuation decision
    """
    workflow = state.get("workflow", {})
    continue_opt = workflow.get("continue_optimization", False)
    goal_achieved = workflow.get("goal_achieved", False)

    if goal_achieved:
        logger.info("Goal achieved, ending workflow")
        return END
    elif continue_opt:
        logger.info("Continuing optimization, re-ingesting diagnostics")
        return INGEST_DIAGNOSTICS
    else:
        logger.info("Stopping optimization")
        return END


def route_after_ingest(state: AgentState) -> Literal["interpret_diagnostics", "__end__"]:
    """Route after diagnostic ingestion.

    Args:
        state: Current agent state

    Returns:
        Next node name, or END if error occurred
    """
    error = state.get("error") or {}
    if isinstance(error, dict) and error.get("message"):
        logger.error(f"Error during ingestion: {error.get('message')}")
        return END

    return INTERPRET_DIAGNOSTICS


def build_graph() -> StateGraph:
    """Build the AccOps agent state graph.

    Creates a StateGraph with all nodes and edges configured.
    The graph implements a human-in-the-loop optimization workflow.

    Returns:
        Configured StateGraph (not compiled)
    """
    logger.info("Building AccOps agent graph")

    # Create state graph
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node(INGEST_DIAGNOSTICS, ingest_diagnostics_node)
    graph.add_node(INTERPRET_DIAGNOSTICS, interpret_diagnostics_node)
    graph.add_node(REASONING_PLANNING, reasoning_planning_node)
    graph.add_node(GENERATE_ACTIONS, generate_actions_node)
    graph.add_node(HUMAN_APPROVAL, human_approval_node)
    graph.add_node(EXECUTE_ACTION, execute_action_node)
    graph.add_node(VERIFY_RESULTS, verify_results_node)
    graph.add_node(DECIDE_CONTINUATION, decide_continuation_node)

    # Set entry point
    graph.add_edge(START, INGEST_DIAGNOSTICS)

    # Linear edges for the main flow
    graph.add_conditional_edges(
        INGEST_DIAGNOSTICS,
        route_after_ingest,
        {
            INTERPRET_DIAGNOSTICS: INTERPRET_DIAGNOSTICS,
            END: END,
        },
    )
    graph.add_edge(INTERPRET_DIAGNOSTICS, REASONING_PLANNING)
    graph.add_edge(REASONING_PLANNING, GENERATE_ACTIONS)
    graph.add_edge(GENERATE_ACTIONS, HUMAN_APPROVAL)

    # Conditional edge after human approval
    graph.add_conditional_edges(
        HUMAN_APPROVAL,
        route_after_approval,
        {
            EXECUTE_ACTION: EXECUTE_ACTION,
            GENERATE_ACTIONS: GENERATE_ACTIONS,
            END: END,
        },
    )

    # Execution flow
    graph.add_edge(EXECUTE_ACTION, VERIFY_RESULTS)
    graph.add_edge(VERIFY_RESULTS, DECIDE_CONTINUATION)

    # Conditional edge for continuation
    graph.add_conditional_edges(
        DECIDE_CONTINUATION,
        route_after_continuation,
        {
            INGEST_DIAGNOSTICS: INGEST_DIAGNOSTICS,
            END: END,
        },
    )

    logger.info("Graph built successfully")
    return graph


def compile_graph(
    graph: StateGraph = None,
    interrupt_before: list[str] = None,
    interrupt_after: list[str] = None,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    """Compile the graph with optional interrupt points.

    Args:
        graph: StateGraph to compile (builds new one if not provided)
        interrupt_before: Node names to interrupt before
        interrupt_after: Node names to interrupt after
        checkpointer: Checkpointer for state persistence

    Returns:
        Compiled graph ready for execution
    """
    if graph is None:
        graph = build_graph()

    # Default: interrupt before human approval for human-in-the-loop
    if interrupt_before is None:
        interrupt_before = [HUMAN_APPROVAL]

    logger.info(f"Compiling graph with interrupt_before={interrupt_before}")

    compiled = graph.compile(
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after or [],
        checkpointer=checkpointer,
    )

    return compiled


def create_agent_config(
    backend: Any,
    llm_client: Any = None,
) -> RunnableConfig:
    """Create configuration dict for graph execution.

    Args:
        backend: AcceleratorBackend instance
        llm_client: LLMClient instance (optional)

    Returns:
        Config dict for graph.invoke() or graph.stream()
    """
    config: RunnableConfig = {
        "configurable": {
            "backend": backend,
        }
    }

    if llm_client:
        config["configurable"]["llm_client"] = llm_client

    return config
