#!/usr/bin/env python
"""CLI entry point for the AccOps Agent.

This script provides a human-in-the-loop interface for running
the accelerator optimization agent using the MCP backend.

Usage:
    uv run python scripts/run_agent.py \\
        --config configs/accelerators/example_linac.yaml \\
        --intent "Optimize horizontal beam size"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph.checkpoint.memory import MemorySaver

from accops_agent.cli import (
    ApprovalResponse,
    format_diagnostics_table,
    format_execution_result,
    format_state_summary,
    format_verification_result,
    get_user_approval,
    get_user_intent,
    print_banner,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from accops_agent.cli.input_handler import ApprovalStatus
from accops_agent.accelerator_interface import MCPBackend
from accops_agent.graph import (
    HUMAN_APPROVAL,
    build_graph,
    compile_graph,
    create_agent_config,
    create_initial_state,
)
from accops_agent.llm import LLMClient


def setup_logging(log_level: str) -> None:
    """Configure logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ],
    )


def run_agent_loop(
    compiled_graph,
    initial_state: dict,
    config: dict,
    thread_id: str = "default",
) -> dict:
    """Run the agent loop with human-in-the-loop approval.

    Args:
        compiled_graph: Compiled LangGraph
        initial_state: Initial agent state
        config: Graph configuration (with backend and llm_client)
        thread_id: Thread ID for checkpointing

    Returns:
        Final agent state
    """
    # Add thread_id to config for checkpointing
    run_config = {
        **config,
        "configurable": {
            **config.get("configurable", {}),
            "thread_id": thread_id,
        },
    }

    print_info("Starting agent workflow...")
    print_info(f"User intent: {initial_state.get('user_intent', 'Not specified')}")
    print()

    # Initial run - will stop at human_approval interrupt
    current_state = None

    try:
        # Stream events for visibility
        for event in compiled_graph.stream(initial_state, run_config, stream_mode="values"):
            current_state = event

            # Show progress
            if current_state.get("current_diagnostics"):
                diag_count = len(current_state["current_diagnostics"])
                if diag_count > 0 and not current_state.get("_shown_diag"):
                    print_info(f"Read {diag_count} diagnostics")

            if current_state.get("diagnostic_interpretation") and not current_state.get("_shown_interp"):
                print_info("Diagnostics interpreted")

            if current_state.get("strategy") and not current_state.get("_shown_strategy"):
                print_info("Strategy generated")

            if current_state.get("proposed_actions") and not current_state.get("_shown_actions"):
                action_count = len(current_state["proposed_actions"])
                print_info(f"Generated {action_count} proposed action(s)")

    except Exception as e:
        print_error(f"Error during graph execution: {e}")
        raise

    # Check if we hit the interrupt (human approval needed)
    if current_state and current_state.get("awaiting_approval"):
        return handle_approval_loop(compiled_graph, current_state, run_config)

    # If no interrupt, check final state
    if current_state:
        if current_state.get("goal_achieved"):
            print_success("Goal achieved!")
        elif current_state.get("error"):
            print_error(f"Workflow ended with error: {current_state['error']}")
        else:
            print_info("Workflow completed.")

    return current_state or initial_state


def handle_approval_loop(compiled_graph, state: dict, config: dict) -> dict:
    """Handle the human approval loop.

    Args:
        compiled_graph: Compiled graph
        state: Current state at interrupt
        config: Graph configuration

    Returns:
        Final state after approval workflow
    """
    current_state = state

    while True:
        # Show current state summary
        print(format_state_summary(current_state))

        # Get proposed actions
        proposed_actions = current_state.get("proposed_actions", [])

        if not proposed_actions:
            print_warning("No actions to approve. Ending workflow.")
            return current_state

        # Get user approval
        approval = get_user_approval(proposed_actions)

        # Update state based on approval
        if approval.status == ApprovalStatus.APPROVED:
            update = {
                "approval_status": "approved",
                "awaiting_approval": False,
            }
        elif approval.status == ApprovalStatus.REJECTED:
            update = {
                "approval_status": "rejected",
                "awaiting_approval": False,
            }
            print_info("Workflow ended by user.")
            return {**current_state, **update}
        else:  # MODIFIED
            update = {
                "approval_status": "modified",
                "awaiting_approval": False,
                "user_feedback": approval.feedback or "",
            }

        # Resume graph with updated state
        print_info("Resuming workflow...")

        try:
            for event in compiled_graph.stream(update, config, stream_mode="values"):
                current_state = event

                # Show execution result if available
                if current_state.get("current_execution_result"):
                    result = current_state["current_execution_result"]
                    print(format_execution_result(result))

                # Show verification if available
                if current_state.get("verification_result"):
                    print(format_verification_result(current_state["verification_result"]))

        except Exception as e:
            print_error(f"Error during execution: {e}")
            return current_state

        # Check if we need another approval (modified flow)
        if current_state.get("awaiting_approval"):
            continue

        # Check if we're done
        if current_state.get("goal_achieved"):
            print_success("Goal achieved!")
            break
        elif current_state.get("error"):
            print_error(f"Error: {current_state['error']}")
            break
        elif not current_state.get("continue_optimization", False):
            print_info("Optimization complete.")
            break

        # Continue to next iteration
        print_info(f"Iteration {current_state.get('iteration_count', 0)} complete.")
        print()

    return current_state


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the AccOps Agent for accelerator optimization via MCP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with accelerator configuration
    uv run python scripts/run_agent.py \\
        --config configs/accelerators/example_linac.yaml \\
        --intent "Minimize horizontal beam size"

    # Interactive mode (prompts for intent)
    uv run python scripts/run_agent.py \\
        --config configs/accelerators/example_linac.yaml
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to accelerator configuration YAML file",
    )

    parser.add_argument(
        "--intent",
        "-i",
        type=str,
        default=None,
        help="Optimization intent/goal (prompts if not provided)",
    )

    parser.add_argument(
        "--max-iterations",
        "-m",
        type=int,
        default=10,
        help="Maximum optimization iterations (default: 10)",
    )

    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="LLM model to use (default: gpt-4)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (or set OPENAI_BASE_URL env var)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Print banner
    print_banner()

    # Create MCP backend (config is loaded via MCP server)
    try:
        print_info(f"Initializing MCP backend with config: {args.config}...")
        backend = MCPBackend(config_path=args.config)
        if not backend.initialize():
            raise RuntimeError("Failed to initialize MCP backend")
        print_success(f"Connected to MCP server: {backend.config.name}")
        print_info(f"  Knobs: {len(backend.config.knobs)}")
        print_info(f"  Diagnostics: {len(backend.config.diagnostics)}")
    except Exception as e:
        print_error(f"Failed to create backend: {e}")
        sys.exit(1)

    # Create LLM client
    try:
        print_info(f"Initializing LLM client (model: {args.model})...")
        llm_client = LLMClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
        )
        print_success("LLM client initialized")
    except Exception as e:
        print_error(f"Failed to initialize LLM client: {e}")
        sys.exit(1)

    # Get user intent
    if args.intent:
        user_intent = args.intent
    else:
        user_intent = get_user_intent()

    print()

    # Build and compile graph
    try:
        print_info("Building agent graph...")
        graph = build_graph()

        # Create checkpointer for state persistence
        checkpointer = MemorySaver()

        # Compile with interrupt before human approval
        compiled = compile_graph(
            graph=graph,
            interrupt_before=[HUMAN_APPROVAL],
            checkpointer=checkpointer,
        )
        print_success("Graph compiled")
    except Exception as e:
        print_error(f"Failed to build graph: {e}")
        sys.exit(1)

    # Create initial state
    initial_state = create_initial_state(user_intent=user_intent)
    initial_state["max_iterations"] = args.max_iterations

    # Create graph config
    graph_config = create_agent_config(
        backend=backend,
        llm_client=llm_client,
    )

    # Run the agent
    try:
        final_state = run_agent_loop(
            compiled_graph=compiled,
            initial_state=initial_state,
            config=graph_config,
        )

        # Show final summary
        print()
        print(format_state_summary(final_state))

        if final_state.get("goal_achieved"):
            print_success("Optimization completed successfully!")
        else:
            print_info("Optimization workflow ended.")

    except KeyboardInterrupt:
        print()
        print_warning("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print_error(f"Agent execution failed: {e}")
        logging.exception("Agent execution failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
