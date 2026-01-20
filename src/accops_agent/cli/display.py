"""Display formatting for CLI output."""

from typing import Any, Dict, List, Optional

from ..accelerator_interface import ActionResult, DiagnosticSnapshot, DiagnosticStatus
from ..graph.state import AgentState, ProposedAction


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def print_banner() -> None:
    """Print application banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗
║                  AccOps Agent - AI Operations                 ║
║            Accelerator Optimization with Human-in-the-Loop    ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def print_info(message: str) -> None:
    """Print info message."""
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {message}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {message}")


def format_diagnostic_status(status: DiagnosticStatus) -> str:
    """Format diagnostic status with color."""
    if status == DiagnosticStatus.NORMAL:
        return f"{Colors.GREEN}NORMAL{Colors.RESET}"
    elif status == DiagnosticStatus.WARNING:
        return f"{Colors.YELLOW}WARNING{Colors.RESET}"
    elif status == DiagnosticStatus.ALARM:
        return f"{Colors.RED}ALARM{Colors.RESET}"
    elif status == DiagnosticStatus.ERROR:
        return f"{Colors.RED}ERROR{Colors.RESET}"
    else:
        return f"{Colors.DIM}UNKNOWN{Colors.RESET}"


def format_diagnostics_table(diagnostics: List[DiagnosticSnapshot]) -> str:
    """Format diagnostics as a table.

    Args:
        diagnostics: List of diagnostic snapshots

    Returns:
        Formatted table string
    """
    if not diagnostics:
        return "No diagnostics available."

    # Calculate column widths
    name_width = max(len(d.diagnostic_name) for d in diagnostics)
    name_width = max(name_width, 10)

    lines = [
        f"\n{Colors.BOLD}{'Diagnostic':<{name_width}}  {'Value':>12}  {'Unit':<8}  {'Status':<10}{Colors.RESET}",
        "-" * (name_width + 40),
    ]

    for diag in diagnostics:
        status_str = format_diagnostic_status(diag.status)
        lines.append(
            f"{diag.diagnostic_name:<{name_width}}  {diag.value:>12.4f}  {diag.unit:<8}  {status_str}"
        )

    return "\n".join(lines)


def format_diagnostic_summary(diagnostics: List[DiagnosticSnapshot]) -> str:
    """Format a summary of diagnostics.

    Args:
        diagnostics: List of diagnostic snapshots

    Returns:
        Summary string
    """
    if not diagnostics:
        return "No diagnostics available."

    total = len(diagnostics)
    normal = sum(1 for d in diagnostics if d.status == DiagnosticStatus.NORMAL)
    warning = sum(1 for d in diagnostics if d.status == DiagnosticStatus.WARNING)
    alarm = sum(1 for d in diagnostics if d.status == DiagnosticStatus.ALARM)

    summary = f"Total: {total} | "
    summary += f"{Colors.GREEN}Normal: {normal}{Colors.RESET} | "
    summary += f"{Colors.YELLOW}Warning: {warning}{Colors.RESET} | "
    summary += f"{Colors.RED}Alarm: {alarm}{Colors.RESET}"

    return summary


def format_action_for_display(action: ProposedAction, index: int = 0) -> str:
    """Format a single action for display.

    Args:
        action: The proposed action
        index: Action index (1-based for display)

    Returns:
        Formatted action string
    """
    lines = [
        f"\n{Colors.BOLD}Action {index + 1}:{Colors.RESET}",
        f"  Parameter: {Colors.CYAN}{action['parameter_name']}{Colors.RESET}",
        f"  Current:   {action.get('current_value', 'N/A'):.4f}" if isinstance(action.get('current_value'), (int, float)) else f"  Current:   {action.get('current_value', 'N/A')}",
        f"  Proposed:  {Colors.YELLOW}{action['proposed_value']:.4f}{Colors.RESET}" if isinstance(action.get('proposed_value'), (int, float)) else f"  Proposed:  {Colors.YELLOW}{action.get('proposed_value', 'N/A')}{Colors.RESET}",
    ]

    if action.get("rationale"):
        lines.append(f"  Rationale: {action['rationale']}")

    if action.get("expected_impact"):
        lines.append(f"  Expected:  {action['expected_impact']}")

    return "\n".join(lines)


def format_actions_table(actions: List[ProposedAction]) -> str:
    """Format proposed actions as a table.

    Args:
        actions: List of proposed actions

    Returns:
        Formatted table string
    """
    if not actions:
        return "No actions proposed."

    lines = [
        f"\n{Colors.BOLD}{'#':<3}  {'Parameter':<20}  {'Current':>12}  {'Proposed':>12}  {'Rationale'}{Colors.RESET}",
        "-" * 80,
    ]

    for i, action in enumerate(actions):
        current = action.get("current_value", 0)
        proposed = action.get("proposed_value", 0)
        rationale = action.get("rationale", "")[:30]

        if isinstance(current, (int, float)) and isinstance(proposed, (int, float)):
            lines.append(
                f"{i + 1:<3}  {action['parameter_name']:<20}  {current:>12.4f}  {proposed:>12.4f}  {rationale}"
            )
        else:
            lines.append(
                f"{i + 1:<3}  {action['parameter_name']:<20}  {str(current):>12}  {str(proposed):>12}  {rationale}"
            )

    return "\n".join(lines)


def format_execution_result(result: ActionResult) -> str:
    """Format execution result for display.

    Args:
        result: Action execution result

    Returns:
        Formatted result string
    """
    if result.success:
        status = f"{Colors.GREEN}SUCCESS{Colors.RESET}"
    else:
        status = f"{Colors.RED}FAILED{Colors.RESET}"

    lines = [
        f"\n{Colors.BOLD}Execution Result:{Colors.RESET}",
        f"  Status:  {status}",
        f"  Message: {result.message}",
    ]

    if result.error:
        lines.append(f"  Error:   {Colors.RED}{result.error}{Colors.RESET}")

    return "\n".join(lines)


def format_state_summary(state: AgentState) -> str:
    """Format a summary of the current agent state.

    Args:
        state: Current agent state

    Returns:
        Formatted summary string
    """
    lines = [
        f"\n{Colors.BOLD}═══ Agent State Summary ═══{Colors.RESET}",
    ]

    # User intent
    if state.get("user_intent"):
        lines.append(f"\n{Colors.CYAN}User Intent:{Colors.RESET} {state['user_intent']}")

    # Machine status
    if state.get("machine_status_summary"):
        lines.append(f"\n{Colors.CYAN}Machine Status:{Colors.RESET} {state['machine_status_summary']}")

    # Diagnostics summary
    diagnostics = state.get("current_diagnostics", [])
    if diagnostics:
        lines.append(f"\n{Colors.CYAN}Diagnostics:{Colors.RESET}")
        lines.append(f"  {format_diagnostic_summary(diagnostics)}")

    # Get analysis state
    analysis = state.get("analysis", {})

    # Interpretation
    if analysis.get("interpretation"):
        interp = analysis["interpretation"]
        if len(interp) > 200:
            interp = interp[:200] + "..."
        lines.append(f"\n{Colors.CYAN}Interpretation:{Colors.RESET}")
        lines.append(f"  {interp}")

    # Identified issues
    issues = analysis.get("issues", [])
    if issues:
        lines.append(f"\n{Colors.CYAN}Identified Issues:{Colors.RESET}")
        for issue in issues[:5]:  # Limit to 5 issues
            lines.append(f"  - {issue}")

    # Strategy
    if analysis.get("strategy"):
        strategy = analysis["strategy"]
        if len(strategy) > 150:
            strategy = strategy[:150] + "..."
        lines.append(f"\n{Colors.CYAN}Strategy:{Colors.RESET}")
        lines.append(f"  {strategy}")

    # Get workflow state for iteration info
    workflow = state.get("workflow", {})
    iteration = workflow.get("iteration_count", 0)
    max_iter = workflow.get("max_iterations", 10)
    lines.append(f"\n{Colors.DIM}Iteration: {iteration}/{max_iter}{Colors.RESET}")

    return "\n".join(lines)


def format_verification_result(verification: str) -> str:
    """Format verification result for display.

    Args:
        verification: Verification result string

    Returns:
        Formatted string
    """
    lines = [
        f"\n{Colors.BOLD}═══ Verification Result ═══{Colors.RESET}",
        verification,
    ]
    return "\n".join(lines)


def format_approval_prompt(actions: List[ProposedAction]) -> str:
    """Format the approval prompt for user.

    Args:
        actions: List of proposed actions

    Returns:
        Formatted prompt string
    """
    lines = [
        f"\n{Colors.BOLD}═══ Actions Pending Approval ═══{Colors.RESET}",
        format_actions_table(actions),
        f"\n{Colors.YELLOW}Please review the proposed actions above.{Colors.RESET}",
        "",
        "Options:",
        f"  {Colors.GREEN}[a]{Colors.RESET}pprove  - Execute the proposed actions",
        f"  {Colors.RED}[r]{Colors.RESET}eject   - Cancel and end the workflow",
        f"  {Colors.YELLOW}[m]{Colors.RESET}odify   - Provide feedback for re-generation",
        f"  {Colors.BLUE}[d]{Colors.RESET}etails  - Show detailed view of actions",
        "",
    ]
    return "\n".join(lines)
