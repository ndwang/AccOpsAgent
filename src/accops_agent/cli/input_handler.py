"""Input handling for CLI user interaction."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .display import (
    Colors,
    format_action_for_display,
    format_approval_prompt,
    print_error,
    print_info,
)
from ..graph.state import ProposedAction


class ApprovalStatus(str, Enum):
    """Status of user approval."""

    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass
class ApprovalResponse:
    """Response from user approval interaction.

    Attributes:
        status: Approval status (approved/rejected/modified)
        feedback: Optional feedback from user (for modifications)
        modified_actions: Optional modified action list
    """

    status: ApprovalStatus
    feedback: Optional[str] = None
    modified_actions: Optional[List[ProposedAction]] = None


def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Get input from user with optional default.

    Args:
        prompt: Prompt to display
        default: Default value if user enters nothing

    Returns:
        User input or default value
    """
    if default:
        display_prompt = f"{prompt} [{default}]: "
    else:
        display_prompt = f"{prompt}: "

    try:
        user_input = input(display_prompt).strip()
        if not user_input and default:
            return default
        return user_input
    except (EOFError, KeyboardInterrupt):
        print()  # New line after ^C
        return ""


def parse_approval_response(response: str) -> Tuple[Optional[ApprovalStatus], bool]:
    """Parse user's approval response.

    Args:
        response: User's input string

    Returns:
        Tuple of (ApprovalStatus or None if invalid, needs_details flag)
    """
    response = response.lower().strip()

    if response in ("a", "approve", "yes", "y"):
        return ApprovalStatus.APPROVED, False
    elif response in ("r", "reject", "no", "n"):
        return ApprovalStatus.REJECTED, False
    elif response in ("m", "modify", "feedback", "f"):
        return ApprovalStatus.MODIFIED, False
    elif response in ("d", "details", "show"):
        return None, True
    else:
        return None, False


def show_action_details(actions: List[ProposedAction]) -> None:
    """Display detailed view of all actions.

    Args:
        actions: List of proposed actions
    """
    print(f"\n{Colors.BOLD}═══ Detailed Action View ═══{Colors.RESET}")

    for i, action in enumerate(actions):
        print(format_action_for_display(action, i))

    print()


def get_modification_feedback() -> str:
    """Get feedback from user for action modification.

    Returns:
        User's feedback string
    """
    print(f"\n{Colors.YELLOW}Please provide feedback for action re-generation:{Colors.RESET}")
    print("(Describe what changes you want, or type 'cancel' to reject)")
    print()

    feedback_lines = []
    print("Enter your feedback (empty line to finish):")

    while True:
        try:
            line = input()
            if not line:
                break
            if line.lower() == "cancel":
                return ""
            feedback_lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print()
            break

    return "\n".join(feedback_lines)


def get_user_approval(actions: List[ProposedAction]) -> ApprovalResponse:
    """Get user approval for proposed actions.

    This function handles the interactive approval workflow:
    1. Display actions and prompt
    2. Get user response
    3. Handle details view, modification feedback, etc.
    4. Return final approval response

    Args:
        actions: List of proposed actions to approve

    Returns:
        ApprovalResponse with user's decision
    """
    while True:
        # Display approval prompt
        print(format_approval_prompt(actions))

        # Get user response
        response = get_user_input("Your choice", "a")

        # Parse response
        status, needs_details = parse_approval_response(response)

        # Handle details request
        if needs_details:
            show_action_details(actions)
            continue

        # Handle invalid input
        if status is None:
            print_error(f"Invalid response: '{response}'. Please enter a, r, m, or d.")
            continue

        # Handle approval
        if status == ApprovalStatus.APPROVED:
            print_info("Actions approved. Proceeding with execution...")
            return ApprovalResponse(status=status)

        # Handle rejection
        if status == ApprovalStatus.REJECTED:
            print_info("Actions rejected. Ending workflow.")
            return ApprovalResponse(status=status)

        # Handle modification
        if status == ApprovalStatus.MODIFIED:
            feedback = get_modification_feedback()
            if not feedback:
                # User cancelled modification
                print_info("Modification cancelled.")
                continue

            print_info("Feedback received. Re-generating actions...")
            return ApprovalResponse(status=status, feedback=feedback)


def get_continue_confirmation() -> bool:
    """Ask user if they want to continue optimization.

    Returns:
        True if user wants to continue, False otherwise
    """
    response = get_user_input("Continue optimization? (y/n)", "y")
    return response.lower() in ("y", "yes")


def get_user_intent() -> str:
    """Get user's optimization intent/goal.

    Returns:
        User's intent string
    """
    print(f"\n{Colors.CYAN}Enter your optimization goal or intent:{Colors.RESET}")
    print("(e.g., 'Minimize horizontal beam size', 'Correct orbit to zero')")
    print()

    intent = get_user_input("Intent")

    if not intent:
        intent = "Optimize machine performance"
        print_info(f"Using default intent: {intent}")

    return intent
