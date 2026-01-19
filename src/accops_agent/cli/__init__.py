"""CLI module for human-in-the-loop interaction."""

from .display import (
    format_action_for_display,
    format_actions_table,
    format_diagnostic_summary,
    format_diagnostics_table,
    format_execution_result,
    format_state_summary,
    format_verification_result,
    print_banner,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from .input_handler import (
    ApprovalResponse,
    get_user_approval,
    get_user_input,
    get_user_intent,
    parse_approval_response,
)

__all__ = [
    # Display functions
    "format_action_for_display",
    "format_actions_table",
    "format_diagnostic_summary",
    "format_diagnostics_table",
    "format_execution_result",
    "format_state_summary",
    "format_verification_result",
    "print_banner",
    "print_error",
    "print_info",
    "print_success",
    "print_warning",
    # Input handling
    "ApprovalResponse",
    "get_user_approval",
    "get_user_input",
    "get_user_intent",
    "parse_approval_response",
]
