"""Prompt templates for LLM-based agent reasoning."""

import logging
from pathlib import Path
from typing import Dict, List

import yaml

from ..diagnostic_control import DiagnosticSnapshot

logger = logging.getLogger(__name__)

# Cache for loaded prompt configs
_PROMPT_CACHE: Dict[str, Dict] = {}


def load_prompt_config(prompt_name: str) -> Dict[str, str]:
    """Load prompt configuration from YAML file.

    Args:
        prompt_name: Name of prompt config (e.g., 'diagnostic_interpretation')

    Returns:
        Dict with 'system_prompt' and 'user_prompt_template' keys
    """
    if prompt_name in _PROMPT_CACHE:
        return _PROMPT_CACHE[prompt_name]

    # Find configs directory (relative to this file)
    config_path = (
        Path(__file__).parent.parent.parent.parent
        / "configs"
        / "prompts"
        / f"{prompt_name}.yaml"
    )

    if not config_path.exists():
        logger.warning(f"Prompt config not found: {config_path}, using defaults")
        return {"system_prompt": "", "user_prompt_template": ""}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        _PROMPT_CACHE[prompt_name] = config
        logger.debug(f"Loaded prompt config: {prompt_name}")
        return config

    except Exception as e:
        logger.error(f"Failed to load prompt config {prompt_name}: {e}")
        return {"system_prompt": "", "user_prompt_template": ""}


# Prompt configuration mapping: config_name -> exported_variable_name
_PROMPT_CONFIG_MAPPING = {
    "diagnostic_interpretation": "DIAGNOSTIC_INTERPRETATION_SYSTEM",
    "reasoning_planning": "REASONING_PLANNING_SYSTEM",
    "action_generation": "ACTION_GENERATION_SYSTEM",
    "verification": "VERIFICATION_SYSTEM",
}

# Load all system prompts from config files
def _load_system_prompts() -> Dict[str, str]:
    """Load all system prompts from config files."""
    prompts = {}
    for config_name, var_name in _PROMPT_CONFIG_MAPPING.items():
        config = load_prompt_config(config_name)
        prompts[var_name] = config.get("system_prompt", "")
    return prompts

_LOADED_PROMPTS = _load_system_prompts()

# Export system prompts as module-level constants
DIAGNOSTIC_INTERPRETATION_SYSTEM = _LOADED_PROMPTS["DIAGNOSTIC_INTERPRETATION_SYSTEM"]
REASONING_PLANNING_SYSTEM = _LOADED_PROMPTS["REASONING_PLANNING_SYSTEM"]
ACTION_GENERATION_SYSTEM = _LOADED_PROMPTS["ACTION_GENERATION_SYSTEM"]
VERIFICATION_SYSTEM = _LOADED_PROMPTS["VERIFICATION_SYSTEM"]


def create_diagnostic_interpretation_prompt(
    diagnostics: List[DiagnosticSnapshot],
    machine_status_summary: str,
) -> str:
    """Create prompt for diagnostic interpretation.

    Args:
        diagnostics: List of diagnostic measurements
        machine_status_summary: Overall machine status

    Returns:
        Formatted prompt string
    """
    diag_lines = []
    for diag in diagnostics:
        status_symbol = "✓" if diag.is_healthy() else "⚠" if diag.status.value == "warning" else "❌"
        diag_lines.append(
            f"  {status_symbol} {diag.diagnostic_name}: {diag.value:.4f} {diag.unit} "
            f"(status: {diag.status.value})"
        )
        if diag.message:
            diag_lines.append(f"     Message: {diag.message}")

    diagnostics_text = "\n".join(diag_lines)

    # Load template from config
    config = load_prompt_config("diagnostic_interpretation")
    template = config.get("user_prompt_template", "")

    # Format template with provided data
    prompt = template.format(
        machine_status_summary=machine_status_summary,
        diagnostics_list=diagnostics_text,
    )

    return prompt


def create_reasoning_planning_prompt(
    user_intent: str,
    diagnostic_interpretation: str,
    identified_issues: List[str],
    current_parameters: Dict[str, float],
) -> str:
    """Create prompt for reasoning and strategy planning.

    Args:
        user_intent: User's goal or request
        diagnostic_interpretation: LLM interpretation of diagnostics
        identified_issues: List of identified issues
        current_parameters: Current parameter values

    Returns:
        Formatted prompt string
    """
    issues_text = "\n".join(f"  - {issue}" for issue in identified_issues) if identified_issues else "  None identified"

    param_lines = [f"  - {name}: {value:.4f}" for name, value in list(current_parameters.items())[:10]]
    if len(current_parameters) > 10:
        param_lines.append(f"  ... and {len(current_parameters) - 10} more")
    params_text = "\n".join(param_lines)

    # Load template from config
    config = load_prompt_config("reasoning_planning")
    template = config.get("user_prompt_template", "")

    # Format template with provided data
    prompt = template.format(
        user_intent=user_intent,
        diagnostic_interpretation=diagnostic_interpretation,
        identified_issues=issues_text,
        current_parameters=params_text,
    )

    return prompt


def create_action_generation_prompt(
    user_intent: str,
    strategy: str,
    reasoning: str,
    current_parameters: Dict[str, float],
    parameter_metadata: Dict[str, Dict],
) -> str:
    """Create prompt for generating specific actions.

    Args:
        user_intent: User's goal
        strategy: High-level strategy
        reasoning: Detailed reasoning
        current_parameters: Current parameter values
        parameter_metadata: Metadata about parameters (limits, types, etc.)

    Returns:
        Formatted prompt string
    """
    param_lines = []
    for name, value in current_parameters.items():
        metadata = parameter_metadata.get(name, {})
        param_lines.append(
            f"  - {name}: {value:.4f} "
            f"(range: [{metadata.get('min', '?')} to {metadata.get('max', '?')}], "
            f"type: {metadata.get('type', 'unknown')})"
        )

    params_text = "\n".join(param_lines)

    # Load template from config
    config = load_prompt_config("action_generation")
    template = config.get("user_prompt_template", "")

    # Format template with provided data
    prompt = template.format(
        user_intent=user_intent,
        strategy=strategy,
        reasoning=reasoning,
        available_parameters=params_text,
    )

    return prompt


def create_verification_prompt(
    action_description: str,
    diagnostics_before: List[DiagnosticSnapshot],
    diagnostics_after: List[DiagnosticSnapshot],
    user_intent: str,
) -> str:
    """Create prompt for verifying action results.

    Args:
        action_description: Description of action taken
        diagnostics_before: Diagnostics before action
        diagnostics_after: Diagnostics after action
        user_intent: Original user intent

    Returns:
        Formatted prompt string
    """
    def format_diag_list(diags: List[DiagnosticSnapshot]) -> str:
        lines = []
        for d in diags:
            status_symbol = "✓" if d.is_healthy() else "⚠" if d.status.value == "warning" else "❌"
            lines.append(f"  {status_symbol} {d.diagnostic_name}: {d.value:.4f} {d.unit}")
        return "\n".join(lines)

    before_text = format_diag_list(diagnostics_before)
    after_text = format_diag_list(diagnostics_after)

    # Load template from config
    config = load_prompt_config("verification")
    template = config.get("user_prompt_template", "")

    # Format template with provided data
    prompt = template.format(
        action_description=action_description,
        user_intent=user_intent,
        diagnostics_before=before_text,
        diagnostics_after=after_text,
    )

    return prompt
