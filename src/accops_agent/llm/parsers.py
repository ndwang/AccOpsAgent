"""Parsers for extracting structured data from LLM outputs."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ..graph.state import ProposedAction
from ..utils.constants import (
    ACTION_PRIORITY_MAX,
    ACTION_PRIORITY_MIN,
    EFFECTIVENESS_MAX,
    EFFECTIVENESS_MIN,
    VERIFICATION_SUMMARY_MAX_LENGTH,
)

logger = logging.getLogger(__name__)


class ProposedActionResponse(BaseModel):
    """Pydantic model for validating LLM-generated action responses."""

    parameter_name: str = Field(..., min_length=1)
    current_value: float = Field(default=0.0)
    proposed_value: float
    rationale: str = Field(default="No rationale provided")
    expected_impact: str = Field(default="Unknown impact")
    priority: int = Field(default=2, ge=ACTION_PRIORITY_MIN, le=ACTION_PRIORITY_MAX)

    @field_validator("parameter_name")
    @classmethod
    def validate_parameter_name(cls, v: str) -> str:
        """Ensure parameter name is non-empty and stripped."""
        return v.strip()

    def to_proposed_action(self) -> ProposedAction:
        """Convert to ProposedAction TypedDict."""
        return {
            "parameter_name": self.parameter_name,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "rationale": self.rationale,
            "expected_impact": self.expected_impact,
            "priority": self.priority,
        }


class VerificationResponse(BaseModel):
    """Pydantic model for validating LLM verification responses."""

    assessment: str = Field(default="NEUTRAL", pattern=r"^(POSITIVE|NEUTRAL|NEGATIVE)$")
    effectiveness: int = Field(default=5, ge=EFFECTIVENESS_MIN, le=EFFECTIVENESS_MAX)
    recommendation: str = Field(default="ITERATE", pattern=r"^(CONTINUE|ITERATE|REVISE)$")
    summary: str = Field(default="")


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text that may contain markdown or other content.

    Args:
        text: Text potentially containing JSON

    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Try direct JSON parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code blocks
    json_pattern = r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```"
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass

    # Try to find JSON anywhere in the text
    json_pattern = r"(\{.*?\}|\[.*?\])"
    matches = re.findall(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            parsed = json.loads(match)
            # Verify it looks like our expected structure
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            continue

    logger.warning("Could not extract JSON from text")
    return None


def parse_actions_from_llm(llm_output: str) -> List[ProposedAction]:
    """Parse proposed actions from LLM output.

    Expects JSON array of action objects with required fields.
    Uses Pydantic models for validation.

    Args:
        llm_output: Raw LLM output text

    Returns:
        List of ProposedAction dicts
    """
    # Extract JSON from output
    parsed = parse_json_from_text(llm_output)

    if parsed is None:
        logger.warning("No JSON found in LLM output, attempting text parsing")
        return parse_actions_from_text(llm_output)

    # Handle both single object and array
    if isinstance(parsed, dict):
        parsed = [parsed]

    actions: List[ProposedAction] = []
    for action_data in parsed:
        try:
            # Use Pydantic model for validation
            validated = ProposedActionResponse.model_validate(action_data)
            actions.append(validated.to_proposed_action())
        except Exception as e:
            logger.warning(f"Action validation failed: {e}, data: {action_data}")
            continue

    logger.info(f"Parsed {len(actions)} action(s) from LLM output")
    return actions


def parse_actions_from_text(text: str) -> List[ProposedAction]:
    """Fallback parser to extract actions from free-form text.

    Looks for patterns like:
    - Parameter: QF1_K1
    - Value: 2.5
    - etc.

    Args:
        text: Free-form text describing actions

    Returns:
        List of ProposedAction dicts (best effort)
    """
    actions: List[ProposedAction] = []

    # Look for parameter name patterns
    param_pattern = r"(?:parameter|param).*?:\s*([A-Z0-9_]+)"
    value_pattern = r"(?:proposed|new|set).*?value.*?:\s*([-+]?\d+\.?\d*)"

    param_matches = re.findall(param_pattern, text, re.IGNORECASE)
    value_matches = re.findall(value_pattern, text, re.IGNORECASE)

    # Try to pair parameters with values
    for param, value in zip(param_matches, value_matches):
        action: ProposedAction = {
            "parameter_name": param,
            "current_value": 0.0,
            "proposed_value": float(value),
            "rationale": "Parsed from text description",
            "expected_impact": "See LLM output for details",
            "priority": 2,
        }
        actions.append(action)

    if actions:
        logger.info(f"Extracted {len(actions)} action(s) from text using pattern matching")

    return actions


def parse_issues_from_text(text: str) -> List[str]:
    """Extract list of issues from LLM interpretation.

    Args:
        text: LLM output text

    Returns:
        List of issue descriptions
    """
    issues = []

    # Look for numbered lists
    list_pattern = r"^\s*\d+\.\s*(.+)$"
    matches = re.findall(list_pattern, text, re.MULTILINE)
    if matches:
        issues.extend(matches)

    # Look for bullet points
    bullet_pattern = r"^\s*[-•]\s*(.+)$"
    matches = re.findall(bullet_pattern, text, re.MULTILINE)
    if matches:
        issues.extend(matches)

    # Look for explicit "Issue:" markers
    issue_pattern = r"Issue:\s*(.+?)(?:\n|$)"
    matches = re.findall(issue_pattern, text, re.IGNORECASE)
    if matches:
        issues.extend(matches)

    # Clean up issues
    issues = [issue.strip() for issue in issues if issue.strip()]

    # Remove duplicates while preserving order
    seen = set()
    unique_issues = []
    for issue in issues:
        if issue.lower() not in seen:
            seen.add(issue.lower())
            unique_issues.append(issue)

    return unique_issues


def parse_verification_result(text: str) -> Dict[str, Any]:
    """Parse verification assessment from LLM output.

    Uses Pydantic model for validation with fallback defaults.

    Args:
        text: LLM verification output

    Returns:
        Dict with assessment, effectiveness, recommendation, summary
    """
    # Extract values from text using regex
    assessment = "NEUTRAL"
    effectiveness = 5
    recommendation = "ITERATE"

    # Look for assessment
    assessment_pattern = r"(?:Assessment|Result):\s*(POSITIVE|NEUTRAL|NEGATIVE)"
    match = re.search(assessment_pattern, text, re.IGNORECASE)
    if match:
        assessment = match.group(1).upper()

    # Look for effectiveness rating
    effectiveness_pattern = r"(?:Effectiveness|Rating):\s*(\d+)"
    match = re.search(effectiveness_pattern, text, re.IGNORECASE)
    if match:
        effectiveness = min(EFFECTIVENESS_MAX, max(EFFECTIVENESS_MIN, int(match.group(1))))

    # Look for recommendation
    recommendation_pattern = r"(?:Recommendation|Should we):\s*(CONTINUE|ITERATE|REVISE)"
    match = re.search(recommendation_pattern, text, re.IGNORECASE)
    if match:
        recommendation = match.group(1).upper()

    # Validate with Pydantic model
    try:
        validated = VerificationResponse(
            assessment=assessment,
            effectiveness=effectiveness,
            recommendation=recommendation,
            summary=text[:VERIFICATION_SUMMARY_MAX_LENGTH],
        )
        return validated.model_dump()
    except Exception as e:
        logger.warning(f"Verification response validation failed: {e}")
        return {
            "assessment": "NEUTRAL",
            "effectiveness": 5,
            "recommendation": "ITERATE",
            "summary": text[:VERIFICATION_SUMMARY_MAX_LENGTH],
        }
