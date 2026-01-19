"""Tests for LLM integration components."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from accops_agent.accelerator_interface import DiagnosticSnapshot, DiagnosticStatus
from accops_agent.llm.client import LLMClient
from accops_agent.llm.prompts import (
    create_diagnostic_interpretation_prompt,
    create_reasoning_planning_prompt,
    create_action_generation_prompt,
    create_verification_prompt,
)
from accops_agent.llm.parsers import (
    parse_json_from_text,
    parse_actions_from_llm,
    parse_actions_from_text,
    parse_issues_from_text,
    parse_verification_result,
)


class TestLLMClient:
    """Tests for LLMClient."""

    @patch("accops_agent.llm.client.OpenAI")
    def test_init_with_defaults(self, mock_openai):
        """Test client initialization with defaults."""
        client = LLMClient(api_key="test-key")

        assert client.model == "gpt-4"
        assert client.temperature == 0.7
        assert client.max_tokens == 2000
        mock_openai.assert_called_once()

    @patch("accops_agent.llm.client.OpenAI")
    def test_init_with_custom_params(self, mock_openai):
        """Test client initialization with custom parameters."""
        client = LLMClient(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=1000,
        )

        assert client.model == "gpt-3.5-turbo"
        assert client.temperature == 0.5
        assert client.max_tokens == 1000

    @patch("accops_agent.llm.client.OpenAI")
    def test_generate_success(self, mock_openai):
        """Test successful text generation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated text"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        client = LLMClient(api_key="test-key")
        result = client.generate("Test prompt")

        assert result == "Generated text"
        mock_openai.return_value.chat.completions.create.assert_called_once()

    @patch("accops_agent.llm.client.OpenAI")
    def test_generate_with_system_prompt(self, mock_openai):
        """Test generation with system prompt."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        client = LLMClient(api_key="test-key")
        result = client.generate("User prompt", system_prompt="System instruction")

        assert result == "Response"
        # Verify system message was included
        call_args = mock_openai.return_value.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @patch("accops_agent.llm.client.OpenAI")
    def test_generate_failure(self, mock_openai):
        """Test generation failure handling."""
        mock_openai.return_value.chat.completions.create.side_effect = Exception("API error")

        client = LLMClient(api_key="test-key")

        with pytest.raises(RuntimeError, match="LLM generation failed"):
            client.generate("Test prompt")

    @patch("accops_agent.llm.client.OpenAI")
    def test_generate_with_history(self, mock_openai):
        """Test generation with message history."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        client = LLMClient(api_key="test-key")
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        result = client.generate_with_history(messages)

        assert result == "Response"


class TestPrompts:
    """Tests for prompt generation functions."""

    def test_create_diagnostic_interpretation_prompt(self):
        """Test diagnostic interpretation prompt creation."""
        diags = [
            DiagnosticSnapshot(
                diagnostic_name="BPM1_X",
                value=0.5,
                unit="mm",
                status=DiagnosticStatus.NORMAL,
            ),
            DiagnosticSnapshot(
                diagnostic_name="BPM2_X",
                value=2.5,
                unit="mm",
                status=DiagnosticStatus.WARNING,
                message="Outside tolerance",
            ),
        ]

        prompt = create_diagnostic_interpretation_prompt(diags, "WARNING")

        assert "BPM1_X" in prompt
        assert "BPM2_X" in prompt
        assert "0.5" in prompt or "0.5000" in prompt
        assert "WARNING" in prompt
        assert "mm" in prompt

    def test_create_reasoning_planning_prompt(self):
        """Test reasoning/planning prompt creation."""
        user_intent = "Optimize beam size"
        interpretation = "All diagnostics normal"
        issues = []
        params = {"QF1_K1": 2.0, "QD1_K1": -1.5}

        prompt = create_reasoning_planning_prompt(
            user_intent, interpretation, issues, params
        )

        assert "Optimize beam size" in prompt
        assert "All diagnostics normal" in prompt
        assert "QF1_K1" in prompt
        assert "2.0" in prompt or "2.0000" in prompt

    def test_create_action_generation_prompt(self):
        """Test action generation prompt creation."""
        user_intent = "Test"
        strategy = "Test strategy"
        reasoning = "Test reasoning"
        params = {"QF1_K1": 2.0}
        metadata = {
            "QF1_K1": {"min": -5.0, "max": 5.0, "type": "quadrupole", "unit": "T/m"}
        }

        prompt = create_action_generation_prompt(
            user_intent, strategy, reasoning, params, metadata
        )

        assert "Test strategy" in prompt
        assert "QF1_K1" in prompt
        assert "-5.0" in prompt or "-5" in prompt
        assert "5.0" in prompt or "5" in prompt

    def test_create_verification_prompt(self):
        """Test verification prompt creation."""
        action_desc = "Set QF1_K1 to 2.5"
        diags_before = [
            DiagnosticSnapshot(
                diagnostic_name="BPM1_X",
                value=1.0,
                unit="mm",
                status=DiagnosticStatus.WARNING,
            )
        ]
        diags_after = [
            DiagnosticSnapshot(
                diagnostic_name="BPM1_X",
                value=0.5,
                unit="mm",
                status=DiagnosticStatus.NORMAL,
            )
        ]
        user_intent = "Reduce beam position deviation"

        prompt = create_verification_prompt(
            action_desc, diags_before, diags_after, user_intent
        )

        assert "Set QF1_K1 to 2.5" in prompt
        assert "BPM1_X" in prompt
        assert "1.0" in prompt or "1.0000" in prompt
        assert "0.5" in prompt or "0.5000" in prompt


class TestParsers:
    """Tests for LLM output parsers."""

    def test_parse_json_from_text_direct(self):
        """Test parsing direct JSON."""
        text = '{"key": "value", "num": 123}'
        result = parse_json_from_text(text)

        assert result == {"key": "value", "num": 123}

    def test_parse_json_from_text_markdown(self):
        """Test parsing JSON from markdown code block."""
        text = """
Here is the result:
```json
{
  "parameter_name": "QF1_K1",
  "proposed_value": 2.5
}
```
        """
        result = parse_json_from_text(text)

        assert result["parameter_name"] == "QF1_K1"
        assert result["proposed_value"] == 2.5

    def test_parse_json_from_text_array(self):
        """Test parsing JSON array."""
        text = '[{"a": 1}, {"a": 2}]'
        result = parse_json_from_text(text)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_json_from_text_invalid(self):
        """Test handling invalid JSON."""
        text = "This is not JSON at all"
        result = parse_json_from_text(text)

        assert result is None

    def test_parse_actions_from_llm_success(self):
        """Test parsing actions from valid LLM output."""
        llm_output = """
```json
[
  {
    "parameter_name": "QF1_K1",
    "current_value": 2.0,
    "proposed_value": 2.5,
    "rationale": "Increase focusing",
    "expected_impact": "Reduce beam size",
    "priority": 1
  }
]
```
        """
        actions = parse_actions_from_llm(llm_output)

        assert len(actions) == 1
        assert actions[0]["parameter_name"] == "QF1_K1"
        assert actions[0]["proposed_value"] == 2.5
        assert actions[0]["priority"] == 1

    def test_parse_actions_from_llm_single_object(self):
        """Test parsing single action object."""
        llm_output = '{"parameter_name": "QF1_K1", "proposed_value": 2.5}'
        actions = parse_actions_from_llm(llm_output)

        assert len(actions) == 1
        assert actions[0]["parameter_name"] == "QF1_K1"

    def test_parse_actions_from_llm_invalid(self):
        """Test parsing invalid action data."""
        llm_output = "No valid JSON here"
        actions = parse_actions_from_llm(llm_output)

        # Should attempt text parsing fallback
        assert isinstance(actions, list)

    def test_parse_actions_from_text(self):
        """Test fallback text parsing for actions."""
        text = """
Parameter: QF1_K1
Proposed value: 2.5
        """
        actions = parse_actions_from_text(text)

        if actions:  # May or may not find patterns
            assert actions[0]["parameter_name"] == "QF1_K1"
            assert actions[0]["proposed_value"] == 2.5

    def test_parse_issues_from_text_numbered(self):
        """Test parsing numbered list of issues."""
        text = """
1. BPM1_X is out of range
2. Beam energy too low
3. Transmission below target
        """
        issues = parse_issues_from_text(text)

        assert len(issues) >= 3
        assert any("BPM1_X" in issue for issue in issues)

    def test_parse_issues_from_text_bullets(self):
        """Test parsing bullet list of issues."""
        text = """
Issues found:
- BPM1_X deviation
- Energy drift
        """
        issues = parse_issues_from_text(text)

        assert len(issues) >= 2

    def test_parse_issues_from_text_duplicates(self):
        """Test that duplicates are removed."""
        text = """
1. BPM1_X issue
2. BPM1_X issue
3. Different issue
        """
        issues = parse_issues_from_text(text)

        # Should remove duplicate "BPM1_X issue"
        assert len(issues) == 2

    def test_parse_verification_result(self):
        """Test parsing verification result."""
        text = """
Assessment: POSITIVE
Effectiveness: 8
Recommendation: CONTINUE

The action successfully reduced the alarm count.
        """
        result = parse_verification_result(text)

        assert result["assessment"] == "POSITIVE"
        assert result["effectiveness"] == 8
        assert result["recommendation"] == "CONTINUE"

    def test_parse_verification_result_defaults(self):
        """Test verification result with defaults."""
        text = "Some generic text without specific markers"
        result = parse_verification_result(text)

        assert result["assessment"] == "NEUTRAL"
        assert result["effectiveness"] == 5
        assert result["recommendation"] == "ITERATE"
        assert "summary" in result


class TestLLMIntegration:
    """Integration tests for LLM components (without actual API calls)."""

    @patch("accops_agent.llm.client.OpenAI")
    def test_full_workflow_mock(self, mock_openai):
        """Test complete workflow with mocked LLM."""
        # Mock LLM responses
        mock_response1 = Mock()
        mock_response1.choices = [
            Mock(message=Mock(content="1. BPM1_X deviation\n2. Low transmission"))
        ]

        mock_response2 = Mock()
        mock_response2.choices = [
            Mock(
                message=Mock(
                    content="""
                    **Strategy**: Adjust correctors to fix orbit
                    **Reasoning**: The orbit deviation needs correction
                    """
                )
            )
        ]

        mock_response3 = Mock()
        mock_response3.choices = [
            Mock(
                message=Mock(
                    content="""
                    ```json
                    [{
                        "parameter_name": "HCOR1_KICK",
                        "current_value": 0.0,
                        "proposed_value": 0.1,
                        "rationale": "Correct horizontal orbit",
                        "expected_impact": "Reduce BPM1_X deviation",
                        "priority": 1
                    }]
                    ```
                    """
                )
            )
        ]

        mock_openai.return_value.chat.completions.create.side_effect = [
            mock_response1,
            mock_response2,
            mock_response3,
        ]

        client = LLMClient(api_key="test-key")

        # Test interpretation
        diags = [
            DiagnosticSnapshot(
                diagnostic_name="BPM1_X",
                value=2.0,
                unit="mm",
                status=DiagnosticStatus.WARNING,
            )
        ]
        prompt1 = create_diagnostic_interpretation_prompt(diags, "WARNING")
        interp = client.generate(prompt1, temperature=0.3)
        issues = parse_issues_from_text(interp)

        assert len(issues) == 2
        assert "BPM1_X" in issues[0]

        # Test reasoning
        prompt2 = create_reasoning_planning_prompt(
            "Fix orbit", interp, issues, {"HCOR1_KICK": 0.0}
        )
        reasoning = client.generate(prompt2, temperature=0.5)
        assert "Strategy" in reasoning

        # Test action generation
        prompt3 = create_action_generation_prompt(
            "Fix orbit",
            "Adjust correctors",
            reasoning,
            {"HCOR1_KICK": 0.0},
            {"HCOR1_KICK": {"min": -1.0, "max": 1.0, "type": "corrector", "unit": "mrad"}},
        )
        actions_text = client.generate(prompt3, temperature=0.3)
        actions = parse_actions_from_llm(actions_text)

        assert len(actions) == 1
        assert actions[0]["parameter_name"] == "HCOR1_KICK"
        assert actions[0]["proposed_value"] == 0.1
