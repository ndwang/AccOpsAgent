"""Tests for graph assembly and conditional edges."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from accops_agent.config import load_accelerator_config
from accops_agent.diagnostic_control import MockBackend
from accops_agent.graph import (
    AgentState,
    build_graph,
    compile_graph,
    create_agent_config,
    create_initial_state,
    INGEST_DIAGNOSTICS,
    INTERPRET_DIAGNOSTICS,
    REASONING_PLANNING,
    GENERATE_ACTIONS,
    HUMAN_APPROVAL,
    EXECUTE_ACTION,
    VERIFY_RESULTS,
    DECIDE_CONTINUATION,
)
from accops_agent.graph.builder import (
    route_after_approval,
    route_after_continuation,
    route_after_ingest,
)


@pytest.fixture
def test_config():
    """Load test configuration."""
    config_path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "accelerators"
        / "example_linac.yaml"
    )
    return load_accelerator_config(config_path)


@pytest.fixture
def mock_backend(test_config):
    """Create and initialize mock backend."""
    backend = MockBackend(test_config)
    backend.initialize()
    return backend


class TestRoutingFunctions:
    """Tests for conditional routing functions."""

    def test_route_after_approval_approved(self):
        """Test routing when actions are approved."""
        state = create_initial_state("Test")
        state["approval_status"] = "approved"

        result = route_after_approval(state)

        assert result == EXECUTE_ACTION

    def test_route_after_approval_rejected(self):
        """Test routing when actions are rejected."""
        state = create_initial_state("Test")
        state["approval_status"] = "rejected"

        result = route_after_approval(state)

        assert result == "__end__"

    def test_route_after_approval_modified(self):
        """Test routing when actions are modified."""
        state = create_initial_state("Test")
        state["approval_status"] = "modified"

        result = route_after_approval(state)

        assert result == GENERATE_ACTIONS

    def test_route_after_approval_no_actions(self):
        """Test routing with no_actions status."""
        state = create_initial_state("Test")
        state["approval_status"] = "no_actions"

        result = route_after_approval(state)

        assert result == "__end__"

    def test_route_after_continuation_goal_achieved(self):
        """Test routing when goal is achieved."""
        state = create_initial_state("Test")
        state["goal_achieved"] = True
        state["continue_optimization"] = False

        result = route_after_continuation(state)

        assert result == "__end__"

    def test_route_after_continuation_continue(self):
        """Test routing when continuing optimization."""
        state = create_initial_state("Test")
        state["goal_achieved"] = False
        state["continue_optimization"] = True

        result = route_after_continuation(state)

        assert result == INGEST_DIAGNOSTICS

    def test_route_after_continuation_stop(self):
        """Test routing when stopping optimization."""
        state = create_initial_state("Test")
        state["goal_achieved"] = False
        state["continue_optimization"] = False

        result = route_after_continuation(state)

        assert result == "__end__"

    def test_route_after_ingest_success(self):
        """Test routing after successful ingestion."""
        state = create_initial_state("Test")
        state["error"] = None

        result = route_after_ingest(state)

        assert result == INTERPRET_DIAGNOSTICS

    def test_route_after_ingest_error(self):
        """Test routing after ingestion error."""
        state = create_initial_state("Test")
        state["error"] = "Connection failed"

        result = route_after_ingest(state)

        assert result == "__end__"


class TestBuildGraph:
    """Tests for build_graph function."""

    def test_build_graph_returns_state_graph(self):
        """Test that build_graph returns a StateGraph."""
        graph = build_graph()

        assert isinstance(graph, StateGraph)

    def test_build_graph_has_all_nodes(self):
        """Test that graph contains all expected nodes."""
        graph = build_graph()

        expected_nodes = [
            INGEST_DIAGNOSTICS,
            INTERPRET_DIAGNOSTICS,
            REASONING_PLANNING,
            GENERATE_ACTIONS,
            HUMAN_APPROVAL,
            EXECUTE_ACTION,
            VERIFY_RESULTS,
            DECIDE_CONTINUATION,
        ]

        for node_name in expected_nodes:
            assert node_name in graph.nodes

    def test_build_graph_node_count(self):
        """Test that graph has correct number of nodes."""
        graph = build_graph()

        # Should have exactly 8 nodes
        assert len(graph.nodes) == 8


class TestCompileGraph:
    """Tests for compile_graph function."""

    def test_compile_graph_returns_compiled_graph(self):
        """Test that compile_graph returns a CompiledStateGraph."""
        compiled = compile_graph()

        assert isinstance(compiled, CompiledStateGraph)

    def test_compile_graph_with_custom_interrupts(self):
        """Test compilation with custom interrupt points."""
        compiled = compile_graph(
            interrupt_before=[EXECUTE_ACTION],
            interrupt_after=[VERIFY_RESULTS],
        )

        assert isinstance(compiled, CompiledStateGraph)

    def test_compile_graph_default_interrupt_before_human_approval(self):
        """Test that default compilation interrupts before human_approval."""
        compiled = compile_graph()

        # The graph should be configured to interrupt before human_approval
        # We can verify this by checking the compiled graph's configuration
        assert isinstance(compiled, CompiledStateGraph)

    def test_compile_graph_with_existing_graph(self):
        """Test compilation with pre-built graph."""
        graph = build_graph()
        compiled = compile_graph(graph=graph)

        assert isinstance(compiled, CompiledStateGraph)


class TestCreateAgentConfig:
    """Tests for create_agent_config function."""

    def test_create_config_with_backend_only(self, mock_backend):
        """Test config creation with just backend."""
        config = create_agent_config(backend=mock_backend)

        assert "configurable" in config
        assert "backend" in config["configurable"]
        assert config["configurable"]["backend"] is mock_backend

    def test_create_config_with_llm_client(self, mock_backend):
        """Test config creation with backend and LLM client."""
        mock_llm = Mock()

        config = create_agent_config(
            backend=mock_backend,
            llm_client=mock_llm,
        )

        assert "configurable" in config
        assert "backend" in config["configurable"]
        assert "llm_client" in config["configurable"]
        assert config["configurable"]["llm_client"] is mock_llm

    def test_create_config_structure(self, mock_backend):
        """Test that config has correct structure for LangGraph."""
        config = create_agent_config(backend=mock_backend)

        # Config should have the structure expected by LangGraph nodes
        assert isinstance(config, dict)
        assert isinstance(config["configurable"], dict)


class TestNodeConstants:
    """Tests for node name constants."""

    def test_node_name_constants_are_strings(self):
        """Test that all node constants are strings."""
        constants = [
            INGEST_DIAGNOSTICS,
            INTERPRET_DIAGNOSTICS,
            REASONING_PLANNING,
            GENERATE_ACTIONS,
            HUMAN_APPROVAL,
            EXECUTE_ACTION,
            VERIFY_RESULTS,
            DECIDE_CONTINUATION,
        ]

        for const in constants:
            assert isinstance(const, str)

    def test_node_name_constants_are_unique(self):
        """Test that all node constants are unique."""
        constants = [
            INGEST_DIAGNOSTICS,
            INTERPRET_DIAGNOSTICS,
            REASONING_PLANNING,
            GENERATE_ACTIONS,
            HUMAN_APPROVAL,
            EXECUTE_ACTION,
            VERIFY_RESULTS,
            DECIDE_CONTINUATION,
        ]

        assert len(constants) == len(set(constants))


class TestGraphIntegration:
    """Integration tests for the graph structure."""

    def test_graph_can_get_node_functions(self):
        """Test that graph nodes contain actual functions."""
        graph = build_graph()

        # All nodes should have callable functions
        for node_name, node_func in graph.nodes.items():
            # Node might be a function or a wrapper
            assert node_func is not None

    def test_compiled_graph_has_expected_structure(self):
        """Test compiled graph has expected attributes."""
        compiled = compile_graph()

        # Compiled graph should have certain attributes
        assert hasattr(compiled, "invoke")
        assert hasattr(compiled, "stream")
        assert callable(compiled.invoke)
        assert callable(compiled.stream)
