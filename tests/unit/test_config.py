"""Tests for configuration system."""

import pytest
from pathlib import Path

from accops_agent.config import (
    AcceleratorConfig,
    KnobDefinition,
    DiagnosticDefinition,
    ConstraintDefinition,
    load_accelerator_config,
)
from accops_agent.config.loader import ConfigLoadError


class TestKnobDefinition:
    """Tests for KnobDefinition model."""

    def test_valid_knob(self):
        """Test creating a valid knob definition."""
        knob = KnobDefinition(
            name="QF1_K1",
            description="Quadrupole strength",
            min_value=-5.0,
            max_value=5.0,
            unit="T/m",
            rate_limit=0.5,
            element_type="quadrupole",
        )
        assert knob.name == "QF1_K1"
        assert knob.min_value == -5.0
        assert knob.max_value == 5.0

    def test_invalid_max_less_than_min(self):
        """Test that max_value must be greater than min_value."""
        with pytest.raises(ValueError, match="max_value must be greater than min_value"):
            KnobDefinition(
                name="QF1_K1",
                description="Quadrupole strength",
                min_value=5.0,
                max_value=-5.0,  # Invalid: max < min
                unit="T/m",
                rate_limit=0.5,
                element_type="quadrupole",
            )


class TestDiagnosticDefinition:
    """Tests for DiagnosticDefinition model."""

    def test_valid_diagnostic(self):
        """Test creating a valid diagnostic definition."""
        diag = DiagnosticDefinition(
            name="BPM1_X",
            description="Beam position",
            measurement_type="orbit",
            unit="mm",
            nominal_value=0.0,
            tolerance=0.5,
            alarm_threshold=2.0,
        )
        assert diag.name == "BPM1_X"
        assert diag.measurement_type == "orbit"

    def test_alarm_threshold_validation(self):
        """Test that alarm_threshold should be >= tolerance."""
        with pytest.raises(ValueError, match="alarm_threshold should be >= tolerance"):
            DiagnosticDefinition(
                name="BPM1_X",
                description="Beam position",
                measurement_type="orbit",
                unit="mm",
                nominal_value=0.0,
                tolerance=0.5,
                alarm_threshold=0.2,  # Invalid: alarm < tolerance
            )


class TestAcceleratorConfig:
    """Tests for AcceleratorConfig model."""

    def test_valid_config(self):
        """Test creating a valid accelerator configuration."""
        config = AcceleratorConfig(
            name="Test Accelerator",
            description="Test description",
            knobs=[
                KnobDefinition(
                    name="QF1_K1",
                    description="Quadrupole strength",
                    min_value=-5.0,
                    max_value=5.0,
                    unit="T/m",
                    rate_limit=0.5,
                    element_type="quadrupole",
                )
            ],
            diagnostics=[
                DiagnosticDefinition(
                    name="BPM1_X",
                    description="Beam position",
                    measurement_type="orbit",
                    unit="mm",
                    nominal_value=0.0,
                    tolerance=0.5,
                    alarm_threshold=2.0,
                )
            ],
        )
        assert config.name == "Test Accelerator"
        assert len(config.knobs) == 1
        assert len(config.diagnostics) == 1

    def test_get_knob(self):
        """Test getting a knob by name."""
        config = AcceleratorConfig(
            name="Test Accelerator",
            description="Test description",
            knobs=[
                KnobDefinition(
                    name="QF1_K1",
                    description="Quadrupole strength",
                    min_value=-5.0,
                    max_value=5.0,
                    unit="T/m",
                    rate_limit=0.5,
                    element_type="quadrupole",
                )
            ],
            diagnostics=[
                DiagnosticDefinition(
                    name="BPM1_X",
                    description="Beam position",
                    measurement_type="orbit",
                    unit="mm",
                    nominal_value=0.0,
                    tolerance=0.5,
                    alarm_threshold=2.0,
                )
            ],
        )
        knob = config.get_knob("QF1_K1")
        assert knob is not None
        assert knob.name == "QF1_K1"

        # Test non-existent knob
        assert config.get_knob("NONEXISTENT") is None

    def test_duplicate_knob_names(self):
        """Test that duplicate knob names are not allowed."""
        with pytest.raises(ValueError, match="Knob names must be unique"):
            AcceleratorConfig(
                name="Test Accelerator",
                description="Test description",
                knobs=[
                    KnobDefinition(
                        name="QF1_K1",
                        description="Quadrupole strength",
                        min_value=-5.0,
                        max_value=5.0,
                        unit="T/m",
                        rate_limit=0.5,
                        element_type="quadrupole",
                    ),
                    KnobDefinition(
                        name="QF1_K1",  # Duplicate name
                        description="Quadrupole strength 2",
                        min_value=-5.0,
                        max_value=5.0,
                        unit="T/m",
                        rate_limit=0.5,
                        element_type="quadrupole",
                    ),
                ],
                diagnostics=[
                    DiagnosticDefinition(
                        name="BPM1_X",
                        description="Beam position",
                        measurement_type="orbit",
                        unit="mm",
                        nominal_value=0.0,
                        tolerance=0.5,
                        alarm_threshold=2.0,
                    )
                ],
            )


class TestConfigLoader:
    """Tests for configuration loader."""

    def test_load_example_config(self):
        """Test loading the example LINAC configuration."""
        # Get path to example config
        config_path = Path(__file__).parent.parent.parent / "configs" / "accelerators" / "example_linac.yaml"

        config = load_accelerator_config(config_path)

        assert config.name == "Example Linear Accelerator"
        assert len(config.knobs) > 0
        assert len(config.diagnostics) > 0
        assert len(config.constraints) > 0

    def test_load_nonexistent_file(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(ConfigLoadError, match="Configuration file not found"):
            load_accelerator_config("nonexistent.yaml")

    def test_get_diagnostic(self):
        """Test getting a diagnostic by name from loaded config."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "accelerators" / "example_linac.yaml"
        config = load_accelerator_config(config_path)

        diag = config.get_diagnostic("BPM1_X")
        assert diag is not None
        assert diag.measurement_type == "orbit"

    def test_get_constraint(self):
        """Test getting a constraint by ID from loaded config."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "accelerators" / "example_linac.yaml"
        config = load_accelerator_config(config_path)

        constraint = config.get_constraint("global_rate_limit")
        assert constraint is not None
        assert constraint.constraint_type == "rate_limit"
