"""Unit tests for PyTao components (parser, commands) without requiring Tao."""

import pytest

from accops_agent.backends.pytao.parser import TaoDataParser
from accops_agent.backends.pytao.commands import (
    build_read_data_command,
    build_set_parameter_command,
    build_get_parameter_command,
    build_run_calculation_command,
    build_read_floor_coordinates,
    build_read_orbit_command,
    build_read_beam_size_command,
    build_read_twiss_command,
)


class TestTaoDataParser:
    """Tests for TaoDataParser without requiring Tao connection."""

    def test_parse_lat_ele_list_empty(self):
        """Test parsing empty output."""
        parser = TaoDataParser()
        result = parser.parse_lat_ele_list("")
        assert result == []

    def test_parse_lat_ele_list_whitespace(self):
        """Test parsing whitespace-only output."""
        parser = TaoDataParser()
        result = parser.parse_lat_ele_list("   \n\n  ")
        assert result == []

    def test_parse_lat_ele_list_simple(self):
        """Test parsing simple lat_ele_list output."""
        parser = TaoDataParser()
        output = "ele_name;k1;s\nQF1;2.5;10.0\nQD1;-2.0;20.0"
        result = parser.parse_lat_ele_list(output)

        assert len(result) == 2
        assert result[0]["ele_name"] == "QF1"
        assert result[0]["k1"] == 2.5
        assert result[0]["s"] == 10.0
        assert result[1]["ele_name"] == "QD1"
        assert result[1]["k1"] == -2.0
        assert result[1]["s"] == 20.0

    def test_parse_lat_ele_list_with_strings(self):
        """Test parsing output with mixed types."""
        parser = TaoDataParser()
        output = "ele_name;type;k1;active\nQF1;quadrupole;2.5;True\nBEND1;sbend;0.1;False"
        result = parser.parse_lat_ele_list(output)

        assert len(result) == 2
        assert result[0]["ele_name"] == "QF1"
        assert result[0]["type"] == "quadrupole"
        assert result[0]["k1"] == 2.5
        assert result[0]["active"] is True
        assert result[1]["active"] is False

    def test_parse_lat_ele_list_mismatched_columns(self):
        """Test handling of mismatched column counts."""
        parser = TaoDataParser()
        output = "ele_name;k1;s\nQF1;2.5;10.0\nQD1;-2.0"  # Missing value
        result = parser.parse_lat_ele_list(output)

        # Should only get the valid row
        assert len(result) == 1
        assert result[0]["ele_name"] == "QF1"

    def test_parse_single_value_direct_float(self):
        """Test parsing direct float value."""
        parser = TaoDataParser()
        result = parser.parse_single_value("3.14159", "test")
        assert result == pytest.approx(3.14159)

    def test_parse_single_value_scientific_notation(self):
        """Test parsing scientific notation."""
        parser = TaoDataParser()
        result = parser.parse_single_value("1.23e-5", "test")
        assert result == pytest.approx(1.23e-5)

        result = parser.parse_single_value("-4.56E+10", "test")
        assert result == pytest.approx(-4.56e10)

    def test_parse_single_value_negative(self):
        """Test parsing negative values."""
        parser = TaoDataParser()
        result = parser.parse_single_value("-123.456", "test")
        assert result == pytest.approx(-123.456)

    def test_parse_single_value_from_formatted(self):
        """Test parsing single value from formatted output."""
        parser = TaoDataParser()
        output = "ele_name;k1\nQF1;2.5"
        result = parser.parse_single_value(output, "k1")
        assert result == 2.5

    def test_parse_single_value_invalid_returns_none(self):
        """Test that invalid input returns None."""
        parser = TaoDataParser()
        result = parser.parse_single_value("not_a_number", "test")
        assert result is None

    def test_parse_orbit_data(self):
        """Test parsing orbit data."""
        parser = TaoDataParser()
        output = "ele_name;orbit_x;orbit_y\nBPM1;0.5;-0.3\nBPM2;1.2;0.8"
        result = parser.parse_orbit_data(output)

        assert len(result) == 2
        assert "BPM1" in result
        assert result["BPM1"]["x"] == 0.5
        assert result["BPM1"]["y"] == -0.3
        assert "BPM2" in result
        assert result["BPM2"]["x"] == 1.2
        assert result["BPM2"]["y"] == 0.8

    def test_parse_beam_size_data(self):
        """Test parsing beam size data."""
        parser = TaoDataParser()
        output = "ele_name;sig_x;sig_y\nEND;0.001;0.0015"
        result = parser.parse_beam_size_data(output)

        assert result["sig_x"] == 0.001
        assert result["sig_y"] == 0.0015

    def test_parse_beam_size_data_empty(self):
        """Test parsing empty beam size data returns defaults."""
        parser = TaoDataParser()
        result = parser.parse_beam_size_data("")

        assert result["sig_x"] == 0.0
        assert result["sig_y"] == 0.0

    def test_convert_value_boolean_true(self):
        """Test boolean conversion for true values."""
        parser = TaoDataParser()
        assert parser._convert_value("True") is True
        assert parser._convert_value("true") is True
        assert parser._convert_value("T") is True
        assert parser._convert_value("t") is True
        assert parser._convert_value("yes") is True
        assert parser._convert_value("Y") is True

    def test_convert_value_boolean_false(self):
        """Test boolean conversion for false values."""
        parser = TaoDataParser()
        assert parser._convert_value("False") is False
        assert parser._convert_value("false") is False
        assert parser._convert_value("F") is False
        assert parser._convert_value("no") is False
        assert parser._convert_value("N") is False

    def test_convert_value_integer(self):
        """Test integer conversion."""
        parser = TaoDataParser()
        result = parser._convert_value("42")
        assert result == 42
        assert isinstance(result, int)

        result = parser._convert_value("-123")
        assert result == -123
        assert isinstance(result, int)

    def test_convert_value_float(self):
        """Test float conversion."""
        parser = TaoDataParser()
        result = parser._convert_value("3.14")
        assert result == 3.14
        assert isinstance(result, float)

        result = parser._convert_value("1.23e-5")
        assert result == pytest.approx(1.23e-5)
        assert isinstance(result, float)

    def test_convert_value_string(self):
        """Test string fallback."""
        parser = TaoDataParser()
        result = parser._convert_value("hello_world")
        assert result == "hello_world"
        assert isinstance(result, str)

        result = parser._convert_value("QF1")
        assert result == "QF1"
        assert isinstance(result, str)


class TestTaoCommands:
    """Tests for Tao command builders."""

    def test_build_read_data_command(self):
        """Test building read data command."""
        cmd = build_read_data_command("QF1", "k1")
        assert "python lat_ele_list" in cmd
        assert "QF1" in cmd
        assert "k1" in cmd
        assert "-attribute" in cmd

    def test_build_read_data_command_multiple_elements(self):
        """Test command structure with different elements."""
        cmd1 = build_read_data_command("BPM1", "orbit_x")
        cmd2 = build_read_data_command("BEND1", "angle")

        assert "BPM1" in cmd1
        assert "orbit_x" in cmd1
        assert "BEND1" in cmd2
        assert "angle" in cmd2

    def test_build_set_parameter_command(self):
        """Test building set parameter command."""
        cmd = build_set_parameter_command("QF1", "k1", 2.5)
        assert "set ele" in cmd
        assert "QF1" in cmd
        assert "k1" in cmd
        assert "2.5" in cmd
        assert "=" in cmd

    def test_build_set_parameter_command_negative_value(self):
        """Test set command with negative value."""
        cmd = build_set_parameter_command("QD1", "k1", -1.5)
        assert "QD1" in cmd
        assert "-1.5" in cmd

    def test_build_set_parameter_command_zero(self):
        """Test set command with zero value."""
        cmd = build_set_parameter_command("HCOR1", "kick", 0.0)
        assert "HCOR1" in cmd
        assert "0.0" in cmd

    def test_build_get_parameter_command(self):
        """Test building get parameter command."""
        cmd = build_get_parameter_command("QF1", "k1")
        assert "python lat_ele_list" in cmd
        assert "QF1" in cmd
        assert "k1" in cmd

    def test_build_run_calculation_command(self):
        """Test building calculation command."""
        cmd = build_run_calculation_command()
        assert "set global" in cmd
        assert "track_type" in cmd

    def test_build_read_floor_coordinates(self):
        """Test building floor coordinates command."""
        cmd = build_read_floor_coordinates("QF1")
        assert "QF1" in cmd
        assert "x" in cmd.lower()
        assert "y" in cmd.lower()
        assert "z" in cmd.lower()

    def test_build_read_orbit_command(self):
        """Test building orbit read command."""
        cmd = build_read_orbit_command(["BPM1", "BPM2", "BPM3"])
        assert "orbit" in cmd.lower()
        # Command should be general enough to read all BPMs

    def test_build_read_beam_size_command(self):
        """Test building beam size command."""
        cmd = build_read_beam_size_command("END")
        assert "END" in cmd
        assert "sig" in cmd.lower()

    def test_build_read_twiss_command(self):
        """Test building Twiss parameters command."""
        cmd = build_read_twiss_command("QF1")
        assert "QF1" in cmd
        assert "beta" in cmd.lower()
        assert "alpha" in cmd.lower()


class TestTaoConnectionInterface:
    """Test TaoConnection interface without requiring Tao."""

    def test_import_tao_connection(self):
        """Test that TaoConnection can be imported."""
        from accops_agent.backends.pytao.connection import TaoConnection

        assert TaoConnection is not None

    def test_tao_connection_initialization_without_init_file(self):
        """Test TaoConnection initialization."""
        from accops_agent.backends.pytao.connection import TaoConnection

        # TaoConnection raises ImportError if pytao not available
        try:
            conn = TaoConnection()
            assert conn is not None
            assert not conn.is_connected()
        except ImportError:
            # Expected if pytao not fully available
            pytest.skip("pytao not fully available")

    def test_tao_connection_initialization_with_init_file(self):
        """Test TaoConnection initialization with init file."""
        from accops_agent.backends.pytao.connection import TaoConnection

        # TaoConnection raises ImportError if pytao not available
        try:
            conn = TaoConnection(init_file="/path/to/init.tao")
            assert conn is not None
            assert conn.init_file == "/path/to/init.tao"
            assert not conn.is_connected()
        except ImportError:
            # Expected if pytao not fully available
            pytest.skip("pytao not fully available")


class TestTaoBackendInterface:
    """Test TaoBackend interface without requiring Tao connection."""

    def test_import_tao_backend(self):
        """Test that TaoBackend can be imported."""
        from accops_agent.backends.pytao import TaoBackend

        assert TaoBackend is not None

    def test_tao_backend_has_required_methods(self):
        """Test TaoBackend has all required interface methods."""
        from accops_agent.backends.pytao import TaoBackend

        required_methods = [
            "initialize",
            "shutdown",
            "is_connected",
            "read_diagnostic",
            "read_all_diagnostics",
            "get_parameter",
            "set_parameter",
            "get_all_parameters",
            "run_calculation",
            "validate_parameter_value",
            "determine_diagnostic_status",
        ]

        for method in required_methods:
            assert hasattr(TaoBackend, method), f"Missing method: {method}"
