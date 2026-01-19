"""Parsers for Tao output data."""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaoDataParser:
    """Parser for Tao command outputs.

    Converts Tao output strings into structured Python data.
    """

    @staticmethod
    def parse_lat_ele_list(output: str) -> List[Dict[str, Any]]:
        """Parse output from python lat_ele_list command.

        The output format is typically:
        ele_name;attribute1;attribute2;...
        value1;value2;...

        Args:
            output: Raw output from Tao command

        Returns:
            List of dictionaries with parsed data
        """
        if not output or output.strip() == "":
            return []

        try:
            lines = output.strip().split("\n")
            if len(lines) < 2:
                return []

            # Parse header line to get attribute names
            header = lines[0].strip()
            attributes = [attr.strip() for attr in header.split(";")]

            results = []
            for line in lines[1:]:
                if not line.strip():
                    continue

                values = [val.strip() for val in line.split(";")]
                if len(values) != len(attributes):
                    logger.warning(f"Mismatched attributes and values: {line}")
                    continue

                # Build dictionary
                data = {}
                for attr, val in zip(attributes, values):
                    # Try to convert to appropriate type
                    data[attr] = TaoDataParser._convert_value(val)

                results.append(data)

            return results

        except Exception as e:
            logger.error(f"Failed to parse lat_ele_list output: {e}")
            return []

    @staticmethod
    def parse_single_value(output: str, attribute: str) -> Optional[float]:
        """Parse a single numeric value from Tao output.

        Args:
            output: Raw output from Tao command
            attribute: Name of the attribute being parsed

        Returns:
            Parsed float value or None if parsing fails
        """
        try:
            # Try direct conversion first
            return float(output.strip())
        except ValueError:
            pass

        try:
            # Try to extract number from formatted output
            parsed = TaoDataParser.parse_lat_ele_list(output)
            if parsed and len(parsed) > 0:
                if attribute in parsed[0]:
                    return float(parsed[0][attribute])

            # Try regex extraction
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", output)
            if match:
                return float(match.group(0))

            logger.warning(f"Could not parse single value from: {output}")
            return None

        except Exception as e:
            logger.error(f"Failed to parse single value: {e}")
            return None

    @staticmethod
    def parse_orbit_data(output: str) -> Dict[str, Dict[str, float]]:
        """Parse orbit data from Tao output.

        Args:
            output: Raw output from orbit command

        Returns:
            Dictionary mapping element names to {x, y} orbit values
        """
        results = TaoDataParser.parse_lat_ele_list(output)
        orbit_data = {}

        for item in results:
            if "ele_name" in item:
                ele_name = item["ele_name"]
                orbit_data[ele_name] = {
                    "x": item.get("orbit_x", 0.0),
                    "y": item.get("orbit_y", 0.0),
                }

        return orbit_data

    @staticmethod
    def parse_beam_size_data(output: str) -> Dict[str, float]:
        """Parse beam size data from Tao output.

        Args:
            output: Raw output from beam size command

        Returns:
            Dictionary with sig_x and sig_y values
        """
        results = TaoDataParser.parse_lat_ele_list(output)
        if results and len(results) > 0:
            return {
                "sig_x": results[0].get("sig_x", 0.0),
                "sig_y": results[0].get("sig_y", 0.0),
            }
        return {"sig_x": 0.0, "sig_y": 0.0}

    @staticmethod
    def _convert_value(value_str: str) -> Any:
        """Convert string value to appropriate Python type.

        Args:
            value_str: String value to convert

        Returns:
            Converted value (float, int, str, or bool)
        """
        # Try boolean
        if value_str.lower() in ("true", "t", "yes", "y"):
            return True
        if value_str.lower() in ("false", "f", "no", "n"):
            return False

        # Try numeric
        try:
            # Try integer first
            if "." not in value_str and "e" not in value_str.lower():
                return int(value_str)
            # Try float
            return float(value_str)
        except ValueError:
            pass

        # Return as string
        return value_str
