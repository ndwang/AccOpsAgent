"""Tao connection management for the MCP server process."""

import logging
from pathlib import Path
from typing import Optional

try:
    from pytao import Tao
except ImportError:
    # Allow module to load even if pytao is not installed
    Tao = None

logger = logging.getLogger(__name__)


class TaoConnection:
    """Manages Tao instance lifecycle and connection state."""

    def __init__(self, init_file: Optional[str] = None):
        """Initialize Tao connection manager.

        Args:
            init_file: Path to Tao initialization file (optional)
        """
        if Tao is None:
            raise ImportError(
                "pytao is not installed. Install it with: uv add pytao"
            )

        self.init_file = init_file
        self.tao: Optional[Tao] = None
        self._connected = False
        self.last_error: Optional[str] = None

    def connect(self) -> bool:
        """Establish connection to Tao."""
        self.last_error = None
        try:
            if self.init_file:
                # Verify init file exists
                init_path = Path(self.init_file)
                if not init_path.exists():
                    self.last_error = f"Init file not found: {self.init_file}"
                    logger.error(self.last_error)
                    return False

                # Initialize Tao with init file
                logger.info(f"Initializing Tao with init file: {self.init_file}")
                self.tao = Tao(init_file=str(init_path))
            else:
                # Initialize Tao without init file
                logger.info("Initializing Tao without init file")
                self.tao = Tao()

            self._connected = True
            logger.info("Tao connection established")
            return True

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to connect to Tao: {e}")
            self._connected = False
            return False

    def disconnect(self) -> bool:
        """Close Tao connection."""
        try:
            if self.tao is not None:
                # Tao cleanup (if needed)
                self.tao = None
                self._connected = False
                logger.info("Tao connection closed")
            return True

        except Exception as e:
            logger.error(f"Error during Tao disconnection: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._connected and self.tao is not None

    def execute_command(self, command: str) -> str:
        """Execute a Tao command."""
        if not self.is_connected():
            raise RuntimeError("Not connected to Tao")

        try:
            logger.debug(f"Executing Tao command: {command}")
            result = self.tao.cmd(command)
            return result

        except Exception as e:
            logger.error(f"Tao command failed: {command} - {e}")
            raise RuntimeError(f"Tao command execution failed: {e}") from e
