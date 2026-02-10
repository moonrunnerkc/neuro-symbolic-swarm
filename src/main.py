# Author: Bradley R. Kinnard
"""Entry point: launches the swarm backend and PyQt6 UI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.swarm import SwarmChatbot
from src.ui.controller import MainWindow

# structured log format per project conventions
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def run() -> int:
    """initialize swarm + UI and enter the event loop."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger("swarm-chatbot")
    logger.info("starting swarm chatbot")

    # start the swarm backend
    swarm = SwarmChatbot()

    # launch the Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Swarm Chatbot")

    window = MainWindow(swarm=swarm)
    window.show()

    exit_code = app.exec()
    logger.info("shutdown complete (exit code %d)", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
