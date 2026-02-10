# Author: Bradley R. Kinnard
"""Reusable agent info card for the left sidebar."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from src.ui.theme import (
    ACCENT,
    BG_ELEVATED,
    BG_TERTIARY,
    BORDER,
    BORDER_LIGHT,
    FONT_SIZE,
    FONT_SIZE_SMALL,
    STATUS_ACTIVE,
    STATUS_ERROR,
    STATUS_IDLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class AgentCard(QFrame):
    """compact card showing agent role, model, and status."""

    def __init__(self, role: str, model: str, status: str = "idle", parent=None):
        super().__init__(parent)
        self._role = role
        self._model = model
        self._status = status
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(72)
        self.setMaximumHeight(90)
        self._apply_card_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # top row: role name + status badge
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self._role_label = QLabel(self._role)
        self._role_label.setStyleSheet(
            f"color: {ACCENT}; font-weight: bold; font-size: {FONT_SIZE}px; "
            f"background: transparent; border: none;"
        )
        top_row.addWidget(self._role_label)
        top_row.addStretch()

        self._status_badge = QLabel()
        self._refresh_status_badge()
        top_row.addWidget(self._status_badge)
        layout.addLayout(top_row)

        # model name -- clearly readable
        self._model_label = QLabel(self._model)
        self._model_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SIZE_SMALL}px; "
            f"background: transparent; border: none;"
        )
        layout.addWidget(self._model_label)

        # log / activity line
        self._log_label = QLabel("idle")
        self._log_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: {FONT_SIZE_SMALL}px; "
            f"background: transparent; border: none;"
        )
        self._log_label.setWordWrap(True)
        layout.addWidget(self._log_label)

    def _apply_card_style(self) -> None:
        self.setStyleSheet(f"""
            AgentCard {{
                background-color: {BG_TERTIARY};
                border: 1px solid {BORDER_LIGHT};
                border-radius: 8px;
            }}
            AgentCard:hover {{
                background-color: {BG_ELEVATED};
                border-color: {ACCENT};
            }}
        """)

    def update_status(self, status: str) -> None:
        self._status = status
        self._refresh_status_badge()

    def update_log(self, text: str) -> None:
        display = text[:60] if len(text) <= 60 else text[:57] + "..."
        self._log_label.setText(display)
        self._log_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SIZE_SMALL}px; "
            f"background: transparent; border: none;"
        )

    def _refresh_status_badge(self) -> None:
        """colored text badge with dot indicator."""
        configs = {
            "active": (STATUS_ACTIVE, "\u25cf ACTIVE"),
            "idle": (STATUS_IDLE, "\u25cb idle"),
            "error": (STATUS_ERROR, "\u25cf ERROR"),
            "stopped": (STATUS_ERROR, "\u25cf STOP"),
        }
        color, text = configs.get(self._status, (STATUS_IDLE, "\u25cb idle"))
        self._status_badge.setText(text)
        self._status_badge.setStyleSheet(
            f"color: {color}; font-size: {FONT_SIZE_SMALL}px; "
            f"font-weight: bold; background: transparent; border: none;"
        )
        self._status_badge.setToolTip(self._status)
