# Author: Bradley R. Kinnard
"""Central chat display and input widget."""

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui.theme import (
    ACCENT,
    ACCENT_DIM,
    ACCENT_FAINT,
    BG_DEEP,
    BG_PRIMARY,
    BG_SECONDARY,
    BG_TERTIARY,
    BORDER,
    BORDER_ACCENT,
    BORDER_LIGHT,
    FONT_SIZE,
    FONT_SIZE_SMALL,
    FONT_SIZE_TINY,
    SEND_BG,
    SWARM_BUBBLE,
    SWARM_BUBBLE_BORDER,
    TEXT_BRIGHT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    USER_BUBBLE,
    USER_BUBBLE_BORDER,
)


class ChatArea(QWidget):
    """main chat display with message bubbles and input bar."""

    query_submitted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # chat history area
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_DEEP};
                color: {TEXT_PRIMARY};
                border: none;
                padding: 16px;
                font-size: {FONT_SIZE}px;
            }}
        """)
        layout.addWidget(self._chat_display, stretch=1)

        # thinking indicator bar (hidden by default)
        self._thinking_bar = QWidget()
        self._thinking_bar.setFixedHeight(36)
        self._thinking_bar.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_TERTIARY};
                border-top: 1px solid {BORDER_ACCENT};
                border-bottom: 1px solid {BORDER_ACCENT};
            }}
        """)
        thinking_layout = QHBoxLayout(self._thinking_bar)
        thinking_layout.setContentsMargins(16, 4, 16, 4)
        thinking_layout.setSpacing(8)

        self._thinking_dots = QLabel()
        self._thinking_dots.setFixedWidth(30)
        self._thinking_dots.setStyleSheet(
            f"color: {ACCENT}; font-size: 16px; font-weight: bold; "
            f"background: transparent; border: none;"
        )
        thinking_layout.addWidget(self._thinking_dots)

        self._thinking_stage = QLabel("thinking")
        self._thinking_stage.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SIZE_SMALL}px; "
            f"background: transparent; border: none;"
        )
        thinking_layout.addWidget(self._thinking_stage, stretch=1)

        self._thinking_bar.setVisible(False)
        layout.addWidget(self._thinking_bar)

        # dot animation timer
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.setInterval(400)
        self._dot_timer.timeout.connect(self._animate_dots)

        # input bar at bottom
        input_container = QWidget()
        input_container.setFixedHeight(64)
        input_container.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_SECONDARY};
                border-top: 1px solid {BORDER_LIGHT};
            }}
        """)
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(16, 12, 16, 12)
        input_layout.setSpacing(12)

        self._input_field = QLineEdit()
        self._input_field.setPlaceholderText("type a message...")
        self._input_field.setMinimumHeight(38)
        self._input_field.returnPressed.connect(self._on_submit)
        self._input_field.textChanged.connect(self._on_text_changed)
        input_layout.addWidget(self._input_field, stretch=1)

        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("sendButton")
        self._send_btn.setFixedSize(100, 38)
        self._send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_btn.setEnabled(False)  # starts disabled until text entered
        self._send_btn.clicked.connect(self._on_submit)
        input_layout.addWidget(self._send_btn)

        layout.addWidget(input_container)

    def _on_submit(self) -> None:
        text = self._input_field.text().strip()
        if not text:
            return
        self._input_field.clear()
        self.add_user_message(text)
        self.query_submitted.emit(text)

    def _on_text_changed(self, text: str) -> None:
        """enable send button only when there's actual text."""
        has_text = bool(text.strip())
        self._send_btn.setEnabled(has_text)

    def add_user_message(self, text: str) -> None:
        """user message: left-aligned, subtle dark bubble."""
        html = f"""
        <div style="
            background-color: {USER_BUBBLE};
            color: {TEXT_PRIMARY};
            border: 1px solid {USER_BUBBLE_BORDER};
            border-radius: 10px;
            padding: 10px 14px;
            margin: 6px 160px 6px 8px;
            font-size: {FONT_SIZE}px;
            text-align: left;
        "><span style="color: {TEXT_MUTED}; font-size: {FONT_SIZE_SMALL}px;">you</span><br>{_escape(text)}</div>
        """
        self._chat_display.append(html)
        self._defer_scroll()

    def add_swarm_message(self, text: str, source: str = "swarm") -> None:
        """swarm response: left-aligned text in blue-tinted bubble."""
        html = f"""
        <div style="
            background-color: {SWARM_BUBBLE};
            color: {TEXT_PRIMARY};
            border: 1px solid {SWARM_BUBBLE_BORDER};
            border-radius: 10px;
            padding: 10px 14px;
            margin: 6px 8px 6px 160px;
            font-size: {FONT_SIZE}px;
            text-align: left;
        "><span style="color: {ACCENT}; font-size: {FONT_SIZE_SMALL}px;">[{_escape(source)}]</span><br>{_escape(text)}</div>
        """
        self._chat_display.append(html)
        self._defer_scroll()

    def add_system_message(self, text: str) -> None:
        html = f"""
        <div style="
            color: {TEXT_MUTED};
            text-align: left;
            margin: 10px 60px 10px 8px;
            font-size: {FONT_SIZE_SMALL}px;
        ">{_escape(text)}</div>
        """
        self._chat_display.append(html)
        self._defer_scroll()

    def set_input_enabled(self, enabled: bool) -> None:
        self._input_field.setEnabled(enabled)
        if enabled:
            # re-check: only enable send if there's text
            has_text = bool(self._input_field.text().strip())
            self._send_btn.setEnabled(has_text)
            self._send_btn.setText("Send")
        else:
            self._send_btn.setEnabled(False)
            self._send_btn.setText("thinking...")

    def clear_chat(self) -> None:
        self._chat_display.clear()

    def _defer_scroll(self) -> None:
        QTimer.singleShot(50, self._scroll_bottom)

    def _scroll_bottom(self) -> None:
        sb = self._chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    # -- thinking indicator --

    def show_thinking(self, stage: str = "thinking") -> None:
        """show the animated thinking bar with a stage label."""
        self._thinking_stage.setText(stage)
        self._dot_count = 0
        self._thinking_dots.setText("\u2022")
        self._thinking_bar.setVisible(True)
        self._dot_timer.start()

    def update_thinking_stage(self, stage: str) -> None:
        """update the stage text while thinking bar is visible."""
        self._thinking_stage.setText(stage)

    def hide_thinking(self) -> None:
        """hide the thinking bar and stop animation."""
        self._dot_timer.stop()
        self._thinking_bar.setVisible(False)

    def _animate_dots(self) -> None:
        """cycle through dot animation frames."""
        self._dot_count = (self._dot_count + 1) % 4
        dots = "\u2022" * (self._dot_count + 1)
        self._thinking_dots.setText(dots)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
