# Author: Bradley R. Kinnard
"""Central chat display, input widget, and live pipeline monitor."""

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
    STATUS_ACTIVE,
    STATUS_ERROR,
    STATUS_WARN,
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

        # -- live pipeline monitor (hidden by default) --
        # scrolling log that shows real-time agent activity so the user
        # sees "work happening" instead of staring at a spinner
        self._pipeline_monitor = QWidget()
        self._pipeline_monitor.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_TERTIARY};
                border-top: 1px solid {BORDER_ACCENT};
                border-bottom: 1px solid {BORDER_ACCENT};
            }}
        """)
        monitor_layout = QVBoxLayout(self._pipeline_monitor)
        monitor_layout.setContentsMargins(0, 0, 0, 0)
        monitor_layout.setSpacing(0)

        # header row: animated dots + current phase label
        monitor_header = QWidget()
        monitor_header.setFixedHeight(28)
        monitor_header.setStyleSheet(f"""
            QWidget {{
                background-color: {ACCENT_FAINT};
                border: none;
                border-bottom: 1px solid {BORDER_ACCENT};
            }}
        """)
        header_layout = QHBoxLayout(monitor_header)
        header_layout.setContentsMargins(12, 2, 12, 2)
        header_layout.setSpacing(8)

        self._thinking_dots = QLabel()
        self._thinking_dots.setFixedWidth(24)
        self._thinking_dots.setStyleSheet(
            f"color: {ACCENT}; font-size: 14px; font-weight: bold; "
            f"background: transparent; border: none;"
        )
        header_layout.addWidget(self._thinking_dots)

        self._monitor_title = QLabel("PIPELINE ACTIVE")
        self._monitor_title.setStyleSheet(
            f"color: {ACCENT}; font-size: {FONT_SIZE_SMALL}px; "
            f"font-weight: bold; letter-spacing: 1px; "
            f"background: transparent; border: none;"
        )
        header_layout.addWidget(self._monitor_title)
        header_layout.addStretch()

        self._thinking_stage = QLabel("")
        self._thinking_stage.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SIZE_TINY}px; "
            f"background: transparent; border: none;"
        )
        header_layout.addWidget(self._thinking_stage)

        monitor_layout.addWidget(monitor_header)

        # scrolling log area
        self._monitor_log = QTextEdit()
        self._monitor_log.setReadOnly(True)
        self._monitor_log.setFixedHeight(140)
        self._monitor_log.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_DEEP};
                color: {TEXT_SECONDARY};
                border: none;
                padding: 8px 12px;
                font-size: {FONT_SIZE_SMALL}px;
                font-family: "Roboto Mono", "Consolas", monospace;
            }}
        """)
        monitor_layout.addWidget(self._monitor_log)

        self._pipeline_monitor.setVisible(False)
        layout.addWidget(self._pipeline_monitor)

        # dot animation timer
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.setInterval(400)
        self._dot_timer.timeout.connect(self._animate_dots)

        # input bar at bottom
        input_container = QWidget()
        input_container.setObjectName("inputBar")
        input_container.setFixedHeight(64)
        input_container.setStyleSheet(f"""
            QWidget#inputBar {{
                background-color: {BG_SECONDARY};
                border-top: 1px solid {BORDER_LIGHT};
            }}
        """)
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(16, 12, 16, 12)
        input_layout.setSpacing(12)

        self._input_field = QLineEdit()
        self._input_field.setPlaceholderText("enter a query...")
        self._input_field.setMinimumHeight(38)
        self._input_field.returnPressed.connect(self._on_submit)
        self._input_field.textChanged.connect(self._on_text_changed)
        input_layout.addWidget(self._input_field, stretch=1)

        self._send_btn = QPushButton("Submit Query")
        self._send_btn.setObjectName("sendButton")
        self._send_btn.setFixedHeight(38)
        self._send_btn.setMinimumWidth(140)
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
        "><span style="color: {ACCENT}; font-size: {FONT_SIZE_SMALL}px;">[{_escape(source)}]</span><br>{_format_message(text)}</div>
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
            self._send_btn.setText("Submit Query")
        else:
            self._send_btn.setEnabled(False)
            self._send_btn.setText("working...")

    def clear_chat(self) -> None:
        self._chat_display.clear()

    def _defer_scroll(self) -> None:
        QTimer.singleShot(50, self._scroll_bottom)

    def _scroll_bottom(self) -> None:
        sb = self._chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    # -- pipeline monitor --

    # color map for inline status formatting
    _STATUS_COLORS = {
        "active": STATUS_ACTIVE,
        "idle": TEXT_MUTED,
        "error": STATUS_ERROR,
        "gated": STATUS_WARN,
        "responded": ACCENT,
        "blocked": STATUS_ERROR,
        "rejected": STATUS_ERROR,
        "passed": STATUS_ACTIVE,
        "locked": STATUS_WARN,
    }

    def show_thinking(self, stage: str = "processing") -> None:
        """reveal the pipeline monitor and start streaming stages."""
        self._monitor_log.clear()
        self._thinking_stage.setText(stage)
        self._dot_count = 0
        self._thinking_dots.setText("\u25CF")
        self._pipeline_monitor.setVisible(True)
        self._dot_timer.start()
        self._append_monitor_line(stage, "phase")

    def update_thinking_stage(self, stage: str) -> None:
        """append a new stage line to the scrolling monitor."""
        self._thinking_stage.setText(stage)
        self._append_monitor_line(stage, "phase")

    def update_agent_status(self, role: str, status: str, detail: str = "") -> None:
        """show a formatted agent status cue in the pipeline monitor.

        renders as: [Parser: ACTIVE]  or  [Critic: REJECTED DRAFT 1]
        """
        status_upper = status.upper()
        color = self._STATUS_COLORS.get(status.lower(), TEXT_SECONDARY)
        extra = f" {_escape(detail)}" if detail else ""
        html = (
            f'<span style="color:{TEXT_MUTED};">\u2502</span> '
            f'<span style="color:{ACCENT};">[{_escape(role)}:</span> '
            f'<span style="color:{color}; font-weight:bold;">{_escape(status_upper)}</span>'
            f'<span style="color:{ACCENT};">]</span>'
            f'<span style="color:{TEXT_SECONDARY};">{extra}</span>'
        )
        self._monitor_log.append(html)
        self._scroll_monitor()

    def update_ledger_status(self, action: str, detail: str = "") -> None:
        """show a ledger event: [Ledger: LOCKED], [Ledger: ROLLBACK], etc."""
        color = self._STATUS_COLORS.get(action.lower(), STATUS_WARN)
        extra = f" — {_escape(detail)}" if detail else ""
        html = (
            f'<span style="color:{TEXT_MUTED};">\u2502</span> '
            f'<span style="color:{STATUS_WARN};">[Ledger:</span> '
            f'<span style="color:{color}; font-weight:bold;">{_escape(action.upper())}</span>'
            f'<span style="color:{STATUS_WARN};">]</span>'
            f'<span style="color:{TEXT_SECONDARY};">{extra}</span>'
        )
        self._monitor_log.append(html)
        self._scroll_monitor()

    def hide_thinking(self) -> None:
        """collapse the pipeline monitor after processing completes."""
        self._dot_timer.stop()
        # brief delay before hiding so the user sees the final state
        QTimer.singleShot(600, lambda: self._pipeline_monitor.setVisible(False))

    def _append_monitor_line(self, text: str, kind: str = "info") -> None:
        """add a timestamped line to the scrolling monitor."""
        if kind == "phase":
            html = (
                f'<span style="color:{ACCENT};">\u25B8</span> '
                f'<span style="color:{TEXT_PRIMARY};">{_escape(text)}</span>'
            )
        else:
            html = (
                f'<span style="color:{TEXT_MUTED};">\u2502</span> '
                f'<span style="color:{TEXT_SECONDARY};">{_escape(text)}</span>'
            )
        self._monitor_log.append(html)
        self._scroll_monitor()

    def _scroll_monitor(self) -> None:
        """keep the monitor scrolled to the latest entry."""
        sb = self._monitor_log.verticalScrollBar()
        sb.setValue(sb.maximum())

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


def _format_message(text: str) -> str:
    """render markdown code fences as styled <pre> blocks, escape the rest.

    handles ```lang ... ``` fenced blocks and `inline code` spans.
    uses <table> for code blocks because QTextEdit chokes on nested divs.
    also catches unclosed fences (opening ``` without closing ```).
    """
    import re

    code_bg = "#0D0D18"
    code_border = "#2A2A40"
    code_text = "#A8D8A8"
    label_color = "#5A6270"

    # match fenced code blocks — closed or unclosed at end of string
    fence_re = re.compile(r"```(\w*)\n(.*?)(?:```|$)", flags=re.DOTALL)

    parts: list[str] = []
    last_end = 0

    for m in fence_re.finditer(text):
        before = text[last_end:m.start()]
        parts.append(_escape(before))

        lang = m.group(1) or ""
        code = m.group(2).rstrip("\n")
        escaped_code = (
            code.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        lang_html = (
            f'<span style="color:{label_color}; font-size:10px;">{lang}</span><br>'
            if lang else ""
        )
        # use <table> with a single cell — QTextEdit renders this
        # far more reliably than nested <div> elements
        parts.append(
            f'<br><table cellpadding="8" cellspacing="0" '
            f'style="background-color:{code_bg}; '
            f'border:1px solid {code_border}; '
            f'border-radius:4px; margin:4px 0;" width="100%">'
            f'<tr><td>{lang_html}'
            f'<pre style="margin:0; white-space:pre-wrap; '
            f'font-family:Roboto Mono,Consolas,monospace; '
            f'font-size:12px; color:{code_text}; '
            f'background-color:{code_bg};">'
            f'{escaped_code}</pre></td></tr></table><br>'
        )
        last_end = m.end()

    parts.append(_escape(text[last_end:]))
    result = "".join(parts)

    # inline code: `...`
    result = re.sub(
        r"`([^`]+)`",
        lambda m: (
            f'<span style="background-color:{code_bg}; '
            f'padding:1px 4px; font-family:Roboto Mono,Consolas,monospace; '
            f'font-size:12px; color:{code_text};">'
            f'{m.group(1)}</span>'
        ),
        result,
    )
    return result
