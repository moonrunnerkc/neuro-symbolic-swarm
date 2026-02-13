# Author: Bradley R. Kinnard
"""Right sidebar: neuro-symbolic diagnostics panel."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui.theme import (
    ACCENT,
    ACCENT_DIM,
    ACCENT_FAINT,
    BG_DEEP,
    BG_SECONDARY,
    BG_TERTIARY,
    BORDER,
    BORDER_LIGHT,
    FONT_SIZE,
    FONT_SIZE_SMALL,
    STATUS_ACTIVE,
    STATUS_ERROR,
    STATUS_WARN,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class SidebarRight(QWidget):
    """right panel showing live system diagnostics and proof of work."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {BG_SECONDARY};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        header = QLabel("Diagnostics")
        header.setObjectName("header")
        layout.addWidget(header)

        # -- active agents group --
        agents_group = QGroupBox("Active Agents")
        agents_layout = QVBoxLayout(agents_group)
        agents_layout.setSpacing(4)

        self._agent_rows: dict[str, QLabel] = {}
        self._agents_container = QVBoxLayout()
        agents_layout.addLayout(self._agents_container)
        layout.addWidget(agents_group)

        # -- state ledger group --
        ledger_group = QGroupBox("State Ledger")
        ledger_layout = QVBoxLayout(ledger_group)
        ledger_layout.setSpacing(6)

        self._ledger_stats = QLabel("no facts")
        self._ledger_stats.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SIZE_SMALL}px;"
        )
        self._ledger_stats.setWordWrap(True)
        ledger_layout.addWidget(self._ledger_stats)

        self._ledger_facts = QTextEdit()
        self._ledger_facts.setReadOnly(True)
        self._ledger_facts.setMinimumHeight(100)
        self._ledger_facts.setMaximumHeight(180)
        self._ledger_facts.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_DEEP};
                color: {ACCENT};
                border: 1px solid {BORDER};
                font-size: {FONT_SIZE_SMALL}px;
                padding: 6px;
                font-family: "Roboto Mono", monospace;
            }}
        """)
        ledger_layout.addWidget(self._ledger_facts)
        layout.addWidget(ledger_group)

        layout.addStretch()

    # -- public methods --

    def set_agent_status(self, role: str, model: str, status: str) -> None:
        """update or create a row in the active agents section."""
        color_map = {
            "active": STATUS_ACTIVE,
            "idle": TEXT_MUTED,
            "error": STATUS_ERROR,
        }
        color = color_map.get(status, TEXT_MUTED)
        dot = "\u25CF"  # filled circle

        if role not in self._agent_rows:
            label = QLabel()
            label.setStyleSheet(f"font-size: {FONT_SIZE_SMALL}px; padding: 1px 0;")
            self._agents_container.addWidget(label)
            self._agent_rows[role] = label

        row = self._agent_rows[role]
        row.setText(
            f'<span style="color:{color}">{dot}</span> '
            f'<span style="color:{TEXT_PRIMARY}">{role}</span> '
            f'<span style="color:{TEXT_MUTED}">({model})</span>'
        )

    def set_ledger_state(self, thread_id: str, facts: list[dict]) -> None:
        """display the current thread's state ledger contents."""
        if not facts:
            self._ledger_stats.setText(f"thread: {thread_id} | no facts")
            self._ledger_facts.setPlainText("(empty ledger)")
            return

        self._ledger_stats.setText(
            f"thread: {thread_id} | {len(facts)} locked facts"
        )
        lines = []
        for f in facts:
            pred = f.get("predicate", "?")
            obj = f.get("obj", "?")
            # highlight protected predicates
            protected = {"setting", "genre", "era", "timeline", "planet"}
            marker = "\U0001F512" if pred in protected else "\u2022"
            lines.append(f"  {marker} {pred}: {obj}")
        self._ledger_facts.setPlainText("\n".join(lines))

    def set_debug_stats(self, stats: dict) -> None:
        """refresh agent list from swarm status."""
        # update agent rows
        for agent_info in stats.get("agents", []):
            self.set_agent_status(
                role=agent_info["role"],
                model=agent_info["model"],
                status=agent_info.get("status", "idle"),
            )

    def append_log(self, text: str) -> None:
        """no-op — pipeline log now lives in the main chat area monitor."""
        pass

    def clear_log(self) -> None:
        """no-op — pipeline log removed."""
        pass

    # keep backward compat for any code calling append_debug
    def append_debug(self, text: str) -> None:
        pass
