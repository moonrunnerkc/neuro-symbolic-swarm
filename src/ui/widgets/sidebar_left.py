# Author: Bradley R. Kinnard
"""Left sidebar: agent status cards + thread list."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.ui.theme import (
    ACCENT,
    BG_PRIMARY,
    BG_SECONDARY,
    BORDER,
    FONT_SIZE,
    FONT_SIZE_HEADER,
    TEXT_MUTED,
    TEXT_PRIMARY,
)
from src.ui.widgets.agent_card import AgentCard


class SidebarLeft(QWidget):
    """left panel with agent cards and thread selector."""

    thread_selected = pyqtSignal(str)
    new_thread_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._agent_cards: dict[str, AgentCard] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {BG_SECONDARY};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # agents header
        agents_header = QLabel("Agents")
        agents_header.setObjectName("header")
        layout.addWidget(agents_header)

        # scrollable agent cards
        self._agents_scroll = QScrollArea()
        self._agents_scroll.setWidgetResizable(True)
        self._agents_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._agents_container = QWidget()
        self._agents_container.setStyleSheet(f"background-color: {BG_SECONDARY};")
        self._agents_layout = QVBoxLayout(self._agents_container)
        self._agents_layout.setSpacing(6)
        self._agents_layout.addStretch()
        self._agents_scroll.setWidget(self._agents_container)
        layout.addWidget(self._agents_scroll, stretch=2)

        # threads header + new thread button
        threads_row = QHBoxLayout()
        threads_row.setSpacing(6)
        threads_header = QLabel("Threads")
        threads_header.setObjectName("header")
        threads_row.addWidget(threads_header)
        threads_row.addStretch()

        self._new_thread_btn = QPushButton("+ New")
        self._new_thread_btn.setFixedHeight(26)
        self._new_thread_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._new_thread_btn.clicked.connect(self.new_thread_requested.emit)
        threads_row.addWidget(self._new_thread_btn)
        layout.addLayout(threads_row)

        # thread list
        self._thread_list = QListWidget()
        self._thread_list.itemClicked.connect(self._on_thread_clicked)
        layout.addWidget(self._thread_list, stretch=1)

    def add_agent_card(self, role: str, model: str) -> None:
        if role in self._agent_cards:
            return
        card = AgentCard(role=role, model=model)
        idx = self._agents_layout.count() - 1
        self._agents_layout.insertWidget(idx, card)
        self._agent_cards[role] = card

    def remove_agent_card(self, role: str) -> None:
        if card := self._agent_cards.pop(role, None):
            self._agents_layout.removeWidget(card)
            card.deleteLater()

    def update_agent(self, role: str, status: str, log: str = "") -> None:
        if card := self._agent_cards.get(role):
            card.update_status(status)
            if log:
                card.update_log(log)

    def add_thread(self, thread_id: str, summary: str = "") -> None:
        display = thread_id
        if summary:
            display += f" -- {summary[:35]}"
        item = QListWidgetItem(display)
        item.setData(Qt.ItemDataRole.UserRole, thread_id)
        self._thread_list.addItem(item)

    def remove_thread(self, thread_id: str) -> None:
        for i in range(self._thread_list.count()):
            item = self._thread_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == thread_id:
                self._thread_list.takeItem(i)
                break

    def select_thread(self, thread_id: str) -> None:
        """programmatically select a thread in the list."""
        for i in range(self._thread_list.count()):
            item = self._thread_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == thread_id:
                self._thread_list.setCurrentItem(item)
                break

    def _on_thread_clicked(self, item: QListWidgetItem) -> None:
        tid = item.data(Qt.ItemDataRole.UserRole)
        if tid:
            self.thread_selected.emit(tid)
