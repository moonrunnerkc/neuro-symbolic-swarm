# Author: Bradley R. Kinnard
"""Left sidebar: thread list and management."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.theme import (
    ACCENT,
    BG_SECONDARY,
    FONT_SIZE,
    FONT_SIZE_SMALL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class SidebarLeft(QWidget):
    """left panel with thread selector and management controls."""

    thread_selected = pyqtSignal(str)
    thread_delete_requested = pyqtSignal(str)
    new_thread_requested = pyqtSignal()
    clear_all_threads_requested = pyqtSignal()
    clear_memory_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {BG_SECONDARY};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # threads header + new thread button
        threads_row = QHBoxLayout()
        threads_row.setSpacing(6)
        threads_header = QLabel("Threads")
        threads_header.setObjectName("header")
        threads_row.addWidget(threads_header)
        threads_row.addStretch()

        self._new_thread_btn = QPushButton("+ New")
        self._new_thread_btn.setFixedHeight(26)
        self._new_thread_btn.setMinimumWidth(60)
        self._new_thread_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._new_thread_btn.clicked.connect(self.new_thread_requested.emit)
        threads_row.addWidget(self._new_thread_btn)
        layout.addLayout(threads_row)

        # thread list
        self._thread_list = QListWidget()
        self._thread_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._thread_list.itemClicked.connect(self._on_thread_clicked)
        self._thread_list.customContextMenuRequested.connect(
            self._on_thread_context_menu
        )
        layout.addWidget(self._thread_list, stretch=1)

        # clear all threads
        self._clear_threads_btn = QPushButton("Clear All Threads")
        self._clear_threads_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_threads_btn.clicked.connect(
            self.clear_all_threads_requested.emit
        )
        layout.addWidget(self._clear_threads_btn)

        # -- memory stats + clear --
        self._memory_stats = QLabel("entries: 0 | vectors: 0")
        self._memory_stats.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SIZE_SMALL}px;"
        )
        self._memory_stats.setWordWrap(True)
        layout.addWidget(self._memory_stats)

        self._clear_mem_btn = QPushButton("Clear Memory")
        self._clear_mem_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_mem_btn.clicked.connect(self.clear_memory_requested.emit)
        layout.addWidget(self._clear_mem_btn)

    # -- agent card stubs (agents now live on right sidebar diagnostics) --

    def add_agent_card(self, role: str, model: str) -> None:
        pass

    def remove_agent_card(self, role: str) -> None:
        pass

    def update_agent(self, role: str, status: str, log: str = "") -> None:
        pass

    def update_memory_stats(self, stats: dict) -> None:
        """refresh the memory stats label from swarm status."""
        mem = stats.get("memory", {})
        self._memory_stats.setText(
            f"entries: {mem.get('entry_count', 0)} | "
            f"vectors: {mem.get('index_vectors', 0)} | "
            f"threads: {len(stats.get('threads', []))}"
        )

    # -- thread management --

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

    def clear_all_threads(self) -> None:
        """remove every thread from the list widget."""
        self._thread_list.clear()

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

    def _on_thread_context_menu(self, pos) -> None:
        """right-click context menu on a thread item."""
        item = self._thread_list.itemAt(pos)
        if not item:
            return
        tid = item.data(Qt.ItemDataRole.UserRole)
        if not tid:
            return

        menu = QMenu(self)
        delete_action = QAction(f"Delete '{tid}'", self)
        delete_action.triggered.connect(lambda: self.thread_delete_requested.emit(tid))
        menu.addAction(delete_action)
        menu.exec(self._thread_list.mapToGlobal(pos))
