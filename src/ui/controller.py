# Author: Bradley R. Kinnard
"""Main UI controller. Wires swarm backend to PyQt6 widgets."""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QSplitter,
    QWidget,
)

from src.swarm import SwarmNexus
from src.ui.theme import BG_PRIMARY, get_stylesheet
from src.ui.widgets.chat_area import ChatArea
from src.ui.widgets.sidebar_left import SidebarLeft
from src.ui.widgets.sidebar_right import SidebarRight

logger = logging.getLogger(__name__)


class SwarmWorker(QObject):
    """runs swarm.respond() off the main thread."""

    response_ready = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(dict)
    progress_updated = pyqtSignal(str)

    def __init__(self, swarm: SwarmNexus):
        super().__init__()
        self._swarm = swarm

    @pyqtSlot(str, str)
    def process_query(self, query: str, thread_id: str) -> None:
        try:
            response = self._swarm.respond(
                query, thread_id,
                on_progress=lambda stage: self.progress_updated.emit(stage),
            )
            self.response_ready.emit(response, thread_id)
            self.status_updated.emit(self._swarm.get_status())
        except Exception as exc:
            logger.error("swarm worker error: %s", exc)
            self.error_occurred.emit(str(exc))
            try:
                self.status_updated.emit(self._swarm.get_status())
            except Exception:
                pass


class MainWindow(QMainWindow):
    """3-panel dark layout with swarm backend."""

    _dispatch_query = pyqtSignal(str, str)

    def __init__(self, swarm: SwarmNexus, parent=None):
        super().__init__(parent)
        self._swarm = swarm
        self._current_thread: Optional[str] = None

        self.setWindowTitle("Swarm Command Console")
        self.setMinimumSize(1280, 760)
        self.setStyleSheet(get_stylesheet())

        self._setup_ui()
        self._setup_worker()
        self._wire_signals()
        self._bootstrap()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._sidebar_left = SidebarLeft()
        self._chat_area = ChatArea()
        self._sidebar_right = SidebarRight()

        splitter.addWidget(self._sidebar_left)
        splitter.addWidget(self._chat_area)
        splitter.addWidget(self._sidebar_right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([270, 720, 270])

        root.addWidget(splitter)

    def _setup_worker(self) -> None:
        self._worker_thread = QThread()
        self._worker = SwarmWorker(self._swarm)
        self._worker.moveToThread(self._worker_thread)

        self._dispatch_query.connect(self._worker.process_query)
        self._worker.response_ready.connect(self._on_response)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.status_updated.connect(self._on_status_update)
        self._worker.progress_updated.connect(self._on_progress)

        self._worker_thread.start()

    def _wire_signals(self) -> None:
        self._chat_area.query_submitted.connect(self._on_query_submitted)
        self._sidebar_left.thread_selected.connect(self._on_thread_selected)
        self._sidebar_left.thread_delete_requested.connect(self._on_delete_thread)
        self._sidebar_left.new_thread_requested.connect(self._on_new_thread)
        self._sidebar_left.clear_all_threads_requested.connect(
            self._on_clear_all_threads
        )
        self._sidebar_left.clear_memory_requested.connect(self._on_clear_memory)

    def _bootstrap(self) -> None:
        status = self._swarm.get_status()
        for agent_info in status.get("agents", []):
            self._sidebar_left.add_agent_card(
                role=agent_info["role"],
                model=agent_info["model"],
            )

        # list any existing threads in the sidebar
        existing = status.get("threads", [])
        for tid in existing:
            self._sidebar_left.add_thread(tid)

        # always start a fresh conversation on launch
        default_tid = "default"
        self._swarm.create_thread(default_tid)
        self._current_thread = default_tid
        if default_tid not in existing:
            self._sidebar_left.add_thread(default_tid, "general conversation")
        self._sidebar_left.select_thread(default_tid)

        self._chat_area.add_system_message(
            "swarm nexus initialized. submit a query to begin deep reasoning."
        )
        self._sidebar_right.set_debug_stats(status)
        self._sidebar_left.update_memory_stats(status)

    # -- slots --

    @pyqtSlot(str)
    def _on_query_submitted(self, query: str) -> None:
        if not self._current_thread:
            self._chat_area.add_system_message("no active thread")
            return
        self._chat_area.set_input_enabled(False)
        self._chat_area.show_thinking("embedding query")
        self._sidebar_right.append_log(f"> {query[:60]}")
        self._dispatch_query.emit(query, self._current_thread)

    @pyqtSlot(str, str)
    def _on_response(self, response: str, thread_id: str) -> None:
        self._chat_area.hide_thinking()
        self._chat_area.add_swarm_message(response)
        self._chat_area.set_input_enabled(True)

    @pyqtSlot(str)
    def _on_error(self, error: str) -> None:
        self._chat_area.hide_thinking()
        self._chat_area.add_system_message(f"error: {error}")
        self._chat_area.set_input_enabled(True)
        self._sidebar_right.append_log(f"ERROR: {error}")

    @pyqtSlot(str)
    def _on_progress(self, stage: str) -> None:
        """route swarm progress emissions to the pipeline monitor + sidebar.

        structured cues:
          "Role:active|idle|error"  -> agent status card + monitor
          "ledger:ACTION detail"    -> ledger status in monitor
          "Critic rejected/approved" -> critic verdict in monitor
          plain string              -> phase label in monitor
        """
        # agent status transitions: "Parser:active", "Critic:idle"
        if ":" in stage and stage.split(":", 1)[1] in ("active", "idle", "error"):
            role, status = stage.split(":", 1)
            self._sidebar_left.update_agent(role, status)
            agent = next(
                (a for a in self._swarm._agents if a.role == role), None
            )
            if agent:
                self._sidebar_right.set_agent_status(role, agent.model, status)
            # also show in pipeline monitor
            self._chat_area.update_agent_status(role, status)
            self._sidebar_right.append_log(f"[{role}: {status.upper()}]")
            return

        # ledger events: "ledger:locked", "ledger:blocked", "ledger:rejected X",
        #                "ledger:passed X", "ledger:rollback"
        if stage.startswith("ledger:"):
            parts = stage[7:].split(" ", 1)
            action = parts[0]
            detail = parts[1] if len(parts) > 1 else ""
            self._chat_area.update_ledger_status(action, detail)
            self._sidebar_right.append_log(f"[Ledger: {action.upper()}] {detail}")
            return

        # critic verdicts: "Critic rejected draft 1 (Parser)"
        if stage.startswith("Critic rejected") or stage.startswith("Critic approved"):
            action = "rejected" if "rejected" in stage else "approved"
            self._chat_area.update_agent_status("Critic", action, stage)
            self._sidebar_right.append_log(stage)
            return

        # generic phase update
        self._chat_area.update_thinking_stage(stage)
        self._sidebar_right.append_log(stage)

    @pyqtSlot(dict)
    def _on_status_update(self, status: dict) -> None:
        for agent_info in status.get("agents", []):
            self._sidebar_left.update_agent(
                role=agent_info["role"],
                status=agent_info["status"],
                log=f"processed: {agent_info.get('messages_processed', 0)}",
            )
        self._sidebar_right.set_debug_stats(status)
        self._sidebar_left.update_memory_stats(status)
        # update state ledger display for current thread
        self._refresh_ledger()

    @pyqtSlot(str)
    def _on_thread_selected(self, thread_id: str) -> None:
        self._current_thread = thread_id
        self._chat_area.clear_chat()
        self._chat_area.add_system_message(f"switched to thread: {thread_id}")
        try:
            history = self._swarm.get_thread_history(thread_id)
            for msg in history:
                if msg["role"] == "user":
                    self._chat_area.add_user_message(msg["content"])
                else:
                    self._chat_area.add_swarm_message(msg["content"])
        except ValueError:
            pass
        self._refresh_ledger()

    def _refresh_ledger(self) -> None:
        """push current thread's state ledger facts to the right sidebar."""
        if not self._current_thread:
            return
        try:
            facts = self._swarm._state.query(
                self._current_thread, include_global=True
            )
            fact_dicts = [f.to_dict() for f in facts]
            self._sidebar_right.set_ledger_state(
                self._current_thread, fact_dicts
            )
        except Exception:
            self._sidebar_right.set_ledger_state(self._current_thread, [])

    @pyqtSlot(float)
    def _on_threshold_changed(self, value: float) -> None:
        self._swarm._app_config.score_threshold = value
        for agent in self._swarm._agents:
            agent.threshold = value
        self._sidebar_right.append_log(f"threshold -> {value:.2f}")

    @pyqtSlot(int)
    def _on_max_cycles_changed(self, value: int) -> None:
        self._swarm._app_config.max_cycles = value
        self._sidebar_right.append_log(f"max_cycles -> {value}")

    @pyqtSlot(int)
    def _on_context_window_changed(self, value: int) -> None:
        self._swarm._app_config.context_window = value
        self._sidebar_right.append_log(f"context_window -> {value}")

    @pyqtSlot()
    def _on_clear_memory(self) -> None:
        self._swarm.clear_memory()
        self._sidebar_right.append_log("memory cleared")
        status = self._swarm.get_status()
        self._sidebar_right.set_debug_stats(status)
        self._sidebar_left.update_memory_stats(status)
        self._refresh_ledger()

    @pyqtSlot()
    def _on_clear_all_threads(self) -> None:
        """wipe all threads, state ledger, and reset to a fresh default thread."""
        removed = self._swarm.clear_all_threads()
        self._sidebar_left.clear_all_threads()

        # start fresh
        default_tid = "default"
        self._swarm.create_thread(default_tid)
        self._current_thread = default_tid
        self._sidebar_left.add_thread(default_tid, "general conversation")
        self._sidebar_left.select_thread(default_tid)

        self._chat_area.clear_chat()
        self._chat_area.add_system_message(
            f"cleared {len(removed)} threads. fresh start."
        )
        self._sidebar_right.append_log(f"cleared {len(removed)} threads")
        status = self._swarm.get_status()
        self._sidebar_right.set_debug_stats(status)
        self._sidebar_left.update_memory_stats(status)
        self._refresh_ledger()

    @pyqtSlot(str)
    def _on_delete_thread(self, thread_id: str) -> None:
        """delete a single thread and switch away if it was active."""
        deleted = self._swarm.delete_thread(thread_id)
        if not deleted:
            return
        self._sidebar_left.remove_thread(thread_id)
        self._sidebar_right.append_log(f"deleted thread: {thread_id}")

        # if we just deleted the active thread, switch to another or create default
        if self._current_thread == thread_id:
            remaining = self._swarm.get_status().get("threads", [])
            if remaining:
                fallback = remaining[0]
                self._current_thread = fallback
                self._sidebar_left.select_thread(fallback)
                self._on_thread_selected(fallback)
            else:
                default_tid = "default"
                self._swarm.create_thread(default_tid)
                self._current_thread = default_tid
                self._sidebar_left.add_thread(default_tid, "general conversation")
                self._sidebar_left.select_thread(default_tid)
                self._chat_area.clear_chat()
                self._chat_area.add_system_message("thread deleted. fresh start.")

        status = self._swarm.get_status()
        self._sidebar_right.set_debug_stats(status)
        self._sidebar_left.update_memory_stats(status)
        self._refresh_ledger()

    @pyqtSlot()
    def _on_new_thread(self) -> None:
        name, ok = QInputDialog.getText(self, "New Thread", "Thread name:")
        if not ok or not name.strip():
            return
        tid = name.strip().replace(" ", "-").lower()
        try:
            self._swarm.create_thread(tid)
            # only add to sidebar if not already listed
            existing = self._swarm.get_status().get("threads", [])
            self._sidebar_left.add_thread(tid, name.strip())
            self._sidebar_left.select_thread(tid)
            self._current_thread = tid
            self._chat_area.clear_chat()
            self._chat_area.add_system_message(f"new thread: {tid}")
            status = self._swarm.get_status()
            self._sidebar_right.set_debug_stats(status)
            self._sidebar_left.update_memory_stats(status)
        except Exception as exc:
            self._sidebar_right.append_log(f"thread error: {exc}")

    def closeEvent(self, event) -> None:
        self._swarm.close()
        self._worker_thread.quit()
        self._worker_thread.wait(3000)
        super().closeEvent(event)
