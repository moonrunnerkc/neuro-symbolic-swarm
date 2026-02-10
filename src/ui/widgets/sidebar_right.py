# Author: Bradley R. Kinnard
"""Right sidebar: settings panel + debug info."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui.theme import (
    ACCENT,
    BG_DEEP,
    BG_SECONDARY,
    BORDER,
    BORDER_LIGHT,
    FONT_SIZE,
    FONT_SIZE_SMALL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class SidebarRight(QWidget):
    """right panel with config controls and debug output."""

    threshold_changed = pyqtSignal(float)
    max_cycles_changed = pyqtSignal(int)
    context_window_changed = pyqtSignal(int)
    clear_memory_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background-color: {BG_SECONDARY};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        header = QLabel("Settings")
        header.setObjectName("header")
        layout.addWidget(header)

        # -- config group --
        config_group = QGroupBox("Swarm Config")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(8)

        # threshold
        threshold_label = QLabel("Score Threshold")
        threshold_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-weight: bold; font-size: {FONT_SIZE}px;"
        )
        config_layout.addWidget(threshold_label)

        threshold_row = QHBoxLayout()
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, 100)
        self._threshold_slider.setValue(15)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_row.addWidget(self._threshold_slider, stretch=1)

        self._threshold_value = QLabel("0.15")
        self._threshold_value.setFixedWidth(50)
        self._threshold_value.setStyleSheet(
            f"color: {ACCENT}; font-weight: bold; font-size: {FONT_SIZE}px;"
        )
        threshold_row.addWidget(self._threshold_value)
        config_layout.addLayout(threshold_row)

        # max cycles
        cycles_label = QLabel("Max Cycles")
        cycles_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-weight: bold; font-size: {FONT_SIZE}px;"
        )
        config_layout.addWidget(cycles_label)

        self._cycles_combo = QComboBox()
        self._cycles_combo.addItems(["1", "2", "3", "5", "10"])
        self._cycles_combo.setCurrentText("1")
        self._cycles_combo.currentTextChanged.connect(
            lambda v: self.max_cycles_changed.emit(int(v))
        )
        config_layout.addWidget(self._cycles_combo)
        layout.addWidget(config_group)

        # -- memory group --
        memory_group = QGroupBox("Memory")
        memory_layout = QVBoxLayout(memory_group)
        memory_layout.setSpacing(8)

        ctx_label = QLabel("Context Window")
        ctx_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-weight: bold; font-size: {FONT_SIZE}px;"
        )
        memory_layout.addWidget(ctx_label)

        ctx_row = QHBoxLayout()
        self._ctx_slider = QSlider(Qt.Orientation.Horizontal)
        self._ctx_slider.setRange(0, 20)
        self._ctx_slider.setValue(5)
        self._ctx_slider.valueChanged.connect(self._on_ctx_changed)
        ctx_row.addWidget(self._ctx_slider, stretch=1)

        self._ctx_value = QLabel("5")
        self._ctx_value.setFixedWidth(30)
        self._ctx_value.setStyleSheet(
            f"color: {ACCENT}; font-weight: bold; font-size: {FONT_SIZE}px;"
        )
        ctx_row.addWidget(self._ctx_value)
        memory_layout.addLayout(ctx_row)

        self._memory_stats = QLabel("entries: 0")
        self._memory_stats.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SIZE_SMALL}px;"
        )
        self._memory_stats.setWordWrap(True)
        memory_layout.addWidget(self._memory_stats)

        self._clear_mem_btn = QPushButton("Clear Memory")
        self._clear_mem_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_mem_btn.clicked.connect(self.clear_memory_requested.emit)
        memory_layout.addWidget(self._clear_mem_btn)
        layout.addWidget(memory_group)

        # -- debug group --
        debug_group = QGroupBox("Debug")
        debug_layout = QVBoxLayout(debug_group)

        self._debug_output = QTextEdit()
        self._debug_output.setReadOnly(True)
        self._debug_output.setMinimumHeight(140)
        self._debug_output.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_DEEP};
                color: {TEXT_SECONDARY};
                border: 1px solid {BORDER};
                font-size: {FONT_SIZE_SMALL}px;
                padding: 8px;
            }}
        """)
        debug_layout.addWidget(self._debug_output)
        layout.addWidget(debug_group)

        layout.addStretch()

    def _on_threshold_changed(self, value: int) -> None:
        threshold = value / 100.0
        self._threshold_value.setText(f"{threshold:.2f}")
        self.threshold_changed.emit(threshold)

    def get_max_cycles(self) -> int:
        return int(self._cycles_combo.currentText())

    def get_threshold(self) -> float:
        return self._threshold_slider.value() / 100.0

    def append_debug(self, text: str) -> None:
        self._debug_output.append(text)
        sb = self._debug_output.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_debug_stats(self, stats: dict) -> None:
        mem = stats.get("memory", {})
        lines = [
            f"agents:  {stats.get('agent_count', '?')}",
            f"active:  {stats.get('active_agents', '?')}",
            f"memory:  {mem.get('entry_count', stats.get('memory_size', '?'))} entries",
            f"vectors: {mem.get('index_vectors', '?')}",
            f"threads: {len(stats.get('threads', []))}",
        ]
        self._debug_output.setPlainText("\n".join(lines))
        # update memory stats widget
        self._memory_stats.setText(
            f"entries: {mem.get('entry_count', 0)} | "
            f"vectors: {mem.get('index_vectors', 0)}"
        )

    def _on_ctx_changed(self, value: int) -> None:
        self._ctx_value.setText(str(value))
        self.context_window_changed.emit(value)
