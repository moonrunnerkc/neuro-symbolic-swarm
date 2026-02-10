# Author: Bradley R. Kinnard
"""Dark grey/blue 'evil genius' theme for the Grok-style UI.
Professional, high-contrast, no warm tones."""

# -- core palette: dark greys + steel blue accents --
BG_DEEP = "#0A0A0F"          # deepest background
BG_PRIMARY = "#0E0E14"       # main window bg
BG_SECONDARY = "#14141E"     # panels, sidebars
BG_TERTIARY = "#1A1A28"      # cards, elevated surfaces
BG_ELEVATED = "#20202E"      # hover states, active items
BG_INPUT = "#12121C"         # input fields

# -- accent: cold blue --
ACCENT = "#4A9EFF"           # primary interactive color
ACCENT_HOVER = "#6AB4FF"     # hover state
ACCENT_DIM = "#2D6BB5"       # pressed / muted accent
ACCENT_GLOW = "#3A8AEE"      # subtle glow effects
ACCENT_FAINT = "#1A3050"     # very subtle tint

# -- text --
TEXT_PRIMARY = "#E8EAF0"     # high contrast white-blue
TEXT_SECONDARY = "#A0A8B8"   # secondary info
TEXT_MUTED = "#5A6270"       # dimmed / inactive
TEXT_BRIGHT = "#FFFFFF"      # maximum emphasis

# -- borders --
BORDER = "#252535"           # standard border
BORDER_LIGHT = "#353548"     # prominent borders
BORDER_ACCENT = "#2A4A6A"   # accent-tinted border

# -- status colors --
STATUS_ACTIVE = "#4AFF8A"   # green - running
STATUS_IDLE = "#5A6270"      # grey - waiting
STATUS_ERROR = "#FF4A4A"     # red - failed
STATUS_WARN = "#FFB84A"      # amber - caution

# -- chat bubbles --
USER_BUBBLE = "#151525"
USER_BUBBLE_BORDER = "#2A2A40"
SWARM_BUBBLE = "#0F1A2A"
SWARM_BUBBLE_BORDER = "#1A3050"

# -- send button (distinct bright blue) --
SEND_BG = "#2A6FD0"
SEND_HOVER = "#3580E8"
SEND_PRESSED = "#1E5AAA"

# -- typography --
FONT_FAMILY = "Roboto Mono"
FONT_SIZE = 13
FONT_SIZE_SMALL = 11
FONT_SIZE_HEADER = 17
FONT_SIZE_TINY = 10


def get_stylesheet() -> str:
    """full QSS stylesheet -- dark grey/blue evil genius theme."""
    return f"""
    /* -- global -- */
    QMainWindow, QWidget {{
        background-color: {BG_PRIMARY};
        color: {TEXT_PRIMARY};
        font-family: "{FONT_FAMILY}", "Consolas", monospace;
        font-size: {FONT_SIZE}px;
    }}

    /* -- scrollbars -- */
    QScrollBar:vertical {{
        background: {BG_SECONDARY};
        width: 8px;
        margin: 0;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER_LIGHT};
        min-height: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {ACCENT_DIM};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        height: 0;
    }}

    /* -- input fields -- */
    QLineEdit {{
        background-color: {BG_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 8px;
        padding: 10px 14px;
        font-size: {FONT_SIZE}px;
        selection-background-color: {ACCENT_DIM};
    }}
    QLineEdit:focus {{
        border-color: {ACCENT};
    }}
    QLineEdit:disabled {{
        color: {TEXT_MUTED};
        border-color: {BORDER};
    }}

    /* -- buttons -- */
    QPushButton {{
        background-color: {ACCENT_FAINT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_ACCENT};
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: {FONT_SIZE}px;
    }}
    QPushButton:hover {{
        background-color: {ACCENT_DIM};
        border-color: {ACCENT};
        color: {TEXT_BRIGHT};
    }}
    QPushButton:pressed {{
        background-color: {ACCENT};
        color: {BG_DEEP};
    }}
    QPushButton:disabled {{
        background-color: {BG_TERTIARY};
        color: {TEXT_MUTED};
        border-color: {BORDER};
    }}

    /* -- send button special class -- */
    QPushButton#sendButton {{
        background-color: {SEND_BG};
        color: {TEXT_BRIGHT};
        border: none;
        border-radius: 8px;
        font-size: {FONT_SIZE}px;
        font-weight: bold;
    }}
    QPushButton#sendButton:hover {{
        background-color: {SEND_HOVER};
    }}
    QPushButton#sendButton:pressed {{
        background-color: {SEND_PRESSED};
    }}
    QPushButton#sendButton:disabled {{
        background-color: {BORDER};
        color: {TEXT_MUTED};
    }}

    /* -- lists -- */
    QListWidget {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 6px;
        outline: none;
        font-size: {FONT_SIZE}px;
    }}
    QListWidget::item {{
        padding: 10px 8px;
        border-bottom: 1px solid {BORDER};
    }}
    QListWidget::item:selected {{
        background-color: {BG_ELEVATED};
        color: {ACCENT};
        border-left: 3px solid {ACCENT};
    }}
    QListWidget::item:hover {{
        background-color: {BG_TERTIARY};
    }}

    /* -- text displays -- */
    QTextEdit, QTextBrowser {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 8px;
        font-size: {FONT_SIZE}px;
    }}

    /* -- labels -- */
    QLabel {{
        color: {TEXT_PRIMARY};
        font-size: {FONT_SIZE}px;
    }}
    QLabel#header {{
        font-size: {FONT_SIZE_HEADER}px;
        font-weight: bold;
        color: {ACCENT};
        padding: 2px 0;
    }}
    QLabel#muted {{
        color: {TEXT_MUTED};
        font-size: {FONT_SIZE_SMALL}px;
    }}

    /* -- sliders -- */
    QSlider::groove:horizontal {{
        background: {BORDER};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT};
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {ACCENT_HOVER};
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT_DIM};
        border-radius: 3px;
    }}

    /* -- combo boxes -- */
    QComboBox {{
        background-color: {BG_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: {FONT_SIZE}px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
        selection-background-color: {BG_ELEVATED};
        border: 1px solid {BORDER_LIGHT};
    }}

    /* -- splitters -- */
    QSplitter::handle {{
        background-color: {BORDER};
        width: 1px;
    }}

    /* -- group boxes -- */
    QGroupBox {{
        border: 1px solid {BORDER_LIGHT};
        border-radius: 6px;
        margin-top: 14px;
        padding: 18px 10px 10px 10px;
        font-weight: bold;
        font-size: {FONT_SIZE}px;
        color: {TEXT_SECONDARY};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: {ACCENT};
    }}

    /* -- scroll areas -- */
    QScrollArea {{
        background-color: transparent;
        border: none;
    }}

    /* -- input dialog styling -- */
    QInputDialog {{
        background-color: {BG_SECONDARY};
        color: {TEXT_PRIMARY};
    }}
    """
