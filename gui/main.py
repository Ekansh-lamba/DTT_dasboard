"""
Digital Tyre Testing (DTT) Automation Platform — Desktop GUI entry point.

Run from the project root::

    python -m gui.main
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from gui import theme
from gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("DTT WFT Automation Platform")
    app.setOrganizationName("Apollo Tyres")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(theme.stylesheet())

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
