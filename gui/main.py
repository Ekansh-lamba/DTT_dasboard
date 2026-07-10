"""
Digital Tyre Testing (DTT) Automation Platform — desktop entry point.

    python -m gui.main            # launch the GUI

When packaged as a single-file executable, the GUI relaunches this same
executable with ``--run-pipeline`` to run the analysis backend as a subprocess
(a frozen exe has no ``python -m dtt.pipeline`` available).
"""

from __future__ import annotations

import sys


def main() -> int:
    # Frozen-exe backend dispatch: run the pipeline CLI instead of the GUI.
    if len(sys.argv) > 1 and sys.argv[1] == "--run-pipeline":
        del sys.argv[1]
        from dtt.pipeline import _cli
        _cli()
        return 0

    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont
    from gui import theme
    from gui.main_window import MainWindow

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