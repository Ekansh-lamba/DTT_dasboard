"""
Digital Tyre Testing (DTT) Automation Platform — desktop entry point.

    python -m gui.main            # launch the GUI

When packaged as a single-file executable, the GUI relaunches this same
executable with ``--run-pipeline`` to run the analysis backend as a subprocess
(a frozen exe has no ``python -m dtt.pipeline`` available).
"""

from __future__ import annotations

import sys


def _self_test() -> int:
    """Import everything a packaged build needs, including the lazy imports.

    Several dependencies are selected by name at runtime — openpyxl via
    ``pd.ExcelWriter(engine="openpyxl")``, matplotlib's PDF backend via
    ``PdfPages`` — so PyInstaller's static analysis cannot see them. A build
    missing one of those looks perfectly healthy until a user clicks Export and
    gets a traceback. This flag exercises them at build-verification time.
    """
    checks = [
        ("dtt.preprocessing", lambda: __import__("dtt.preprocessing", fromlist=["x"])),
        ("dtt.comparison", lambda: __import__("dtt.comparison", fromlist=["x"])),
        ("dtt.comparison_io", lambda: __import__("dtt.comparison_io", fromlist=["x"])),
        ("gui.pages.comparison_page",
         lambda: __import__("gui.pages.comparison_page", fromlist=["x"])),
        ("scipy.ndimage", lambda: __import__("scipy.ndimage", fromlist=["x"])),
        ("openpyxl (Excel export)", lambda: __import__("openpyxl")),
        ("matplotlib PDF backend",
         lambda: __import__("matplotlib.backends.backend_pdf", fromlist=["x"])),
        ("rainflow", lambda: __import__("rainflow")),
        ("pptx (report builder)", lambda: __import__("pptx")),
    ]
    failed = 0
    for label, fn in checks:
        try:
            fn()
            print(f"  OK      {label}")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"  FAILED  {label}: {type(exc).__name__}: {exc}")
    print(f"\n{len(checks) - failed}/{len(checks)} imports OK")
    return 1 if failed else 0


def main() -> int:
    # Frozen-exe backend dispatch: run the pipeline CLI instead of the GUI.
    if len(sys.argv) > 1 and sys.argv[1] == "--run-pipeline":
        del sys.argv[1]
        from dtt.pipeline import _cli
        _cli()
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        return _self_test()

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