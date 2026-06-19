"""Base class for every screen.

A page is given the active :class:`Study` via :meth:`load_study`. Pages that
need no study (Dashboard, New Study, History) simply override what they need.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from gui.models.repository import Study, StudyRepository


class BasePage(QWidget):
    """Common contract for navigable screens."""

    # Emitted by pages that want the shell to navigate elsewhere.
    navigate_requested = Signal(str)          # page key

    def __init__(self, repo: StudyRepository, parent=None):
        super().__init__(parent)
        self.setObjectName("RootWidget")
        self.repo = repo
        self.study: Optional[Study] = None

    def load_study(self, study: Optional[Study]) -> None:
        """Called whenever the active study changes or the page is shown."""
        self.study = study
        self.refresh()

    def refresh(self) -> None:
        """Override to repopulate widgets from ``self.study``."""
