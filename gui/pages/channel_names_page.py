"""Screen — Channel Names: the one place channel naming is set.

Every channel is shown under a standard name (``FR_Fx_2`` and ``FR_Fx`` are
both ``FR_Fx``; see :mod:`dtt.channel_names`), and a display name can be set
for any of them -- "Front Right Fx", "Steer L Fz". The names are stored once,
in ``channel_names.json`` in the data folder, and used by every figure title,
box plot, table and report slide produced afterwards, for every study.

Data files are never renamed: ``processed_data.csv`` keeps the spelling the
recording used, so existing studies and the FAMOS cross-check keep working.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from gui.pages.base_page import BasePage
from gui.widgets.common import Card, ScrollPage, SectionTitle
from dtt.channel_names import (
    NAMES_FILE, canonical_name, channel_unit, load_names, save_names,
)
from dtt.channels import COMPONENTS, parse_channel
from dtt.config import OPTIONAL_MOMENT_CHANNELS, MANDATORY_CHANNELS

_COL_SOURCE, _COL_STANDARD, _COL_UNIT, _COL_DISPLAY = range(4)


def _wheel_first(std: str) -> tuple:
    """Sort key: wheel channels first, by position then component; the rest
    after them in file order (``sorted`` is stable)."""
    parsed = parse_channel(std)
    if parsed is None:
        return (1,)
    pos, comp = parsed
    return (0, pos.axle, pos.side, pos.sub or "", COMPONENTS.index(comp))


class ChannelNamesPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = ScrollPage()
        outer.addWidget(scroll)
        body = scroll.body()

        body.addWidget(SectionTitle("Channel Names"))
        card = Card()
        self.info = QLabel()
        self.info.setWordWrap(True)
        card.layout().addWidget(self.info)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Name in this study's file", "Standard name", "Unit", "Display name (editable)"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(_COL_DISPLAY, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setMinimumHeight(460)
        card.layout().addWidget(self.table)

        row = QHBoxLayout()
        self.save_btn = QPushButton("Save names")
        self.save_btn.setObjectName("Primary")
        self.save_btn.clicked.connect(self.save)
        self.reset_btn = QPushButton("Clear all display names")
        self.reset_btn.setObjectName("Secondary")
        self.reset_btn.clicked.connect(self._clear)
        row.addWidget(self.save_btn)
        row.addWidget(self.reset_btn)
        row.addStretch(1)
        self.status = QLabel("")
        row.addWidget(self.status)
        card.layout().addLayout(row)
        body.addWidget(card)
        body.addStretch(1)

    # ------------------------------------------------------------------ data
    def _study_columns(self) -> List[str]:
        if self.study is None or not self.study.processed_csv.exists():
            return []
        try:
            cols = list(pd.read_csv(self.study.processed_csv, nrows=0).columns)
        except Exception:                                         # noqa: BLE001
            return []
        return [c for c in cols if str(c).lower() != "time"]

    def rows(self) -> List[tuple]:
        """``[(source names, standard name), ...]`` in display order."""
        cols = self._study_columns()
        if not cols:
            cols = MANDATORY_CHANNELS + OPTIONAL_MOMENT_CHANNELS
        by_std: Dict[str, List[str]] = {}
        for c in cols:
            by_std.setdefault(canonical_name(c), []).append(str(c))
        for std in load_names():                  # names saved for other layouts
            by_std.setdefault(std, [])
        ordered = sorted(by_std, key=_wheel_first)
        return [(", ".join(by_std[std]), std) for std in ordered]

    def refresh(self) -> None:
        names = load_names()
        self.table.setRowCount(0)
        for src, std in self.rows():
            r = self.table.rowCount()
            self.table.insertRow(r)
            for col, text in ((_COL_SOURCE, src or "—"), (_COL_STANDARD, std),
                              (_COL_UNIT, channel_unit(std) or "—")):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, col, item)
            self.table.setItem(r, _COL_DISPLAY, QTableWidgetItem(names.get(std, "")))
        where = (f"study <b>{self.study.name}</b>" if self._study_columns()
                 else "the standard 4-wheel layout (open a study to see its channels)")
        self.info.setText(
            f"Channels of {where}. The <b>standard name</b> is the same for every "
            f"spelling of a channel (FR_Fx_2 → FR_Fx). Type a <b>display name</b> to "
            f"use it in every figure, box plot, table and report from the next run on "
            f"— leave it blank to keep the standard name. To update a study already "
            f"processed, re-run it in Analysis mode. Data files are never renamed.<br>"
            f"Saved in <code>{NAMES_FILE}</code>.")
        self.status.setText("")

    def edited_names(self) -> Dict[str, str]:
        out = {}
        for r in range(self.table.rowCount()):
            std = self.table.item(r, _COL_STANDARD).text()
            item = self.table.item(r, _COL_DISPLAY)
            if item is not None and item.text().strip():
                out[std] = item.text().strip()
        return out

    def save(self) -> None:
        # keep names for channels this study does not have
        names = load_names()
        shown = {self.table.item(r, _COL_STANDARD).text()
                 for r in range(self.table.rowCount())}
        for std in shown:
            names.pop(std, None)
        names.update(self.edited_names())
        try:
            save_names(names)
        except OSError as ex:
            QMessageBox.critical(self, "Channel Names", f"Could not save: {ex}")
            return
        self.status.setText(f"Saved {len(load_names())} display name(s).")

    def _clear(self) -> None:
        for r in range(self.table.rowCount()):
            self.table.setItem(r, _COL_DISPLAY, QTableWidgetItem(""))
        self.status.setText("Cleared — press Save names to keep this.")
