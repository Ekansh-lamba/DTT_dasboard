from __future__ import annotations
from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QLabel,
    QPushButton, QStackedWidget, QComboBox, QButtonGroup, QMessageBox,
)
from gui import theme
from gui.models.repository import StudyRepository, Study
from gui.workers.pipeline_worker import PipelineController, RunRequest
from gui.pages.dashboard_page import DashboardPage
from gui.pages.new_study_page import NewStudyPage
from gui.pages.processing_page import ProcessingPage
from gui.pages.preprocess_page import PreprocessPage
from gui.pages.signals_page import SignalsPage
from gui.pages.validation_page import ValidationPage
from gui.pages.statistics_page import StatisticsPage
from gui.pages.histograms_page import HistogramsPage
from gui.pages.heatmaps_page import HeatmapsPage
from gui.pages.boxplots_page import BoxplotsPage
from gui.pages.rainflow_page import RainflowPage
from gui.pages.reports_page import ReportsPage
from gui.pages.comparison_page import ComparisonPage
from gui.pages.history_page import HistoryPage
NAV_ITEMS = [
    ("dashboard",  "Dashboard",   "▣"),
    ("new_study",  "New Study",   "＋"),
    ("processing", "Processing",  "⟳"),
    ("preprocess", "Preprocess",  "⚙"),
    ("signals",    "Signals",     "≋"),
    ("validation", "Validation",  "✓"),
    ("statistics", "Statistics",  "∑"),
    ("histograms", "Histograms",  "▥"),
    ("heatmaps",   "Heatmaps",    "▦"),
    ("boxplots",   "Boxplots",    "◫"),
    ("rainflow",   "Rainflow",    "∿"),
    ("reports",    "Reports",     "▤"),
    ("comparison", "Compare",     "⇄"),
    ("history",    "History",     "≡"),
]
_STUDY_PAGES = {
    "preprocess", "signals", "validation", "statistics", "histograms",
    "heatmaps", "boxplots", "rainflow", "reports",
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital Tyre Testing (DTT) Automation Platform")
        self.resize(1320, 860)

        self.repo = StudyRepository()
        self.pipeline = PipelineController(self)
        self.active_study: Optional[Study] = None

        root = QWidget()
        root.setObjectName("RootWidget")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_sidebar())

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        right.addWidget(self._build_topbar())
        self.stack = QStackedWidget()
        right.addWidget(self.stack, 1)
        right_wrap = QWidget()
        right_wrap.setObjectName("RootWidget")
        right_wrap.setLayout(right)
        layout.addWidget(right_wrap, 1)

        self._build_pages()
        self._wire_pipeline()

        # Initial state
        self._select_initial_study()
        self.navigate("dashboard")

    # UI construction
    def _build_sidebar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("Sidebar")
        bar.setFixedWidth(220)
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(14, 18, 14, 18)
        lay.setSpacing(6)

        brand = QHBoxLayout()
        logo = QLabel("◆")
        logo.setStyleSheet(f"color:{theme.ORANGE}; font-size:24px;")
        text = QVBoxLayout()
        text.setSpacing(0)
        t1 = QLabel("DTT Platform"); t1.setObjectName("BrandTitle")
        t2 = QLabel("Apollo WFT Analytics"); t2.setObjectName("BrandSub")
        text.addWidget(t1); text.addWidget(t2)
        brand.addWidget(logo)
        brand.addLayout(text)
        brand.addStretch(1)
        lay.addLayout(brand)
        lay.addSpacing(14)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        from PySide6.QtCore import QSize
        from gui.widgets.icons import preprocess_icon
        self.nav_buttons: Dict[str, QPushButton] = {}
        for key, label, glyph in NAV_ITEMS:
            if key == "preprocess":
                btn = QPushButton(f"   {label}")
                btn.setIcon(preprocess_icon())
                btn.setIconSize(QSize(16, 16))
            else:
                btn = QPushButton(f"  {glyph}   {label}")
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self.navigate(k))
            self.nav_group.addButton(btn)
            self.nav_buttons[key] = btn
            lay.addWidget(btn)

        lay.addStretch(1)
        ver = QLabel("Module 1 · v1.0")
        ver.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:11px;")
        lay.addWidget(ver)
        return bar

    def _build_topbar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(58)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(24, 0, 24, 0)

        self.page_title = QLabel("Dashboard")
        self.page_title.setObjectName("PageTitle")
        lay.addWidget(self.page_title)
        lay.addStretch(1)

        lay.addWidget(QLabel("Active study:"))
        self.study_selector = QComboBox()
        self.study_selector.setMinimumWidth(220)
        self.study_selector.currentIndexChanged.connect(self._on_study_changed)
        lay.addWidget(self.study_selector)
        return bar

    def _build_pages(self) -> None:
        self.pages: Dict[str, QWidget] = {}

        self.dashboard  = DashboardPage(self.repo)
        self.new_study  = NewStudyPage(self.repo)
        self.processing = ProcessingPage(self.repo)
        self.preprocess = PreprocessPage(self.repo)
        self.signals    = SignalsPage(self.repo)
        self.validation = ValidationPage(self.repo)
        self.statistics = StatisticsPage(self.repo)
        self.histograms = HistogramsPage(self.repo)
        self.heatmaps   = HeatmapsPage(self.repo)
        self.boxplots   = BoxplotsPage(self.repo)
        self.rainflow   = RainflowPage(self.repo)
        self.reports    = ReportsPage(self.repo)
        self.comparison = ComparisonPage(self.repo)
        self.history    = HistoryPage(self.repo)

        mapping = {
            "dashboard": self.dashboard, "new_study": self.new_study,
            "processing": self.processing, "preprocess": self.preprocess,
            "signals": self.signals, "validation": self.validation,
            "statistics": self.statistics, "histograms": self.histograms,
            "heatmaps": self.heatmaps, "boxplots": self.boxplots,
            "rainflow": self.rainflow, "reports": self.reports,
            "comparison": self.comparison, "history": self.history,
        }
        for key, _, _ in NAV_ITEMS:
            page = mapping[key]
            self.pages[key] = page
            self.stack.addWidget(page)

        # Cross-page navigation signals
        for page in self.pages.values():
            if hasattr(page, "navigate_requested"):
                page.navigate_requested.connect(self._handle_nav_request)
        self.new_study.start_requested.connect(self._start_pipeline)
        self.reports.regenerate_requested.connect(self._regenerate_report)
        self.history.open_study_requested.connect(self._open_study_by_name)
        self.history.studies_changed.connect(self._on_studies_changed)
        self.dashboard.navigate_requested.connect(self._handle_nav_request)

    def _on_studies_changed(self) -> None:
        self._reload_study_selector(self.study_selector.currentData())
        self.dashboard.refresh()

    def _wire_pipeline(self) -> None:
        self.pipeline.log_line.connect(self.processing.append_log)
        self.pipeline.stage_started.connect(self.processing.on_stage_started)
        self.pipeline.stage_finished.connect(self.processing.on_stage_finished)
        self.pipeline.progress.connect(self.processing.on_progress)
        self.pipeline.finished.connect(self._on_pipeline_finished)

    # Navigation
    def navigate(self, key: str) -> None:
        if key not in self.pages:
            return
        self.stack.setCurrentWidget(self.pages[key])
        self.nav_buttons[key].setChecked(True)
        label = next(lbl for k, lbl, _ in NAV_ITEMS if k == key)
        self.page_title.setText(label)

        page = self.pages[key]
        if key in _STUDY_PAGES:
            page.load_study(self.active_study)
        elif key == "dashboard":
            self.dashboard.refresh()
        elif key == "history":
            self.history.refresh()
        elif key == "new_study":
            self.new_study.reload_csvs()

    def _handle_nav_request(self, request: str) -> None:
        if request.startswith("open:"):
            self._open_study_by_name(request[len("open:"):])
        else:
            self.navigate(request)

    # Active study management
    def _reload_study_selector(self, select: Optional[str] = None) -> None:
        self.study_selector.blockSignals(True)
        self.study_selector.clear()
        studies = self.repo.list_studies()
        for s in studies:
            self.study_selector.addItem(s.name, s.name)
        if select:
            idx = self.study_selector.findData(select)
            if idx >= 0:
                self.study_selector.setCurrentIndex(idx)
        self.study_selector.blockSignals(False)
        name = self.study_selector.currentData()
        self.active_study = self.repo.get_study(name) if name else None

    def _select_initial_study(self) -> None:
        latest = self.repo.latest_study()
        self._reload_study_selector(latest.name if latest else None)

    def _on_study_changed(self, _index: int) -> None:
        name = self.study_selector.currentData()
        self.active_study = self.repo.get_study(name) if name else None
        # Refresh whichever study-bound page is visible
        current = self.stack.currentWidget()
        for key, page in self.pages.items():
            if page is current and key in _STUDY_PAGES:
                page.load_study(self.active_study)

    def _open_study_by_name(self, name: str) -> None:
        idx = self.study_selector.findData(name)
        if idx < 0:
            self._reload_study_selector(name)
        else:
            self.study_selector.setCurrentIndex(idx)
        self.active_study = self.repo.get_study(name)
        self.navigate("validation")
    def _start_pipeline(self, request: RunRequest) -> None:
        if self.pipeline.is_running:
            QMessageBox.information(self, "Pipeline busy",
                                    "A pipeline run is already in progress.")
            return
        self._pending_study = request.study
        self.processing.begin(request.study)
        self.processing.cancel_btn.clicked.connect(self.pipeline.cancel)
        self.navigate("processing")
        self.pipeline.start(request)
    def _regenerate_report(self, study: Study) -> None:
        v = study.validation()
        csv_name = v.get("file_name", "")
        csv_path = self.repo.csv_dir / csv_name if csv_name else None
        if not csv_path or not csv_path.exists():
            QMessageBox.information(
                self, "Regenerate report",
                "Original CSV not found in csv/. Use 'New Study' to re-run with "
                "the same study name to regenerate outputs.")
            self.navigate("new_study")
            return
        req = RunRequest(
            csv_path=csv_path,
            vehicle=study.vehicle if study.vehicle != "—" else "Vehicle",
            study=study.name,
        )
        self._start_pipeline(req)

    def _on_pipeline_finished(self, success: bool, info: str) -> None:
        self.processing.on_finished(success, info)
        if success and info:
            self._reload_study_selector(info)
            self.active_study = self.repo.get_study(info)
            self.dashboard.refresh()
            self.history.refresh()
            QMessageBox.information(
                self, "Analysis complete",
                f"Study '{info}' processed successfully.")
            self.navigate("validation")
        elif not success:
            QMessageBox.warning(self, "Pipeline failed", info)
