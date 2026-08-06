# PyInstaller build spec for the DTT WFT Automation Platform (single-file Windows exe).
#   pyinstaller DTT-Platform.spec
#
# The app only uses Qt Core/Gui/Widgets/Svg — every other (large) Qt module is
# excluded so the build is fast and the executable stays reasonably small.

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# build_exe.ps1 -Console sets this to keep a console window attached, which is
# the only way to see a startup traceback from the packaged app.
CONSOLE = os.environ.get("DTT_BUILD_CONSOLE") == "1"

# build_exe.ps1 -OneFile produces a single self-contained .exe with nothing to
# extract by hand — the "just send it and double-click" build.
ONEFILE = os.environ.get("DTT_BUILD_ONEFILE") == "1"

hiddenimports = (
    collect_submodules("dtt")
    + collect_submodules("gui")
    # scipy.ndimage supplies the FAMOS smo/de-glitch kernels (uniform_filter1d,
    # median_filter); scipy submodules load lazily, so name it explicitly.
    # openpyxl is selected by name (pd.ExcelWriter(engine="openpyxl")) for the
    # comparison workbook export, so static analysis cannot see the import.
    + ["scipy.signal", "scipy.ndimage", "scipy.special", "rainflow", "pptx",
       "openpyxl", "matplotlib.backends.backend_pdf",
       "PySide6.QtSvg", "PySide6.QtPrintSupport"]
)

# matplotlib needs its data files (fonts, mpl-data); scipy ships compiled libs.
datas = collect_data_files("matplotlib")

excludes = [
    # Heavy Qt modules the app never imports
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput", "PySide6.Qt3DAnimation",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets", "PySide6.QtQml",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtPositioning",
    "PySide6.QtLocation", "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtSensors",
    "PySide6.QtSerialPort", "PySide6.QtWebSockets", "PySide6.QtWebChannel",
    "PySide6.QtSql", "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtTest",
    "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtScxml", "PySide6.QtStateMachine", "PySide6.QtTextToSpeech",
    "PySide6.QtUiTools", "PySide6.QtRemoteObjects", "PySide6.QtNetworkAuth",
    # Other unused heavyweights
    "tkinter", "PyQt5", "PyQt6", "IPython", "pytest", "notebook", "sphinx",
    # Large ML / data libs installed in the env but NOT used by this app
    "torch", "torchvision", "torchaudio", "tensorflow", "tensorboard", "keras",
    "sklearn", "scikit_learn", "polars", "cv2", "numba", "llvmlite", "sympy",
    "jax", "jaxlib", "onnx", "onnxruntime", "transformers", "cupy", "pyarrow",
    "numpy.distutils", "scipy.weave",
]

a = Analysis(
    ["gui/main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

if ONEFILE:
    # Single self-contained .exe. The runtime unpacks to a temp folder on each
    # launch, and because the GUI relaunches itself (--run-pipeline) for each
    # analysis, that unpack cost is paid again per run. Simplest to hand over.
    exe = EXE(
        pyz, a.scripts, a.binaries, a.datas, [],
        name="DTT-Platform",
        debug=False,
        strip=False,
        upx=False,
        console=CONSOLE,
        runtime_tmpdir=None,
    )
else:
    # onedir (default): one shared extracted folder, so launching and every
    # subsequent analysis run start immediately. Ship the whole folder.
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="DTT-Platform",
        debug=False,
        strip=False,
        upx=False,
        console=CONSOLE,
    )
    coll = COLLECT(
        exe, a.binaries, a.datas,
        strip=False,
        upx=False,
        name="DTT-Platform",
    )