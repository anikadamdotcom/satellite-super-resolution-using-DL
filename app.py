# Desktop app: Sentinel-2 2x super-resolution (RCAN) and land-cover mapping.
# Pipeline: image ingest -> RCAN 2x super-resolution -> land-cover detection on the original image -> export

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Use the satsr package from src/ without requiring `pip install -e .`.
_SRC = os.path.join(os.path.dirname(os.path.realpath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# On Windows, make sure PyTorch's DLLs are found (avoids WinError 1114).
# torch must be imported before PyQt5.
if sys.platform == "win32":
    try:
        import importlib.util
        _spec = importlib.util.find_spec("torch")
        if _spec and _spec.submodule_search_locations:
            _torch_lib = os.path.join(list(_spec.submodule_search_locations)[0], "lib")
            if os.path.isdir(_torch_lib):
                os.add_dll_directory(_torch_lib)
                os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

import gc
import numpy as np
import torch

from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QFrame, QComboBox, QMessageBox, QCheckBox,
    QSlider, QButtonGroup, QProgressBar, QDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QMenu, QSizePolicy, QScrollArea, QGridLayout
)

import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from PIL import Image, ImageFilter

from satsr.models.rcan import L1BSRSR
from satsr.models.unet_segmentation import LandCoverSegmenter, CLASS_DEFINITIONS, REAL_CLASS_DEFINITIONS
from satsr.models.evaluation_metrics import compute_full_sr_evaluation
from satsr.runtime_support import configure_cpu_threads

APP_DIR = os.path.dirname(os.path.realpath(__file__))

# Friendly names for the five detected classes
FRIENDLY = {
    "Buildings": "Buildings",
    "Roads": "Roads",
    "Vegetation": "Trees & crops",
    "Water": "Water",
    "Bare Land": "Bare ground",
}


# -------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------

from satsr.geo_io import (BandData, read_band, percentile_stretch, load_any_user_image,
                    write_geotiff, write_single_band_geotiff, load_band_stack,
                    pixel_area_m2, output_valid_mask, render_rgb, stretch_limits,
                    save_image, save_text, resolve_output_path, IMAGE_FILTER)


def rgb_to_qpixmap(img):
    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    return QPixmap.fromImage(QImage(img.data, w, h, w * 3, QImage.Format_RGB888).copy())


from satsr.gui_support import JobMixin, open_image_dialog, save_dialog, folder_dialog
from satsr.results_export import (export_results_zip, score_rgb, difference_from_bicubic,
                            difference_rgb, compose_compare)
from satsr.geo_io import derived_note


STYLE = """
* { font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif; font-size: 13px; color: #1f2328; }
QMainWindow, QWidget#root { background: #f4f5f7; }
QFrame#sidebar { background: #ffffff; border-right: 1px solid #e3e5e8; }
QLabel#appTitle { font-size: 17px; font-weight: 600; }
QLabel#stepTitle { font-size: 13px; font-weight: 600; }
QLabel#hint { color: #6b7280; font-size: 12px; }
QLabel#status { color: #4b5563; }
QPushButton { background: #ffffff; border: 1px solid #d0d4da; border-radius: 6px; padding: 8px 12px; text-align: center; }
QPushButton:hover { background: #f3f4f6; }
QPushButton:disabled { color: #a0a6af; background: #f7f8f9; border-color: #e5e7eb; }
QPushButton#primary { background: #2563eb; border: 1px solid #2563eb; color: #ffffff; font-weight: 600; }
QPushButton#primary:hover { background: #1d4fd0; }
QPushButton#primary:disabled { background: #b9c9f2; border-color: #b9c9f2; color: #ffffff; }
QPushButton#seg { border-radius: 0; padding: 7px 16px; border: 1px solid #d0d4da; margin-left: -1px; }
QPushButton#seg:checked { background: #1f2328; color: #ffffff; border-color: #1f2328; }
QPushButton#seg:disabled { color: #b5bac2; }
QPushButton#link { border: none; background: transparent; color: #2563eb; padding: 4px 0; text-align: left; }
QPushButton#link:hover { text-decoration: underline; background: transparent; }
QComboBox { background: #ffffff; border: 1px solid #d0d4da; border-radius: 6px; padding: 6px 10px; }
QComboBox QAbstractItemView { background: #ffffff; selection-background-color: #e8effd; selection-color: #1f2328; border: 1px solid #d0d4da; }
QProgressBar { border: none; background: #e5e7eb; height: 4px; border-radius: 2px; }
QProgressBar::chunk { background: #2563eb; border-radius: 2px; }
QSlider::groove:horizontal { height: 4px; background: #e5e7eb; border-radius: 2px; }
QSlider::handle:horizontal { background: #2563eb; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QMenu { background: #ffffff; border: 1px solid #d0d4da; padding: 4px; }
QMenu::item { padding: 6px 18px; }
QMenu::item:selected { background: #e8effd; }
QTableWidget { background: #ffffff; border: 1px solid #e3e5e8; gridline-color: #eef0f2; }
QHeaderView::section { background: #f4f5f7; border: none; border-bottom: 1px solid #e3e5e8; padding: 6px; font-weight: 600; }
"""


# -------------------------------------------------------------
# Widgets
# -------------------------------------------------------------

class ImageView(QLabel):
    """Shows an image scaled to fit; accepts drops and click-to-open at any time."""
    clicked_empty = pyqtSignal()
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAcceptDrops(True)
        self._pix: Optional[QPixmap] = None
        self.show_empty()

    def show_empty(self):
        self._pix = None
        self.clear()
        self.setText("Drop an image here\nor click to choose one")
        self.setStyleSheet("background: #ffffff; border: 2px dashed #c9cdd4; border-radius: 10px;"
                           "color: #6b7280; font-size: 15px;")
        self.setCursor(Qt.PointingHandCursor)

    def set_array(self, arr: np.ndarray):
        self._pix = rgb_to_qpixmap(arr)
        self.setStyleSheet("background: #ffffff; border: 1px solid #e3e5e8; border-radius: 10px;")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to choose another image")
        self._refit()

    def _refit(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.width() - 16, self.height() - 16,
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refit()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked_empty.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if os.path.isfile(p):
                self.file_dropped.emit(p)
                break


def make_step(title: str, hint: str = "") -> Tuple[QWidget, QVBoxLayout]:
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    t = QLabel(title)
    t.setObjectName("stepTitle")
    lay.addWidget(t)
    if hint:
        h = QLabel(hint)
        h.setObjectName("hint")
        h.setWordWrap(True)
        lay.addWidget(h)
    return box, lay


class DetailsDialog(QDialog):
    def __init__(self, parent, stats: Optional[Dict], metrics: Optional[Dict], conf: Optional[float]):
        super().__init__(parent)
        self.setWindowTitle("Details")
        self.resize(560, 420)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(10)

        lay.addWidget(self._title("What was found"))
        if stats:
            tbl = QTableWidget(len(stats), 3)
            tbl.setHorizontalHeaderLabels(["Type", "Share of image", "Area (hectares)"])
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            for r, (name, d) in enumerate(stats.items()):
                it = QTableWidgetItem("  " + FRIENDLY.get(name, name))
                it.setForeground(QColor(d["color_hex"]))
                f = QFont(); f.setBold(True); it.setFont(f)
                tbl.setItem(r, 0, it)
                tbl.setItem(r, 1, QTableWidgetItem(f"{d['percentage']} %"))
                tbl.setItem(r, 2, QTableWidgetItem(str(d["area_hectares"]) if d["area_hectares"] is not None else "Unknown"))
            lay.addWidget(tbl, 1)
            if conf is not None:
                lay.addWidget(QLabel(f"Average score (uncalibrated): {conf}%"))
        else:
            lay.addWidget(QLabel("Run step 3 to see this."))

        lay.addWidget(self._title("Similarity to bicubic (no HR ground truth)"))
        if metrics:
            lay.addWidget(QLabel(
                f"PSNR {metrics['psnr_db']} dB   |   SSIM {metrics['ssim']}   |   "
                f"SAM {metrics['sam_degrees']}°   |   Edge detail {metrics['edge_gain_pct']:+}%"))
        else:
            lay.addWidget(QLabel("Run step 2 to see this."))
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QHBoxLayout(); row.addStretch(1); row.addWidget(close)
        lay.addLayout(row)

    @staticmethod
    def _title(t):
        l = QLabel(t); l.setObjectName("stepTitle"); return l


# -------------------------------------------------------------
# Main window
# -------------------------------------------------------------

class MainWindow(QMainWindow, JobMixin):
    VIEWS = [("original", "Original"), ("enhanced", "Enhanced"), ("compare", "Compare"),
             ("map", "Map"), ("overlay", "Map on photo"), ("confidence", "Scores")]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Super-Resolution & Land Cover")
        self.resize(1280, 800)
        self.setMinimumSize(980, 640)
        self.setAcceptDrops(True)

        # State
        self.image_name: Optional[str] = None
        self.ref_band: Optional[BandData] = None
        self.input_bgrn: Optional[np.ndarray] = None
        self.rgb_lr: Optional[np.ndarray] = None
        self.sr_bgrn: Optional[np.ndarray] = None
        self.bicubic_bgrn: Optional[np.ndarray] = None
        self.diff_magnitude: Optional[np.ndarray] = None
        self.diff_stats: Optional[Dict] = None
        self.rgb_sr_base: Optional[np.ndarray] = None
        self.seg: Optional[Dict] = None
        self.metrics: Optional[Dict] = None
        self.conf: Optional[float] = None

        self.cpu_threads = configure_cpu_threads()
        self.device = torch.device("cpu")
        self.gpu_available = torch.cuda.is_available()
        self._pending_device = None
        self.sr_engine: Optional[L1BSRSR] = None
        self.seg_engine = None
        self.checkpoint_dir = os.path.join(APP_DIR, "checkpoints", "demo")
        self._init_jobs()
        self._detect_after_enhance = False

        self._build_ui()
        self._refresh_state()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Sidebar
        side = QFrame(); side.setObjectName("sidebar"); side.setFixedWidth(300)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(20, 20, 20, 16)
        sl.setSpacing(18)

        title = QLabel("Satellite Mapping"); title.setObjectName("appTitle")
        sub = QLabel("2x RCAN enhancement and real-trained land cover. Fine imagery works best for roads and buildings.")
        sub.setObjectName("hint"); sub.setWordWrap(True)
        sl.addWidget(title); sl.addWidget(sub)

        self.chk_cpu = QCheckBox("Run without GPU (CPU)")
        self.chk_cpu.setChecked(True)
        self.chk_cpu.setToolTip("Checked: run enhancement and detection on any laptop CPU. "
                                "Uncheck to use an available NVIDIA CUDA GPU.")
        self.chk_cpu.toggled.connect(self._on_cpu_toggled)
        self.lbl_device = QLabel()
        self.lbl_device.setObjectName("hint"); self.lbl_device.setWordWrap(True)
        sl.addWidget(self.chk_cpu); sl.addWidget(self.lbl_device)
        self._sync_device_control()

        # 1. Open
        s1, l1 = make_step("1.  Choose an image", "PNG, JPG, TIFF, BMP or WebP. You can also drag a file into the window.")
        self.btn_open = QPushButton("Open image…"); self.btn_open.setObjectName("primary")
        self.btn_open.clicked.connect(self.choose_image)
        self.combo_sample = QComboBox()
        self.combo_sample.addItem("Real satellite samples…")
        self.combo_sample.activated.connect(self._on_sample)
        self._demo_paths = {}
        demo_folder = os.path.join(APP_DIR, "demo_images")
        if os.path.isdir(demo_folder):
            from satsr.geo_io import IMAGE_EXTENSIONS
            for name in sorted(os.listdir(demo_folder)):
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    index = self.combo_sample.count()
                    self._demo_paths[index] = os.path.join(demo_folder, name)
                    self.combo_sample.addItem("Sentinel-2: "+name)
            fine_sample = os.path.join(demo_folder, "high_resolution", "palu_8.tif")
            if os.path.isfile(fine_sample):
                self._demo_paths[self.combo_sample.count()] = fine_sample
                self.combo_sample.addItem("Palu: fine-resolution roads and buildings")
        l1.addWidget(self.btn_open); l1.addWidget(self.combo_sample)
        self.btn_bands = QPushButton("Open four satellite bands...")
        self.btn_bands.clicked.connect(self.choose_bands)
        l1.addWidget(self.btn_bands)
        sl.addWidget(s1)

        # 2. Enhance
        s2, l2 = make_step("2.  Make it sharper", "Doubles dimensions using the selected model.")
        self.btn_enhance = QPushButton("Enhance (2×)"); self.btn_enhance.setObjectName("primary")
        self.btn_enhance.clicked.connect(self.on_enhance)
        self.chk_sharp = QCheckBox("Extra sharpening")
        self.sld_sharp = QSlider(Qt.Horizontal); self.sld_sharp.setRange(50, 300); self.sld_sharp.setValue(150)
        self.sld_sharp.setEnabled(False)
        self.chk_sharp.toggled.connect(self._sharp_changed)
        self.sld_sharp.valueChanged.connect(self._sharp_changed)
        l2.addWidget(self.btn_enhance); l2.addWidget(self.chk_sharp); l2.addWidget(self.sld_sharp)
        sl.addWidget(s2)

        # 3. Detect
        s3, l3 = make_step("3.  Find what is in it", "Marks buildings, roads, trees, water and bare ground.")
        self.combo_detector = QComboBox()
        self.use_real_detector = True
        self.detector_kind = "ensemble"
        self.combo_detector.addItems(["Combined real models (recommended)", "U-Net experiment / custom model",
                                      "Hugging Face SegFormer (fast)", "SparseMask (previous model)"])
        self.combo_detector.setToolTip("Combined: SparseMask and Hugging Face SegFormer with flip averaging. "
                                       "Fast: SegFormer alone. Both use real OpenEarthMap training and run offline on CPU. "
                                       "NIR checks use measured bands only. "
                                       "Synthetic: original research experiment, unsuitable for real-scene accuracy claims.")
        self.combo_detector.currentIndexChanged.connect(self._detector_changed)
        l3.addWidget(self.combo_detector)
        self.lbl_detection_note = QLabel("Roads/buildings need fine imagery. Sentinel-2's 10 m pixels "
                                        "can miss small objects. Map labels are predictions.")
        self.lbl_detection_note.setWordWrap(True); self.lbl_detection_note.setObjectName("hint")
        l3.addWidget(self.lbl_detection_note)
        self.btn_detect = QPushButton("Detect"); self.btn_detect.setObjectName("primary")
        self.btn_detect.clicked.connect(self.on_detect)
        l3.addWidget(self.btn_detect)
        legend = QHBoxLayout(); legend.setSpacing(10)
        col = QVBoxLayout(); col.setSpacing(3)
        for info in REAL_CLASS_DEFINITIONS.values():
            row = QHBoxLayout(); row.setSpacing(8)
            sw = QLabel(); sw.setFixedSize(12, 12)
            sw.setStyleSheet(f"background:{info['hex']}; border-radius:2px;")
            tx = QLabel(FRIENDLY.get(info["name"], info["name"])); tx.setObjectName("hint")
            row.addWidget(sw); row.addWidget(tx); row.addStretch(1)
            col.addLayout(row)
        l3.addLayout(col)
        sl.addWidget(s3)

        sl.addStretch(1)
        self.btn_details = QPushButton("Details and scores"); self.btn_details.setObjectName("link")
        self.btn_details.clicked.connect(self.show_details)
        sl.addWidget(self.btn_details)
        self.btn_models = QPushButton("Choose model folder...")
        self.btn_models.clicked.connect(self.choose_models)
        sl.addWidget(self.btn_models)
        self.lbl_model = QLabel("Real-data land cover model; RCAN enhancement trained on real Sentinel-2/VENµS pairs")
        self.lbl_model.setObjectName("hint"); self.lbl_model.setWordWrap(True)
        sl.addWidget(self.lbl_model)
        side_scroll = QScrollArea()
        side_scroll.setWidget(side); side_scroll.setWidgetResizable(True)
        side_scroll.setFixedWidth(320); side_scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(side_scroll)

        # Main area
        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setContentsMargins(24, 20, 24, 16)
        ml.setSpacing(12)

        actions = QHBoxLayout()
        self.btn_choose = QPushButton("Choose image…")
        self.btn_choose.clicked.connect(self.choose_image)
        self.btn_run_all = QPushButton("Run all (enhance + map)")
        self.btn_run_all.setObjectName("primary")
        self.btn_run_all.clicked.connect(self.on_detect)
        actions.addWidget(self.btn_choose); actions.addWidget(self.btn_run_all)
        actions.addStretch(1)
        ml.addLayout(actions)

        seg_row = QHBoxLayout(); seg_row.setSpacing(0)
        self.view_group = QButtonGroup(self); self.view_group.setExclusive(True)
        self.view_btns: Dict[str, QPushButton] = {}
        for key, label in self.VIEWS:
            b = QPushButton(label); b.setObjectName("seg"); b.setCheckable(True)
            b.clicked.connect(lambda _, k=key: self._show_view(k))
            self.view_group.addButton(b); self.view_btns[key] = b
            seg_row.addWidget(b)
        seg_row.addStretch(1)
        self.lbl_file = QLabel(""); self.lbl_file.setObjectName("hint")
        self.lbl_file.setMaximumWidth(260)
        self.lbl_file.setTextFormat(Qt.PlainText)
        seg_row.addWidget(self.lbl_file)
        ml.addLayout(seg_row)

        self.viewer = ImageView()
        self.viewer.clicked_empty.connect(self.choose_image)
        self.viewer.file_dropped.connect(self.load_file)
        ml.addWidget(self.viewer, 1)

        self.download_panel = QFrame()
        downloads = QVBoxLayout(self.download_panel)
        downloads.setContentsMargins(0, 0, 0, 0); downloads.setSpacing(6)
        download_heading = QHBoxLayout()
        label = QLabel("Downloads"); label.setObjectName("stepTitle")
        self.lbl_download_hint = QLabel(""); self.lbl_download_hint.setObjectName("hint")
        self.lbl_download_hint.setWordWrap(True)
        download_heading.addWidget(label); download_heading.addWidget(self.lbl_download_hint, 1)
        downloads.addLayout(download_heading)
        grid = QGridLayout(); grid.setSpacing(6)
        for attr, text, callback, row, col in (
                ("btn_save_img", "Image PNG/JPG", self.on_save_image, 0, 0),
                ("btn_save_map", "Map PNG/TIF", self.on_save_map, 0, 1),
                ("btn_save_geotiff", "GeoTIFF", self.on_save_geotiff, 0, 2),
                ("btn_save_view", "Current view", self.on_save_view, 0, 3),
                ("btn_save_scores", "Scores PNG", self.on_save_scores, 1, 0),
                ("btn_save_report", "Report", self.on_save_report, 1, 1)):
            button = QPushButton(text); button.clicked.connect(callback)
            setattr(self, attr, button); grid.addWidget(button, row, col)
        self.btn_save_all = QPushButton("All results ZIP…")
        self.btn_save_all.setObjectName("primary"); self.btn_save_all.clicked.connect(self.on_save_all)
        grid.addWidget(self.btn_save_all, 1, 2, 1, 2)
        for col in range(4):
            grid.setColumnStretch(col, 1)
        downloads.addLayout(grid); ml.addWidget(self.download_panel)
        self.btn_save_geotiff.setToolTip("Requires georeferenced satellite imagery. Photos can be saved as PNG/JPG.")
        self.btn_save_all.setToolTip("Run all first. Includes images, maps, scores, arrays and reports; GIS files when available.")
        self.btn_save_view.setToolTip("Download the currently selected view as PNG, including Compare or Map on photo.")

        bottom = QHBoxLayout()
        self.lbl_status = QLabel("Open an image to begin."); self.lbl_status.setObjectName("status")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextFormat(Qt.PlainText)
        self.progress = QProgressBar(); self.progress.setFixedWidth(160); self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        bottom.addWidget(self.lbl_status, 1); bottom.addWidget(self.progress)
        ml.addLayout(bottom)
        outer.addWidget(main, 1)

    # ---------------- drag & drop on window ----------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if os.path.isfile(p):
                self.load_file(p)
                break

    # ---------------- state ----------------
    def _refresh_state(self):
        idle = not self._is_busy
        has_img = self.input_bgrn is not None
        has_sr = self.sr_bgrn is not None
        has_seg = self.seg is not None
        self.chk_cpu.setEnabled(idle and self.gpu_available)
        self.combo_detector.setEnabled(idle)
        for widget in (self.btn_open, self.btn_choose, self.btn_bands, self.btn_models, self.combo_sample, self.btn_details):
            widget.setEnabled(idle)
        self.btn_enhance.setEnabled(idle and has_img)
        self.btn_detect.setEnabled(idle and has_img)
        self.btn_run_all.setEnabled(idle and has_img)
        self.btn_save_img.setEnabled(idle and has_sr)
        self.btn_save_map.setEnabled(idle and has_seg)
        self.btn_save_geotiff.setEnabled(idle and has_sr and self.ref_band.crs is not None)
        self.btn_save_view.setEnabled(idle and has_img)
        self.btn_save_scores.setEnabled(idle and has_seg)
        self.btn_save_report.setEnabled(idle and has_img)
        self.btn_save_all.setEnabled(idle and has_seg)
        self.lbl_download_hint.setText("All results ready." if has_seg else
                                       "Run all to enable maps, scores and the ZIP download." if has_img else
                                       "Choose an image, then press Run all.")
        self.chk_sharp.setEnabled(idle and has_sr)
        self.sld_sharp.setEnabled(idle and has_sr and self.chk_sharp.isChecked())
        available = {"original":has_img, "enhanced":has_sr, "compare":has_sr,
                     "map":has_seg, "overlay":has_seg, "confidence":has_seg}
        for key, button in self.view_btns.items():
            button.setEnabled(idle and available[key])
        self.setAcceptDrops(idle)
        self.viewer.setAcceptDrops(idle)

    def _sync_device_control(self):
        self.chk_cpu.blockSignals(True)
        self.chk_cpu.setChecked(self.device.type == "cpu")
        self.chk_cpu.blockSignals(False)
        if self.device.type == "cpu":
            suffix = "GPU optional." if self.gpu_available else "No CUDA GPU required or available."
            self.lbl_device.setText(f"Running on CPU ({self.cpu_threads} threads). {suffix}")
        else:
            self.lbl_device.setText(f"Running on GPU: {self._gpu_name}")

    def _on_cpu_toggled(self, checked):
        target = torch.device("cpu" if checked else "cuda")
        if self._is_busy or target == self.device:
            self._sync_device_control()
            return False
        if target.type == "cuda" and not self.gpu_available:
            self._sync_device_control()
            return False
        self._pending_device = target

        def operation():
            # Load both models before committing so a driver/checkpoint failure
            # preserves the previous device, models and downloadable results.
            sr_path = os.path.join(self.checkpoint_dir, "rcan.pt")
            seg_path = os.path.join(self.checkpoint_dir, "unet.pt")
            sr = L1BSRSR(sr_path if os.path.isfile(sr_path) else None, str(target))
            seg = self._make_segmenter(str(target), self.checkpoint_dir)
            name = torch.cuda.get_device_name(0) if target.type == "cuda" else None
            return sr, seg, name

        def done(result):
            previous = self.device
            self.sr_engine, self.seg_engine, self._gpu_name = result
            self.device = target
            self._pending_device = None
            self._reset_results()
            self.chk_sharp.setChecked(False)
            self._detect_after_enhance = False
            self._sync_device_control()
            self._refresh_state()
            if self.input_bgrn is not None:
                self._show_view("original")
            self.lbl_status.setText(f"Using {target.type.upper()}. "
                                    "Press Run all to process the selected image." if self.input_bgrn is not None else
                                    f"Using {target.type.upper()}. Open an image to begin.")
            if previous.type == "cuda":
                # Old model tensors have been released; return cached GPU memory.
                gc.collect()
                torch.cuda.empty_cache()

        try:
            return self._start_job(operation, done, f"Switching to {target.type.upper()}...")
        except Exception as error:
            self._job_failed(str(error))
            return False

    def _busy(self, on, text=""):
        previous = self._is_busy
        self._is_busy = bool(on)
        self.progress.setVisible(on)
        if on:
            self.progress.setRange(0, 0)
            self.lbl_status.setText(text)
        if on and not previous:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        elif previous and not on:
            QApplication.restoreOverrideCursor()
        self._refresh_state()

    # ---------------- views ----------------
    def _enhanced_rgb(self) -> Optional[np.ndarray]:
        if self.rgb_sr_base is None:
            return None
        if self.chk_sharp.isChecked():
            pct = self.sld_sharp.value()
            return np.array(Image.fromarray(self.rgb_sr_base).filter(
                ImageFilter.UnsharpMask(radius=2, percent=pct, threshold=3)))
        return self.rgb_sr_base

    def _original_scaled(self) -> np.ndarray:
        """Original photo, enlarged to match the enhanced size so the two compare fairly."""
        if self.rgb_sr_base is not None:
            h, w, _ = self.rgb_sr_base.shape
            return np.array(Image.fromarray(self.rgb_lr).resize((w, h), Image.BICUBIC))
        return self.rgb_lr

    def _sr_label(self) -> str:
        if self.sr_engine is None or not self.sr_engine.weights_loaded:
            return "bicubic baseline (no SR model loaded)"
        return f"RCAN ({self.sr_engine.provenance})"

    def _compare(self):
        """Original | enhanced | change vs bicubic, plus its validity mask."""
        enh = self._enhanced_rgb()
        if enh is None or self.diff_magnitude is None:
            return None
        valid = output_valid_mask(self.ref_band, self.sr_bgrn.shape)
        return compose_compare(self._original_scaled(), enh, difference_rgb(self.diff_magnitude, valid), valid)

    def _view_array(self, key: str):
        arr = None
        if key == "original":
            arr = self.rgb_lr
        elif key == "enhanced":
            arr = self._enhanced_rgb()
        elif key == "compare":
            compare = self._compare()
            if compare is not None:
                arr = compare[0]
        elif key == "confidence" and self.seg is not None:
            arr = score_rgb(self.seg["confidence_map"])
        elif key == "map" and self.seg is not None:
            arr = self.seg["color_map"]
        elif key == "overlay" and self.seg is not None and self.rgb_sr_base is not None:
            arr = (self._enhanced_rgb().astype(np.float32) * 0.45 +
                   self.seg["color_map"].astype(np.float32) * 0.55).astype(np.uint8)
        return arr

    def _show_view(self, key: str):
        arr = self._view_array(key)
        if arr is None:
            return
        self.view_btns[key].setChecked(True)
        self.viewer.set_array(arr)

    def _current_view(self) -> Optional[str]:
        for k, b in self.view_btns.items():
            if b.isChecked():
                return k
        return None

    def _sharp_changed(self):
        self.sld_sharp.setEnabled(not self._is_busy and self.chk_sharp.isChecked() and self.sr_bgrn is not None)
        if self._current_view() in ("enhanced", "compare", "overlay"):
            self._show_view(self._current_view())

    # ---------------- loading ----------------
    def choose_image(self):
        if self._is_busy:
            return
        path = open_image_dialog(self, "Choose an image", os.path.join(APP_DIR, "demo_images"), IMAGE_FILTER)
        if path:
            self.load_file(path)

    def _reset_results(self):
        self.sr_bgrn = self.bicubic_bgrn = self.rgb_sr_base = None
        self.diff_magnitude = self.diff_stats = None
        self.seg = self.metrics = self.conf = None

    def load_file(self, path):
        if self._is_busy:
            return False
        # Decode on the worker; a failed upload keeps the previous valid scene intact.
        def done(result):
            bands, image, rgb, meta = result
            self._set_input(bands["B02"], image, rgb, meta["name"])
            self.lbl_status.setText(f"Opened {meta['name']} ({meta['width']} x {meta['height']} px). {meta['source_kind']}. Press Enhance.")
        return self._start_job(lambda: load_any_user_image(path), done, "Opening image...")

    def _set_input(self, ref: BandData, bgrn: np.ndarray, rgb: np.ndarray, name: str):
        from satsr.models.common import validate_image
        validate_image(bgrn)
        if rgb.dtype != np.uint8 or rgb.shape != (*bgrn.shape[:2], 3) or (ref.height, ref.width) != bgrn.shape[:2]:
            raise ValueError("Input preview and raster metadata do not match the image")
        self._detect_after_enhance = False
        self._reset_results()
        self.chk_sharp.setChecked(False)
        self.ref_band, self.input_bgrn, self.rgb_lr, self.image_name = ref, bgrn, rgb, name
        self.lbl_file.setText(name)
        self.lbl_file.setToolTip(name)
        self._show_view("original")
        self._refresh_state()

    def _on_sample(self, idx: int):
        if self._is_busy:
            return
        try:
            if idx in self._demo_paths:
                self.load_file(self._demo_paths[idx])
        except Exception as error:
            QMessageBox.critical(self, "Sample loading failed", str(error))
        finally:
            self.combo_sample.setCurrentIndex(0)

    def load_real_sample(self):
        if self._is_busy:
            return
        path = next(iter(self._demo_paths.values()), None)
        if path:
            return self.load_file(path)
        QMessageBox.information(self, "Sample missing", "Download the real scenes with download_sentinel_demo.py.")

    # ---------------- processing ----------------
    def choose_bands(self):
        if self._is_busy:
            return
        paths = {}
        for key in ("B02", "B03", "B04", "B08"):
            path = open_image_dialog(self, f"Choose {key}", os.path.join(APP_DIR, "demo_images", "satellite_bands"), "Raster (*.tif *.tiff *.jp2)")
            if not path:
                return
            paths[key] = path
        def done(result):
            ref, image, rgb = result
            self._set_input(ref, image, rgb, "Satellite band stack")
            self.lbl_status.setText("Aligned satellite bands loaded. Press Enhance.")
        self._start_job(lambda: load_band_stack(paths), done, "Opening satellite bands...")

    def choose_models(self):
        if self._is_busy:
            return
        label = ("rcan.pt, landcover_real.pt and hf_segformer/" if self.detector_kind == "ensemble" else
                 "rcan.pt and hf_segformer/" if self.detector_kind == "hf" else
                 "rcan.pt and landcover_real.pt" if self.use_real_detector else "rcan.pt and unet.pt")
        folder = folder_dialog(self, "Select folder containing " + label)
        if not folder:
            return
        def operation():
            return (L1BSRSR(os.path.join(folder, "rcan.pt"), str(self.device)),
                    self._make_segmenter(str(self.device), folder))
        def done(result):
            self.checkpoint_dir = folder
            self.sr_engine, self.seg_engine = result
            self._reset_results()
            self.lbl_model.setText("Enhancement: " + self.sr_engine.provenance + "\nDetection: " + self.seg_engine.provenance)
            self._refresh_state()
            if self.input_bgrn is not None:
                self._show_view("original")
        self._start_job(operation, done, "Loading models...")

    def _ensure_engines(self):
        if self.sr_engine is None:
            path = os.path.join(self.checkpoint_dir, "rcan.pt")
            self.sr_engine = L1BSRSR(path if os.path.isfile(path) else None, str(self.device))
        if self.seg_engine is None:
            self.seg_engine = self._make_segmenter(str(self.device), self.checkpoint_dir)

    def _make_segmenter(self, device, folder):
        if self.detector_kind in ("ensemble", "hf"):
            return LandCoverSegmenter(device, hf_weights_path=os.path.join(folder, "hf_segformer"),
                real_weights_path=os.path.join(folder, "landcover_real.pt") if self.detector_kind == "ensemble" else None)
        if self.use_real_detector:
            # A missing real model is an error; never silently use synthetic weights.
            return LandCoverSegmenter(device, real_weights_path=os.path.join(folder, "landcover_real.pt"))
        path = os.path.join(folder, "unet.pt")
        return LandCoverSegmenter(device, path if os.path.isfile(path) else None)

    def _detector_changed(self, index):
        if self._is_busy:
            return
        self.detector_kind = ("ensemble", "synthetic", "hf", "real")[index]
        self.use_real_detector = index != 1
        self.seg_engine = None
        self.seg = self.conf = None
        self._refresh_state()
        if self.input_bgrn is not None:
            self._show_view("original")
        if hasattr(self, "lbl_status"):
            self.lbl_status.setText("Detection model changed. Press Detect to produce a new map.")

    def _job_failed(self, message):
        switching = self._pending_device is not None
        self._pending_device = None
        self._sync_device_control()
        self._detect_after_enhance = False
        self._busy(False)
        self.lbl_status.setText(f"Device change failed; still using {self.device.type.upper()}. "
                                "Previous results are preserved." if switching else
                                "Operation failed. Select another image or retry.")
        QMessageBox.critical(self, "Processing failed", message)

    def on_enhance(self):
        if self.input_bgrn is None or self._is_busy:
            return
        image = self.input_bgrn
        def operation():
            self._ensure_engines()
            bicubic = self.sr_engine.bicubic_upscale(image)
            sr = self.sr_engine.super_resolve(image)
            valid = output_valid_mask(self.ref_band, sr.shape)
            sr[~valid] = 0
            bicubic[~valid] = 0
            metrics = compute_full_sr_evaluation(image, bicubic, sr, valid_mask=valid)
            magnitude, diff_stats = difference_from_bicubic(sr, bicubic, valid)
            return sr, bicubic, metrics, magnitude, diff_stats
        self._start_job(operation, self._enhance_done, "Running super-resolution...")

    def _enhance_done(self, result):
        sr, bicubic, metrics, magnitude, diff_stats = result
        rgb = render_rgb(sr, self.ref_band)
        self.sr_bgrn, self.bicubic_bgrn, self.metrics, self.rgb_sr_base = sr, bicubic, metrics, rgb
        self.diff_magnitude, self.diff_stats = magnitude, diff_stats
        self.seg = self.conf = None
        self._busy(False)
        h, w = self.sr_bgrn.shape[:2]
        self.lbl_model.setText("Enhancement: " + self.sr_engine.provenance + "\nDetection: " + self.seg_engine.provenance)
        method = "RCAN" if self.sr_engine.weights_loaded else "bicubic baseline"
        self.lbl_status.setText(f"{method}: {w} x {h} px, a derived estimate. Right panel: where it differs from "
                                f"bicubic ({diff_stats['changed_pct']}% of pixels by >0.01 reflectance); "
                                "change is not proof of real detail.")
        self._show_view("compare")
        if self._detect_after_enhance:
            self._detect_after_enhance = False
            self.on_detect()

    def on_detect(self):
        if self.input_bgrn is None or self._is_busy:
            return
        if self.sr_bgrn is None:
            self._detect_after_enhance = True
            self.on_enhance()
            return
        image = self.sr_bgrn
        try:
            area = pixel_area_m2(self.ref_band, image.shape)
            valid = output_valid_mask(self.ref_band, image.shape)
        except ValueError as error:
            self._job_failed(str(error))
            return
        def operation():
            self._ensure_engines()
            if self.seg_engine.real_engine is not None:
                observed = self.input_bgrn
                measured = "estimated NIR" not in self.ref_band.source_kind
                return self.seg_engine.segment(observed, rgb=self.rgb_lr,
                    measured_nir=measured, output_shape=tuple(image.shape[:2]), apply_morphology=False,
                    pixel_size_m=None, pixel_area_m2=area,
                    valid_mask=output_valid_mask(self.ref_band,observed.shape), return_probabilities=False)
            return self.seg_engine.segment(image, apply_morphology=False, pixel_size_m=None,
                pixel_area_m2=area, valid_mask=valid, return_probabilities=False)
        self._start_job(operation, self._detect_done, "Running land-cover segmentation...")

    def _detect_done(self, result):
        self.seg = result
        scores = self.seg["confidence_map"]
        self.conf = round(float(np.nanmean(scores))*100, 1) if np.isfinite(scores).any() else None
        self._busy(False)
        self.lbl_status.setText(f"{result['method']}: land-cover map ready. {result['confidence_note']}.")
        self._show_view("overlay")

    def closeEvent(self, event):
        if self._workers:
            self.lbl_status.setText("Processing is running. Close the window after it finishes.")
            event.ignore()
        else:
            event.accept()

    def show_details(self):
        if self._is_busy:
            return
        stats = self.seg["statistics"] if self.seg else None
        DetailsDialog(self, stats, self.metrics, self.conf).exec_()

    # ---------------- saving ----------------
    def on_save_view(self):
        if self._is_busy or self.input_bgrn is None:
            return
        key = self._current_view()
        arr = self._view_array(key)
        if arr is None:
            return
        path = save_dialog(self, "Download current view", f"{key}.png", "PNG (*.png)")
        if not path:
            return
        try:
            path = resolve_output_path(path, {".png"}, ".png")
            if key == "original":
                valid = self.ref_band.valid_mask
            elif key == "compare":
                arr, valid = self._compare()
            else:
                valid = output_valid_mask(self.ref_band, arr.shape)
            derived = key in ("enhanced", "compare", "overlay")
            save_image(path, arr, valid, description=derived_note(self._sr_label()) if derived else None)
            self.lbl_status.setText(f"Saved {os.path.basename(path)}")
        except Exception as error:
            QMessageBox.critical(self, "Could not save", str(error))

    def on_save_scores(self):
        if self._is_busy or self.seg is None:
            return
        path = save_dialog(self, "Download scores", "scores.png", "PNG (*.png)")
        if path:
            try:
                path = resolve_output_path(path, {".png"}, ".png")
                rgb = score_rgb(self.seg["confidence_map"])
                save_image(path, rgb, output_valid_mask(self.ref_band, rgb.shape))
                self.lbl_status.setText(f"Saved {os.path.basename(path)}")
            except Exception as error:
                QMessageBox.critical(self, "Could not save", str(error))

    def on_save_all(self):
        if self._is_busy or self.seg is None:
            return
        base = os.path.splitext(self.image_name or "image")[0]
        path = save_dialog(self, "Download all results", f"{base}_results.zip", "ZIP archive (*.zip)")
        if not path:
            return
        args = (path, self.rgb_lr, self._enhanced_rgb(), self.sr_bgrn, self.ref_band,
                self.seg, self._report_text(), self._report_data())
        extra = dict(bicubic_bands=self.bicubic_bgrn, sr_model=self._sr_label())
        self._start_job(lambda: export_results_zip(*args, **extra),
                        lambda saved: self.lbl_status.setText(f"Downloaded all results: {saved}"),
                        "Preparing all results for download...")

    def on_save_image(self):
        if self._is_busy:
            return
        arr = self._enhanced_rgb()
        if arr is None:
            return
        base = os.path.splitext(self.image_name or "image")[0]
        path = save_dialog(self, "Save enhanced image", f"{base}_enhanced.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        try:
            path = str(save_image(path, arr, output_valid_mask(self.ref_band, arr.shape),
                                  description=derived_note(self._sr_label())))
            self.lbl_status.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Could not save", str(e))

    def on_save_map(self):
        if self._is_busy:
            return
        if self.seg is None:
            return
        base = os.path.splitext(self.image_name or "image")[0]
        path = save_dialog(self, "Save map", f"{base}_map.png", "PNG (*.png);;GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        try:
            path = str(resolve_output_path(path, {".png", ".tif", ".tiff"}, ".png"))
            if path.lower().endswith((".tif", ".tiff")) and self.ref_band:
                write_single_band_geotiff(path, self.seg["class_map"], self.ref_band)
            else:
                save_image(path, self.seg["color_map"], output_valid_mask(self.ref_band, self.seg["color_map"].shape))
            self.lbl_status.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Could not save", str(e))

    def on_save_geotiff(self):
        if self._is_busy:
            return
        if self.sr_bgrn is None or self.ref_band is None:
            return
        path = save_dialog(self, "Save GeoTIFF", "enhanced.tif", "GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        try:
            write_geotiff(path, self.sr_bgrn, self.ref_band, sr_model=self._sr_label())
            self.lbl_status.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Could not save", str(e))

    def _report_data(self):
        return dict(image=self.image_name, source_kind=self.ref_band.source_kind,
                    source_metadata=self.ref_band.source_metadata or {},
                    sr_model_source=self.sr_engine.provenance if self.sr_engine else "not run",
                    segmentation_model_source=self.seg_engine.provenance if self.seg_engine else "not run",
                    device=str(self.device), metrics=self.metrics,
                    difference_from_bicubic=self.diff_stats,
                    enhanced_imagery=derived_note(self._sr_label()) if self.sr_bgrn is not None else None,
                    statistics=self.seg["statistics"] if self.seg else None,
                    confidence_note=self.seg["confidence_note"] if self.seg else None,
                    detection_details={key:self.seg[key] for key in ("method", "prediction_source", "prediction_input_shape",
                        "measured_nir_used", "spectral_correction_pixels", "abstained_native_pixels", "resolution_note") if key in self.seg} if self.seg else None,
                    preview_sharpening_percent=self.sld_sharp.value() if self.chk_sharp.isChecked() else 0)

    def _report_text(self):
        lines = ["# Image Enhancement Report", "",
                 f"- Image: {self.image_name}"]
        if self.ref_band:
            output_size = f"{self.sr_bgrn.shape[1]} x {self.sr_bgrn.shape[0]} px (2x)" if self.sr_bgrn is not None else "not processed"
            lines.append(f"- Input size: {self.ref_band.width} x {self.ref_band.height} px; "
                         f"output: {output_size}")
            metadata = self.ref_band.source_metadata or {}
            for key, label in (("satellite","Satellite"),("location","Location"),
                               ("acquisition_datetime","Acquisition UTC"),("stac_item_url","Source item"),
                               ("attribution","Attribution")):
                if metadata.get(key):
                    lines.append(f"- {label}: {metadata[key]}")
        lines.append(f"- Device: {str(self.device)}")
        lines += ["", "## Similarity diagnostics (not reconstruction accuracy)"]
        if self.metrics:
            m = self.metrics
            lines += [f"- PSNR: {m['psnr_db']} dB", f"- SSIM: {m['ssim']}",
                      f"- SAM: {m['sam_degrees']} degrees", f"- Edge detail gain: {m['edge_gain_pct']:+}%"]
        else:
            lines.append("- Not run.")
        if self.diff_stats:
            d = self.diff_stats
            lines += [f"- Change vs bicubic: mean {d['mean_reflectance']} reflectance, 95th percentile "
                      f"{d['p95_reflectance']}; {d['changed_pct']}% of pixels changed by more than 0.01",
                      "", "## Enhanced imagery is derived",
                      f"- {derived_note(self._sr_label())}",
                      "- Enhanced PNG/JPEG/GeoTIFF files carry this label in their metadata."]
        lines += ["", "## Model and input provenance",
                  f"- Super-resolution: {self.sr_engine.provenance if self.sr_engine else 'not run'}",
                  f"- Segmentation: {self.seg_engine.provenance if self.seg_engine else 'not run'}",
                  f"- Input: {self.ref_band.source_kind if self.ref_band else 'unknown'}",
                  "- PSNR/SSIM/SAM compare to bicubic; no high-resolution ground truth was supplied.",
                  "- Softmax scores are uncalibrated. Enhanced sampling does not establish real spatial detail.",
                  "- Areas are unavailable for unreferenced images and geographic coordinates."]
        if self.seg and self.seg.get("prediction_source"):
            lines += [f"- Detection input: {self.seg['prediction_source']}",
                      f"- Measured NIR checks: {self.seg['measured_nir_used']}",
                      f"- {self.seg['resolution_note']}",
                      "- Developed space is a separate class; it is not relabeled as roads or buildings."]
        lines += ["", "## Land cover"]
        if self.seg:
            for n, d in self.seg["statistics"].items():
                lines.append(f"- {FRIENDLY.get(n, n)}: {d['percentage']}% ({str(d['area_hectares']) + ' ha' if d['area_hectares'] is not None else 'area unknown'})")
        else:
            lines.append("- Not run.")
        return "\n".join(lines)

    def on_save_report(self):
        if self._is_busy or self.input_bgrn is None:
            return
        path = save_dialog(self, "Save report", "report.md", "Markdown (*.md);;Text (*.txt)")
        if not path:
            return
        try:
            path = resolve_output_path(path, {".md", ".txt"}, ".md")
            save_text(path, self._report_text())
            self.lbl_status.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Could not save", str(e))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
