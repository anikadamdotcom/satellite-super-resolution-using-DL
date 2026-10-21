"""CPU-only desktop super-resolution. No PyTorch import is required."""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLineEdit, QMessageBox,
    QFrame, QScrollArea, QFormLayout, QCheckBox, QSlider)
from geo_io import (BandData, read_band, write_geotiff, save_jpeg_with_jgw, percentile_stretch,
    load_any_user_image, stack_loaded_bands, render_rgb, output_valid_mask, IMAGE_FILTER, save_image, derived_note)
from gui_support import JobMixin, open_image_dialog, save_dialog
from onnx_engine import ONNXModel, to_numpy_4ch, from_numpy_to_u16

APP_DIR = Path(__file__).resolve().parent


def check_alignment(bands):
    try:
        stack_loaded_bands(bands)
        return True, None
    except ValueError as error:
        return False, str(error)


def stack_bgrn(b02, b03, b04, b08):
    return stack_loaded_bands(dict(zip(("B02", "B03", "B04", "B08"), (b02,b03,b04,b08))))[1]


def hwc_to_qpixmap(img):
    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8 or img.ndim not in (2, 3) or img.size == 0:
        raise ValueError("Preview must be a uint8 image")
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimage = QImage(img.data, w, h, img.strides[0], QImage.Format_Grayscale8)
    elif img.shape[2] == 3:
        qimage = QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888)
    else:
        raise ValueError("Preview needs one or three channels")
    return QPixmap.fromImage(qimage.copy())


class ImageLabel(QLabel):
    def __init__(self, title="Preview"):
        super().__init__(title)
        self.setMinimumSize(160, 150)
        self.setAlignment(Qt.AlignCenter)

    def set_image(self, pix):
        self.clear()
        if pix is None:
            self.setText("No preview")
        else:
            self.setPixmap(pix.scaled(220, 170, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class InteractiveImageLabel(QLabel):
    def __init__(self, title="Preview"):
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.pix_before = self.pix_after = None
        self.setMinimumSize(400, 300)

    def set_images(self, before, after):
        self.pix_before, self.pix_after = before, after
        self.set_display_image(after)

    def set_display_image(self, pix):
        self.clear()
        if pix is None:
            self.setText("No result")
        else:
            self.setPixmap(pix.scaled(max(1,self.width()-16), max(1,self.height()-16),
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.set_display_image(self.pix_after)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.set_display_image(self.pix_before)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.set_display_image(self.pix_after)


class MainWindow(QMainWindow, JobMixin):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Super-Resolution (ONNX CPU)")
        self.resize(1100, 850)
        self._init_jobs()
        self.paths = {k:None for k in ("B02", "B03", "B04", "B08")}
        self.bands = self.paths.copy()
        self.ref_band = None
        self.result_u16_hwc = self.pixmap_before = self.rgb_after_u8_base = None
        self.engine = self._engine_key = None
        central = QWidget(); self.setCentralWidget(central); root = QVBoxLayout(central)
        root.addWidget(QLabel("2x RCAN super-resolution | included model trained on synthetic scenes"))
        model_box = QGroupBox("Model"); form = QFormLayout(model_box)
        self.model_edit = QLineEdit(str(APP_DIR/"rcan_model.onnx"))
        self.btn_browse_model = QPushButton("Browse model...")
        row = QHBoxLayout(); row.addWidget(self.model_edit); row.addWidget(self.btn_browse_model)
        form.addRow("ONNX checkpoint", row); root.addWidget(model_box)
        self.btn_open_image = QPushButton("Open image (PNG/JPEG/BMP/WebP/TIFF)...")
        root.addWidget(self.btn_open_image)
        inputs = QGroupBox("Or select four aligned single-band satellite rasters")
        grid = QGridLayout(inputs)
        for i, key in enumerate(self.bands):
            label = ImageLabel(key); button = QPushButton("Select "+key+"...")
            setattr(self, "lab_"+key.lower(), label); setattr(self, "btn_"+key.lower(), button)
            grid.addWidget(label, 0, i); grid.addWidget(button, 1, i)
            button.clicked.connect(lambda _, k=key:self.choose_band(k))
        root.addWidget(inputs)
        self.btn_process = QPushButton("Run 2x super-resolution"); root.addWidget(self.btn_process)
        self.lab_out = InteractiveImageLabel("Run processing to see the output")
        root.addWidget(self.lab_out, 1)
        row = QHBoxLayout()
        self.chk_sharpen = QCheckBox("Preview/JPEG sharpening")
        self.slider_sharpen = QSlider(Qt.Horizontal); self.slider_sharpen.setRange(0,300); self.slider_sharpen.setValue(150)
        self.lbl_sharpen_val = QLabel("")
        self.btn_save_tif = QPushButton("Save GeoTIFF...")
        self.btn_save_image = QPushButton("Save PNG/JPEG...")
        self.btn_save_jpg = QPushButton("Save JPEG + JGW + PRJ...")
        for widget in (self.chk_sharpen, self.slider_sharpen, self.lbl_sharpen_val, self.btn_save_image, self.btn_save_tif, self.btn_save_jpg):
            row.addWidget(widget)
        root.addLayout(row); self.lbl_status = QLabel("Open a demo image or select satellite bands.")
        self.lbl_status.setWordWrap(True); self.lbl_status.setTextFormat(Qt.PlainText); root.addWidget(self.lbl_status)
        self.btn_browse_model.clicked.connect(self.choose_model)
        self.btn_open_image.clicked.connect(self.choose_image)
        self.btn_process.clicked.connect(self.on_process)
        self.btn_save_tif.clicked.connect(self.on_save_tif); self.btn_save_jpg.clicked.connect(self.on_save_jpg)
        self.btn_save_image.clicked.connect(self.on_save_image)
        self.chk_sharpen.toggled.connect(self.apply_and_update_preview); self.slider_sharpen.valueChanged.connect(self.apply_and_update_preview)
        self.model_edit.textChanged.connect(self._model_changed)
        self._refresh_state()

    def _reset_results(self):
        self.ref_band = None
        self.result_u16_hwc = self.pixmap_before = self.rgb_after_u8_base = None
        self.lab_out.set_images(None, None)
        self.chk_sharpen.setChecked(False)
        self._refresh_state()

    def _model_changed(self):
        if not self._is_busy:
            self.engine = self._engine_key = None
            self._reset_results()

    def _refresh_state(self):
        idle = not self._is_busy
        for w in (self.btn_open_image, self.btn_browse_model, self.model_edit, self.btn_b02, self.btn_b03, self.btn_b04, self.btn_b08):
            w.setEnabled(idle)
        self.btn_process.setEnabled(idle and all(b is not None for b in self.bands.values()))
        ready = idle and self.result_u16_hwc is not None
        self.btn_save_tif.setEnabled(ready and self.ref_band.crs is not None)
        self.btn_save_image.setEnabled(ready)
        self.btn_save_jpg.setEnabled(ready and self.ref_band.crs is not None)
        self.chk_sharpen.setEnabled(ready)
        self.slider_sharpen.setEnabled(ready and self.chk_sharpen.isChecked())

    def _busy(self, on, text=""):
        old = self._is_busy; self._is_busy = bool(on)
        if on and not old:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        elif old and not on:
            QApplication.restoreOverrideCursor()
        if text:
            self.lbl_status.setText(text)
        self._refresh_state()

    def _job_failed(self, message):
        self._busy(False)
        self.lbl_status.setText("Operation failed. Select another input or retry.")
        QMessageBox.critical(self, "Operation failed", message)

    def choose_model(self):
        if self._is_busy:
            return
        path = open_image_dialog(self, "Select ONNX model", APP_DIR, "ONNX (*.onnx)")
        if path:
            self.model_edit.setText(path)

    def choose_image(self):
        if not self._is_busy:
            path = open_image_dialog(self, "Choose image", APP_DIR/"demo_images", IMAGE_FILTER)
            if path:
                self.load_file(path)

    def load_file(self, path):
        if self._is_busy:
            return False
        def done(result):
            bands, _, _, meta = result
            self._reset_results()
            self.bands = bands
            self.paths = {k: str(path) for k in bands}
            self._update_band_previews()
            self._refresh_state()
            self.lbl_status.setText("Opened "+meta["name"]+". "+meta["source_kind"])
        return self._start_job(lambda:load_any_user_image(path), done, "Opening image...")

    def _update_band_previews(self):
        for key, band in self.bands.items():
            if band is not None:
                getattr(self, "lab_"+key.lower()).set_image(hwc_to_qpixmap(percentile_stretch(band.arr, valid_mask=band.valid_mask)))

    def choose_band(self, key):
        if self._is_busy:
            return
        path = open_image_dialog(self, "Choose "+key, APP_DIR/"demo_images"/"satellite_bands", "Raster (*.tif *.tiff *.jp2)")
        if not path:
            return
        def done(band):
            self._reset_results()
            self.paths[key], self.bands[key] = path, band
            self._update_band_previews(); self._refresh_state()
        self._start_job(lambda:read_band(path), done, "Opening "+key+"...")

    def ensure_all_inputs(self):
        try:
            return stack_loaded_bands(self.bands)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid satellite inputs", str(error))
            return None

    def on_process(self):
        if self._is_busy:
            return
        loaded = self.ensure_all_inputs()
        if loaded is None:
            return
        ref, image, _ = loaded
        # Photo display bounds must retain the original 8-bit colors.
        if "estimated NIR" in ref.source_kind:
            ref.display_limits = self.bands["B02"].display_limits
        model_path = self.model_edit.text().strip()
        def operation():
            path = Path(model_path)
            key = (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
            if self.engine is None or self._engine_key != key:
                engine = ONNXModel(path)
                self.engine, self._engine_key = engine, key
            output = self.engine.super_resolve(image)
            output[~output_valid_mask(ref, output.shape)] = 0
            return ref, image, output
        self._start_job(operation, self._process_done, "Running ONNX super-resolution...")

    def _process_done(self, result):
        ref, image, output = result
        rgb = render_rgb(output, ref)
        before = render_rgb(image, ref)
        self.ref_band, self.result_u16_hwc, self.rgb_after_u8_base = ref, output, rgb
        h, w = rgb.shape[:2]
        self.pixmap_before = hwc_to_qpixmap(np.asarray(Image.fromarray(before).resize((w,h), Image.Resampling.BICUBIC)))
        self.apply_and_update_preview(); self._refresh_state()
        self.lbl_status.setText("Done. "+self.engine.provenance+". Select another image at any time.")

    def _preview_rgb(self):
        if self.rgb_after_u8_base is None:
            return None
        if self.chk_sharpen.isChecked():
            return np.asarray(Image.fromarray(self.rgb_after_u8_base).filter(ImageFilter.UnsharpMask(radius=2, percent=self.slider_sharpen.value(), threshold=3)))
        return self.rgb_after_u8_base

    def apply_and_update_preview(self):
        self._refresh_state()
        rgb = self._preview_rgb()
        if rgb is not None:
            self.lbl_sharpen_val.setText(str(self.slider_sharpen.value())+"%" if self.chk_sharpen.isChecked() else "")
            self.lab_out.set_images(self.pixmap_before, hwc_to_qpixmap(rgb))

    def on_save_tif(self):
        if self._is_busy or self.result_u16_hwc is None:
            return
        path = save_dialog(self, "Save GeoTIFF", "output_sr.tif", "GeoTIFF (*.tif *.tiff)")
        if path:
            try:
                write_geotiff(path, self.result_u16_hwc, self.ref_band, sr_model="RCAN (ONNX)")
                self.lbl_status.setText("Saved "+path)
            except Exception as error:
                QMessageBox.critical(self, "Save failed", str(error))

    def on_save_jpg(self):
        if self._is_busy or self.result_u16_hwc is None:
            return
        path = save_dialog(self, "Save JPEG + JGW + PRJ", "output_sr.jpg", "JPEG (*.jpg *.jpeg)")
        if path:
            try:
                save_jpeg_with_jgw(path, self._preview_rgb(), self.ref_band, description=derived_note("RCAN (ONNX)"))
                self.lbl_status.setText("Saved JPEG, world file and CRS")
            except Exception as error:
                QMessageBox.critical(self, "Save failed", str(error))

    def on_save_image(self):
        if self._is_busy or self.result_u16_hwc is None:
            return
        path = save_dialog(self, "Save image", "output_sr.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if path:
            try:
                save_image(path, self._preview_rgb(), output_valid_mask(self.ref_band, self.result_u16_hwc.shape),
                           description=derived_note("RCAN (ONNX)"))
                self.lbl_status.setText("Saved "+path)
            except Exception as error:
                QMessageBox.critical(self, "Save failed", str(error))

    def closeEvent(self, event):
        if self._workers:
            self.lbl_status.setText("Wait for processing to finish before closing.")
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv); app.setStyle("Fusion")
    window = MainWindow(); window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
