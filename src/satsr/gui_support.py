"""Worker lifecycle and file dialogs shared by both desktop applications."""
from PyQt5.QtCore import QThread, Qt, QDir, QUrl
from PyQt5.QtWidgets import QFileDialog, QApplication
from pathlib import Path


class ProcessingWorker(QThread):
    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation
        self.result = None
        self.error = None
        self.callback = None

    def run(self):
        try:
            self.result = self.operation()
        except Exception as error:
            self.error = str(error) or type(error).__name__
        finally:
            self.operation = None  # Release captured image arrays even before deferred QObject deletion.


class JobMixin:
    def _init_jobs(self):
        self._is_busy = False
        self._workers = []

    def _start_job(self, operation, done, text):
        if self._is_busy:
            return False
        worker = ProcessingWorker(operation, self)
        worker.callback = done
        self._workers.append(worker)
        self._busy(True, text)
        worker.finished.connect(self._job_finished, Qt.QueuedConnection)
        try:
            worker.start()
        except Exception:
            self._workers.remove(worker)
            worker.operation = worker.callback = None
            worker.deleteLater()
            self._busy(False)
            raise
        return True

    def _job_finished(self):
        worker = self.sender()
        if worker not in self._workers:
            return
        self._workers.remove(worker)
        result, error, callback = worker.result, worker.error, worker.callback
        worker.result = worker.callback = None
        worker.deleteLater()
        self._busy(False)
        try:
            if error is not None:
                self._job_failed(error)
            else:
                callback(result)
        except Exception as failure:
            # Presentation/commit failures must never leave the app stuck busy.
            self._job_failed(str(failure) or type(failure).__name__)
        finally:
            result = None


from .paths import PROJECT_ROOT as ROOT


def _existing_folder(*folders):
    for folder in folders:
        if folder is not None:
            path = Path(folder).expanduser().resolve()
            if path.is_dir():
                return path
    return Path.home()


def _file_dialog(parent, title, directory, filters):
    # Explicitly set the directory after construction so restored Qt history cannot
    # leave the chooser at "My Computer". Qt expects normalized path separators.
    dialog = QFileDialog(parent, title)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setNameFilters(filters.split(";;"))
    dialog.setViewMode(QFileDialog.Detail)
    places = []
    for path in (ROOT/"demo_images", Path.home()/"Downloads", Path.home()/"Pictures",
                 directory, ROOT, Path.home()):
        path = path.resolve()
        url = QUrl.fromLocalFile(QDir.fromNativeSeparators(str(path)))
        if path.is_dir() and url not in places:
            places.append(url)
    dialog.setSidebarUrls(places)
    dialog.setDirectory(QDir.fromNativeSeparators(str(directory)))
    dialog.resize(940, 620)
    return dialog


def open_image_dialog(parent, title, folder, filters):
    history = getattr(parent, "_open_dialog_folders", {})
    key = str(Path(folder).resolve())
    directory = _existing_folder(history.get(key), folder, ROOT/"demo_images")
    dialog = _file_dialog(parent, title, directory, filters)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    try:
        if dialog.exec_() == QFileDialog.Accepted and dialog.selectedFiles():
            path = dialog.selectedFiles()[0]
            history[key] = str(Path(path).parent)
            if parent is not None:
                parent._open_dialog_folders = history
            return path
        return ""
    finally:
        dialog.deleteLater()


def save_dialog(parent, title, suggested, filters):
    suggested = Path(suggested)
    directory = _existing_folder(getattr(parent, "_save_dialog_folder", None),
                                 suggested.parent if suggested.is_absolute() else None,
                                 Path.home()/"Downloads", ROOT)
    dialog = _file_dialog(parent, title, directory, filters)
    dialog.setFileMode(QFileDialog.AnyFile)
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    def select_suffix(name_filter):
        extension = name_filter.split("*.", 1)[1].split()[0].rstrip(")") if "*." in name_filter else ""
        dialog.setDefaultSuffix(extension)
    select_suffix(dialog.selectedNameFilter())
    dialog.filterSelected.connect(select_suffix)
    dialog.selectFile(suggested.name)
    try:
        if dialog.exec_() != QFileDialog.Accepted or not dialog.selectedFiles():
            return ""
        path = dialog.selectedFiles()[0]
        selected = dialog.selectedNameFilter()
        if not Path(path).suffix and "*." in selected:
            extension = selected.split("*.", 1)[1].split()[0].rstrip(")")
            path += "."+extension
        if parent is not None:
            parent._save_dialog_folder = str(Path(path).parent)
        return path
    finally:
        dialog.deleteLater()


def folder_dialog(parent, title):
    return QFileDialog.getExistingDirectory(parent, title, options=QFileDialog.DontUseNativeDialog)
