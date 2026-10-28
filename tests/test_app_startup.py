"""Each app must start in a fresh process.

The other GUI tests share one process in which torch/onnxruntime are already loaded,
which hides native-library import-order crashes (e.g. PyQt5 before onnxruntime on Windows).
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Import the app module in a clean interpreter and build its main window, without entering the event loop.
PROBE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("probe_app", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from PyQt5.QtWidgets import QApplication
app = QApplication([])
window = module.{window}()
window.close()
print("started")
"""


class AppStartupTests(unittest.TestCase):
    def start(self, filename, window):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        result = subprocess.run([sys.executable, "-c", PROBE.format(window=window), str(ROOT/filename)],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("started", result.stdout)

    def test_main_app_starts(self):
        self.start("app.py", "MainWindow")

    def test_onnx_app_starts(self):
        self.start("app_onnx.py", "MainWindow")


if __name__ == "__main__":
    unittest.main()
