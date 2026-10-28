"""CPU-only, bounded-memory RCAN inference with the same tile policy as PyTorch."""
from pathlib import Path
import numpy as np
import onnxruntime as ort


def validate_image(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 4 or min(image.shape[:2]) < 1 or image.dtype.kind not in "uif":
        raise ValueError("Expected a non-empty HxWx4 real image")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 65535:
        raise ValueError("Band values must be finite and between 0 and 65535")
    return image


def to_numpy_4ch(image):
    return np.ascontiguousarray(np.moveaxis(validate_image(image).astype(np.float32)/10000., -1, 0)[None])


def from_numpy_to_u16(batch):
    batch = np.asarray(batch)
    if batch.ndim != 4 or batch.shape[:2] != (1, 4) or min(batch.shape[2:]) < 1 or not np.isfinite(batch).all():
        raise ValueError("Invalid ONNX output")
    return np.rint(np.clip(np.moveaxis(batch[0], 0, -1)*10000, 0, 65535)).astype(np.uint16)


class ONNXModel:
    def __init__(self, onnx_path, tile_size=128, halo=40):
        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        if type(tile_size) is not int or tile_size < 1 or type(halo) is not int or halo < 0:
            raise ValueError("Invalid tile configuration")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 4:
            raise ValueError("Expected one float32 NCHW input and one SR output")
        if isinstance(inputs[0].shape[1], int) and inputs[0].shape[1] != 4:
            raise ValueError("Model requires four input bands")
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("normalization") != "10000.0" or metadata.get("band_order") != "B02,B03,B04,B08":
            raise ValueError("Incompatible ONNX model; export with convert_to_onnx.py")
        self.provenance = metadata.get("data_source", "unspecified training data")
        self.input_name = inputs[0].name
        self.tile_size, self.halo = tile_size, halo

    def super_resolve(self, image):
        image = validate_image(image)
        h, w = image.shape[:2]
        result = np.empty((2*h, 2*w, 4), np.uint16)
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                y1, x1 = min(h, y+self.tile_size), min(w, x+self.tile_size)
                top, left = max(0, y-self.halo), max(0, x-self.halo)
                bottom, right = min(h, y1+self.halo), min(w, x1+self.halo)
                tile = image[top:bottom, left:right]
                batch = self.session.run(None, {self.input_name:to_numpy_4ch(tile)})[0]
                if batch.shape != (1, 4, tile.shape[0]*2, tile.shape[1]*2):
                    raise ValueError("ONNX output must be 1x4x2Hx2W")
                out = from_numpy_to_u16(batch)
                sy, sx = (y-top)*2, (x-left)*2
                result[y*2:y1*2, x*2:x1*2] = out[sy:sy+(y1-y)*2, sx:sx+(x1-x)*2]
        return result
