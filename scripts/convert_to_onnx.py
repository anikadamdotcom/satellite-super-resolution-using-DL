"""Export trained RCAN to ONNX; publish only after graph and numerical validation."""
from _bootstrap import ROOT  # also puts src/ on sys.path
import argparse
from pathlib import Path
import numpy as np
import torch
from satsr.models.rcan import L1BSRSR
from satsr.geo_io import _atomic_write



def export_checkpoint(checkpoint, output):
    engine = L1BSRSR(checkpoint,"cpu")
    output = Path(output)
    if output.resolve() == Path(checkpoint).resolve():
        raise ValueError("ONNX output must not overwrite the checkpoint")
    if output.suffix.lower() != ".onnx":
        raise ValueError("ONNX output requires an .onnx extension")
    output.parent.mkdir(parents=True,exist_ok=True)
    def export(temp):
        import onnx
        import onnxruntime as ort
        sample = torch.rand(1,4,24,32)
        torch.onnx.export(engine.model,sample,str(temp),input_names=["bands"],output_names=["sr"],
                          dynamic_axes={"bands":{2:"height",3:"width"},"sr":{2:"sr_height",3:"sr_width"}},
                          opset_version=17,dynamo=False)
        model = onnx.load(temp)
        for key,value in (("normalization","10000.0"),("band_order","B02,B03,B04,B08"),("data_source",str(engine.provenance))):
            model.metadata_props.add(key=key,value=value)
        onnx.checker.check_model(model); onnx.save(model,temp)
        session = ort.InferenceSession(str(temp),providers=["CPUExecutionProvider"])
        for shape in ((1,4,24,32),(1,4,17,19),(1,4,1,1)):
            x = torch.rand(*shape)
            with torch.inference_mode():
                expected = engine.model(x).numpy()
            actual = session.run(None,{"bands":x.numpy()})[0]
            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
    _atomic_write(output,export)
    print(f"Exported and verified {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",default=str(ROOT/"checkpoints"/"demo"/"rcan.pt"))
    parser.add_argument("--output",default=str(ROOT/"checkpoints"/"demo"/"rcan_model.onnx"))
    args = parser.parse_args()
    export_checkpoint(args.checkpoint,args.output)


if __name__ == "__main__":
    main()
